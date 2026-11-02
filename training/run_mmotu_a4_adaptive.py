from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER, MMOTU_CLASS_SLUGS
from metrics.generation_metrics import build_vgg16_feature_extractor
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.mmotu_a4_common import (
    FeatureDataset, coverage_state, distances, extract_teacher, load_teacher, normalize, radii,
    resolve_teacher_paths, write_json,
)
from training.train_mmotu_lora import PROMPTS
from utils.ultrasound import preprocess_ultrasound_mask, preprocess_ultrasound_pil


NEGATIVE_PROMPT = (
    "color photograph, illustration, cartoon, painting, text-only image, empty frame, duplicated anatomy, "
    "severe geometric distortion, checkerboard artifact, low contrast blank image"
)


@dataclass
class Candidate:
    index: int
    round_index: int
    label: int
    label_name: str
    label_slug: str
    source_image_id: str
    source_path: str
    prior_path: str
    image_path: str
    seed: int


def parse_args():
    parser = argparse.ArgumentParser(description="MMOTU A4 quality-first adaptive diagnosis-aware coverage control")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--a3-lora", default="outputs/mmotu2d/a3_adversarial/online_training/final_lora")
    parser.add_argument("--a3-discriminator", default="outputs/mmotu2d/a3_adversarial/online_training/final_online_discriminator.pt")
    parser.add_argument("--a3-manifest", default="outputs/mmotu2d/a3_adversarial/balanced_train/generated_manifest.csv")
    parser.add_argument("--a3-quality-cache", default="outputs/mmotu2d/a3_adversarial/image_quality_fid_kid_ipr/feature_cache.npz")
    parser.add_argument("--diagnostic-dir", default="outputs/mmotu2d/a4_adaptive/coverage_diagnostic")
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--teacher-root", default="outputs/mmotu2d/convnext_real_baseline/formal")
    parser.add_argument("--teacher-checkpoints", default="", help="Three comma-separated explicit checkpoints")
    parser.add_argument("--teacher-initialization", choices=("scratch", "imagenet"), default="scratch")
    parser.add_argument("--output-dir", default="outputs/mmotu2d/a4_adaptive/controller")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--anchor-strength", type=float, default=0.08)
    parser.add_argument("--gate-power", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--diagnostic-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--quality-quantile", type=float, default=0.10)
    parser.add_argument("--vgg-distance-quantile", type=float, default=0.90)
    parser.add_argument("--base-replacement-fraction", type=float, default=0.04)
    parser.add_argument("--gap-gain", type=float, default=0.25)
    parser.add_argument("--max-replacement-fraction", type=float, default=0.18)
    parser.add_argument("--round-size", type=int, default=24)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-attempt-factor", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def pixel_stats(path: str):
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    histogram = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probability * np.log2(probability)).sum())
    return float(array.std()), entropy, bool(array.std() < 5 or entropy < 1)


class VGGDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths
        self.normal = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as handle:
            image = preprocess_ultrasound_pil(handle.convert("RGB"), 224)
        return self.normal(transforms.functional.to_tensor(image))


@torch.no_grad()
def extract_vgg(paths: List[str], model, device, batch_size: int, num_workers: int):
    loader = DataLoader(VGGDataset(paths), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    values = []
    for images in loader:
        values.append(model(images.to(device, non_blocking=True)).float().cpu().numpy())
    return normalize(np.concatenate(values))


@torch.no_grad()
def discriminator_scores(frame: pd.DataFrame, discriminator, args, device):
    loader = DataLoader(FeatureDataset(frame, args.diagnostic_size), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    realism, structure = [], []
    for images, priors, labels, _indices in loader:
        images = images.to(device, non_blocking=True); priors = priors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = discriminator(images, labels, priors)
        realism.extend(torch.sigmoid(output["realism_logits"]).flatten().cpu().tolist())
        structure.extend(torch.sigmoid(output["structure_logits"]).flatten().cpu().tolist())
    return np.asarray(realism, dtype=np.float32), np.asarray(structure, dtype=np.float32)


def teacher_statistics(teacher_data: Dict[str, Dict[str, np.ndarray]], labels: np.ndarray):
    probabilities = np.stack([data["probabilities"] for data in teacher_data.values()])
    target = np.take_along_axis(probabilities, labels[None, :, None], axis=2).squeeze(2)
    one_hot = np.eye(len(MMOTU_CLASS_ORDER), dtype=bool)[labels]
    other = np.stack([np.where(one_hot, -np.inf, value).max(1) for value in probabilities])
    prediction = probabilities.argmax(2)
    return target.mean(0), (target - other).mean(0), (prediction == labels[None]).mean(0)


def quality_score(target, margin, agreement, realism, structure, nearest_vgg, vgg_scale):
    margin_term = 1 / (1 + np.exp(-np.clip(margin, -20, 20)))
    distance_term = np.exp(-nearest_vgg / max(vgg_scale, 1e-6))
    return np.mean(np.stack([target, margin_term, agreement, realism, structure, distance_term]), axis=0)


def nearest_real_vgg(features, labels, real_features, real_labels):
    result = np.empty(len(features), dtype=np.float32)
    for label in np.unique(labels):
        left = np.flatnonzero(labels == label); right = real_features[real_labels == label]
        result[left] = distances(features[left], right).min(1)
    return result


def prototype_source_multiplier(real_features, synthetic_features, prototype_details, seed):
    k = prototype_details.get("selected_k")
    if not prototype_details.get("stable_multimode") or not k:
        return np.ones(len(real_features), dtype=np.float32)
    model = KMeans(n_clusters=int(k), random_state=seed, n_init=20).fit(real_features)
    real_assignment = model.labels_
    synthetic_assignment = ((synthetic_features[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(2).argmin(1)
    real_p = np.bincount(real_assignment, minlength=int(k)).astype(float); real_p /= real_p.sum()
    synthetic_p = np.bincount(synthetic_assignment, minlength=int(k)).astype(float); synthetic_p /= max(synthetic_p.sum(), 1)
    deficiency = np.maximum(real_p - synthetic_p, 0)
    return (1 + 5 * deficiency[real_assignment]).astype(np.float32)


def generate_candidates(tasks: List[Candidate], pipeline, args, device):
    state = {"source": None, "gate": None}

    def callback(_pipe, _step, _timestep, kwargs):
        kwargs["latents"] = kwargs["latents"] + args.anchor_strength * state["gate"] * (state["source"] - kwargs["latents"])
        return kwargs

    for start in range(0, len(tasks), 4):
        batch = tasks[start:start + 4]
        images, priors = [], []
        for task in batch:
            with Image.open(task.source_path) as handle:
                images.append(preprocess_ultrasound_pil(handle.convert("RGB"), args.image_size))
            with Image.open(task.prior_path) as handle:
                priors.append(transforms.functional.to_tensor(preprocess_ultrasound_mask(handle, args.image_size)))
        prior_tensor = torch.stack(priors).to(device=device, dtype=torch.float16)
        with torch.no_grad():
            image_tensor = pipeline.image_processor.preprocess(images).to(device=device, dtype=torch.float16)
            state["source"] = pipeline.vae.encode(image_tensor).latent_dist.mode() * pipeline.vae.config.scaling_factor
            state["gate"] = F.interpolate(prior_tensor, size=state["source"].shape[-2:], mode="nearest").pow(args.gate_power)
            generated = pipeline(
                prompt=[PROMPTS[task.label_name] for task in batch], negative_prompt=[NEGATIVE_PROMPT] * len(batch),
                image=images, strength=args.strength, guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=[torch.Generator(device="cuda").manual_seed(task.seed) for task in batch],
                callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
            ).images
        for task, image in zip(batch, generated):
            Path(task.image_path).parent.mkdir(parents=True, exist_ok=True); image.save(task.image_path)


def candidate_frame(tasks: List[Candidate]):
    return pd.DataFrame([{
        "image_id": f"{task.source_image_id}__a4__{task.index:05d}", "image_path": task.image_path,
        "label": task.label, "label_name": task.label_name, "label_slug": task.label_slug,
        "source_image_id": task.source_image_id, "source_path": task.source_path,
        "prior_path": task.prior_path, "seed": task.seed, "method": "a4",
        "strength": None, "lora_scale": None, "anchor_strength": None, "gate_power": None,
        "prompt": PROMPTS[task.label_name], "selected": False,
    } for task in tasks])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU A4 controller requires CUDA")
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed); device = torch.device("cuda")
    output = (ROOT / args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").is_file():
        print((output / "completed.json").read_text(encoding="utf-8"), flush=True); return
    diagnostic = json.loads((ROOT / args.diagnostic_dir / "diagnostic_summary.json").read_text(encoding="utf-8"))
    if diagnostic.get("test_accessed"):
        raise RuntimeError("A4 diagnostic accessed test data")
    supported_classes = list(diagnostic.get("supported_classes", []))
    a3 = pd.read_csv(ROOT / args.a3_manifest).reset_index(drop=True)
    train = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    train["source_image_id"] = train.image_id.astype(str); train["prior_path"] = train.mask_path
    expected_synthetic = int(sum(int(train.label.value_counts().max()) - int(value) for value in train.label.value_counts()))
    if train.empty or len(a3) != expected_synthetic:
        raise RuntimeError(
            f"Frozen A3 pool/train protocol mismatch: train={len(train)} A3={len(a3)} expected={expected_synthetic}"
        )

    discriminator = A3DiagnosticDiscriminator(num_classes=len(MMOTU_CLASS_ORDER)).to(device)
    discriminator.load_state_dict(torch.load(ROOT / args.a3_discriminator, map_location="cpu")["model_state"], strict=True)
    discriminator.freeze_all(); discriminator.eval()
    teachers = {}
    real_data, active_data = {}, {}
    cache_dir = (ROOT / args.diagnostic_dir / "cache").resolve()
    teacher_paths = resolve_teacher_paths(ROOT, args.teacher_root, args.teacher_checkpoints)
    for seed, teacher_path in zip((2026, 2027, 2028), teacher_paths):
        name = f"seed_{seed}"; teachers[name] = load_teacher(
            teacher_path, device, args.teacher_initialization == "imagenet"
        )
        real_data[name] = {key: value for key, value in np.load(cache_dir / f"{name}_real.npz", allow_pickle=False).items()}
        active_data[name] = {key: value for key, value in np.load(cache_dir / f"{name}_a3.npz", allow_pickle=False).items()}

    cache = np.load(ROOT / args.a3_quality_cache, allow_pickle=True)
    active_vgg_all = normalize(cache["generated_vgg16"].astype(np.float32))
    real_vgg = normalize(cache["real_vgg16"].astype(np.float32))
    real_vgg_labels = cache["real_labels"].astype(str)
    active_labels_all = a3.label.to_numpy(dtype=np.int64)
    active_label_names_all = a3.label_name.astype(str).to_numpy()
    nearest_vgg_all = nearest_real_vgg(active_vgg_all, active_label_names_all, real_vgg, real_vgg_labels)
    base_teacher_p, base_margin, base_agreement = teacher_statistics(active_data, active_labels_all)
    base_realism, base_structure = discriminator_scores(a3, discriminator, args, device)
    vgg_scale = float(np.median(nearest_vgg_all))
    base_quality_all = quality_score(base_teacher_p, base_margin, base_agreement, base_realism, base_structure, nearest_vgg_all, vgg_scale)

    prototype_payload = json.loads((ROOT / args.diagnostic_dir / "prototype_results.json").read_text(encoding="utf-8"))
    prototype_map = {(row["teacher"], row["class_name"]): row for row in prototype_payload}
    active_records, active_vgg, active_quality, active_teacher_p = {}, {}, {}, {}
    real_features: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in teachers}
    active_features: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in teachers}
    matrices: Dict[str, Dict[str, np.ndarray]] = {}
    radius_map: Dict[str, Dict[str, np.ndarray]] = {}
    real_ids_map, source_ids_map = {}, {}
    thresholds, initial_state, budgets = {}, {}, {}
    for class_name in active_label_names_all:
        if class_name in active_records:
            continue
        mask = active_label_names_all == class_name
        records = a3.loc[mask].to_dict("records")
        for local, record in enumerate(records):
            index = np.flatnonzero(mask)[local]
            record.update({
                "teacher_probability": float(base_teacher_p[index]), "teacher_margin": float(base_margin[index]),
                "teacher_agreement": float(base_agreement[index]), "realism_probability": float(base_realism[index]),
                "structure_probability": float(base_structure[index]), "nearest_real_vgg": float(nearest_vgg_all[index]),
                "quality_score": float(base_quality_all[index]), "origin": "A3", "selected": True,
            })
        active_records[class_name] = records; active_vgg[class_name] = active_vgg_all[mask].copy()
        active_quality[class_name] = base_quality_all[mask].copy(); active_teacher_p[class_name] = base_teacher_p[mask].copy()
        source_ids_map[class_name] = a3.loc[mask, "source_image_id"].astype(str).to_numpy()
        thresholds[class_name] = {
            "teacher_probability": float(np.quantile(base_teacher_p[mask], args.quality_quantile)),
            "teacher_margin": float(np.quantile(base_margin[mask], args.quality_quantile)),
            "teacher_agreement": float(np.quantile(base_agreement[mask], args.quality_quantile)),
            "realism_probability": float(np.quantile(base_realism[mask], args.quality_quantile)),
            "structure_probability": float(np.quantile(base_structure[mask], args.quality_quantile)),
            "nearest_real_vgg_max": float(np.quantile(nearest_vgg_all[mask], args.vgg_distance_quantile)),
        }
        if class_name not in supported_classes:
            budgets[class_name] = 0
            continue
        class_matrices, class_radii = {}, {}
        real_ids = train.loc[train.label_name == class_name, "image_id"].astype(str).to_numpy()
        real_ids_map[class_name] = real_ids
        for teacher_name in teachers:
            rdata, sdata = real_data[teacher_name], active_data[teacher_name]
            rmask = rdata["label_names"].astype(str) == class_name
            smask = sdata["label_names"].astype(str) == class_name
            real_feature = normalize(rdata["local_features"][rmask]); synthetic_feature = normalize(sdata["local_features"][smask])
            real_features[teacher_name][class_name] = real_feature
            active_features[teacher_name][class_name] = synthetic_feature.copy()
            radius = radii(real_feature, args.knn_k); matrix = distances(real_feature, synthetic_feature)
            matrix[real_ids[:, None] == source_ids_map[class_name][None, :]] = np.inf
            class_radii[teacher_name] = radius; class_matrices[teacher_name] = matrix
        matrices[class_name] = class_matrices; radius_map[class_name] = class_radii
        coverage, mean_coverage, excess = coverage_state(class_matrices, class_radii)
        gap = 1 - mean_coverage
        fraction = np.clip(args.base_replacement_fraction + args.gap_gain * gap,
                           args.base_replacement_fraction, args.max_replacement_fraction)
        budget = int(math.ceil(len(records) * fraction))
        budgets[class_name] = budget
        initial_state[class_name] = {"coverage_by_teacher": coverage, "mean_coverage": mean_coverage,
                                     "mean_excess": excess, "gap": gap, "replacement_budget": budget}

    active_classes = [name for name in supported_classes if budgets.get(name, 0) > 0]
    config = {**vars(args), "test_accessed": False, "a3_frozen": True,
              "quality_first": True, "active_classes": active_classes, "budgets": budgets,
              "quality_thresholds": thresholds, "initial_state": initial_state,
              "dynamic_definition": "recompute coverage after every accepted replacement and reprioritize next sources"}
    write_json(output / "run_config.json", config)
    if not active_classes:
        final = a3.copy(); final.to_csv(output / "generated_manifest.csv", index=False)
        completed = {"status": "completed", "adaptive_noop": True, "reason": "no stable train-only deficiency",
                     "synthetic_pool_size": len(final), "test_accessed": False}
        write_json(output / "completed.json", completed); write_json(output / "live_progress.json", completed)
        print(json.dumps(completed, ensure_ascii=False, indent=2)); return

    vgg = build_vgg16_feature_extractor(device)
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        (ROOT / args.model_dir).resolve(), variant="fp16", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True,
    ).to(device)
    pipeline.load_lora_weights((ROOT / args.a3_lora).resolve()); pipeline.fuse_lora(lora_scale=args.lora_scale)
    pipeline.set_progress_bar_config(disable=True)
    candidate_dir = output / "candidates"; candidate_dir.mkdir(parents=True, exist_ok=True)
    accepted = {name: 0 for name in active_classes}; attempts = {name: 0 for name in active_classes}
    audit, all_candidates = [], []; candidate_index = 0; stale_rounds = 0; started = time.time()
    for round_index in range(1, args.max_rounds + 1):
        remaining = {name: min(budgets[name] - accepted[name], int(math.ceil(args.max_attempt_factor * budgets[name])) - attempts[name]) for name in active_classes}
        eligible = [name for name in active_classes if remaining[name] > 0]
        if not eligible or stale_rounds >= 3:
            break
        tasks: List[Candidate] = []; allocations = {name: 0 for name in active_classes}
        for slot in range(args.round_size):
            choices = [name for name in eligible if allocations[name] < remaining[name]]
            if not choices:
                break
            class_name = choices[slot % len(choices)]
            ratios = []
            for teacher_name in teachers:
                ratios.append(matrices[class_name][teacher_name].min(1) / np.maximum(radius_map[class_name][teacher_name], 1e-8))
            source_weight = np.maximum(np.median(np.stack(ratios), axis=0) - 0.85, 0.05) ** 2
            for teacher_name in teachers:
                source_weight *= prototype_source_multiplier(
                    real_features[teacher_name][class_name], active_features[teacher_name][class_name],
                    prototype_map.get((teacher_name, class_name), {}), args.seed,
                )
            source_weight /= source_weight.sum()
            source_local = int(rng.choice(len(source_weight), p=source_weight))
            source_id = real_ids_map[class_name][source_local]
            source = train[train.image_id.astype(str) == source_id].iloc[0]
            label = int(source.label); path = candidate_dir / MMOTU_CLASS_SLUGS[label] / f"{source_id}__a4__{candidate_index:05d}.png"
            tasks.append(Candidate(candidate_index, round_index, label, class_name, MMOTU_CLASS_SLUGS[label],
                                   source_id, str(source.image_path), str(source.mask_path), str(path.resolve()),
                                   940000 + args.seed + candidate_index))
            candidate_index += 1; allocations[class_name] += 1; attempts[class_name] += 1
        write_json(output / "live_progress.json", {"status": "generating", "round": round_index,
                   "tasks": len(tasks), "attempts": attempts, "accepted": accepted, "budgets": budgets})
        generate_candidates(tasks, pipeline, args, device)
        candidates = candidate_frame(tasks); candidates["strength"] = args.strength; candidates["lora_scale"] = args.lora_scale
        candidates["anchor_strength"] = args.anchor_strength; candidates["gate_power"] = args.gate_power
        medical = {}
        for teacher_name, teacher in teachers.items():
            medical[teacher_name] = extract_teacher(
                teacher, candidates, args.diagnostic_size, args.batch_size, args.num_workers, device,
                output / "round_cache" / f"round_{round_index}_{teacher_name}.npz", f"round{round_index} {teacher_name}",
            )
        labels = candidates.label.to_numpy(dtype=np.int64)
        c_target, c_margin, c_agreement = teacher_statistics(medical, labels)
        c_realism, c_structure = discriminator_scores(candidates, discriminator, args, device)
        c_vgg = extract_vgg(candidates.image_path.astype(str).tolist(), vgg, device, args.batch_size, args.num_workers)
        c_nearest = nearest_real_vgg(c_vgg, candidates.label_name.astype(str).to_numpy(), real_vgg, real_vgg_labels)
        c_quality = quality_score(c_target, c_margin, c_agreement, c_realism, c_structure, c_nearest, vgg_scale)
        round_accepted = 0
        for local, task in enumerate(tasks):
            class_name = task.label_name; threshold = thresholds[class_name]
            std, entropy, near_constant = pixel_stats(task.image_path)
            qpass = bool(
                not near_constant and c_target[local] >= threshold["teacher_probability"]
                and c_margin[local] >= threshold["teacher_margin"]
                and c_agreement[local] >= threshold["teacher_agreement"]
                and c_realism[local] >= threshold["realism_probability"]
                and c_structure[local] >= threshold["structure_probability"]
                and c_nearest[local] <= threshold["nearest_real_vgg_max"]
            )
            before_by, before_mean, before_excess = coverage_state(matrices[class_name], radius_map[class_name])
            row = candidates.iloc[local].to_dict(); row.update({
                "teacher_probability": float(c_target[local]), "teacher_margin": float(c_margin[local]),
                "teacher_agreement": float(c_agreement[local]), "realism_probability": float(c_realism[local]),
                "structure_probability": float(c_structure[local]), "nearest_real_vgg": float(c_nearest[local]),
                "quality_score": float(c_quality[local]), "pixel_std": std, "entropy": entropy,
                "near_constant": near_constant, "quality_gate_pass": qpass, "origin": "A4_candidate", "selected": False,
            })
            accepted_now = False; after_mean, after_excess = before_mean, before_excess
            if qpass and accepted[class_name] < budgets[class_name]:
                critical_sets = [set(matrix.argmin(1).tolist()) for matrix in matrices[class_name].values()]
                removable = [i for i in range(len(active_records[class_name])) if sum(i in values for values in critical_sets) <= 1]
                if not removable:
                    removable = list(np.argsort(active_quality[class_name])[:max(4, len(active_quality[class_name]) // 5)])
                remove_index = min(removable, key=lambda i: float(0.7 * active_quality[class_name][i] + 0.3 * active_teacher_p[class_name][i]))
                proposed = {}
                for teacher_name in teachers:
                    rfeature = real_features[teacher_name][class_name]
                    candidate_feature = normalize(medical[teacher_name]["local_features"])[local]
                    column = distances(rfeature, candidate_feature[None]).flatten()
                    column[real_ids_map[class_name] == task.source_image_id] = np.inf
                    matrix = matrices[class_name][teacher_name].copy(); matrix[:, remove_index] = column
                    proposed[teacher_name] = matrix
                _after_by, after_mean, after_excess = coverage_state(proposed, radius_map[class_name])
                utility = after_mean > before_mean + 1e-12 or (abs(after_mean - before_mean) <= 1e-12 and after_excess < before_excess - 1e-5)
                quality_floor = c_quality[local] >= active_quality[class_name][remove_index] - 0.02
                if utility and quality_floor:
                    removed = active_records[class_name][remove_index]
                    row.update({"selected": True, "selection_rule": "quality gates passed; consensus local/prototype coverage improved",
                                "replaced_image_path": str(removed["image_path"]), "replaced_image_id": str(removed["image_id"])})
                    active_records[class_name][remove_index] = dict(row); active_vgg[class_name][remove_index] = c_vgg[local]
                    active_quality[class_name][remove_index] = c_quality[local]; active_teacher_p[class_name][remove_index] = c_target[local]
                    for teacher_name in teachers:
                        active_features[teacher_name][class_name][remove_index] = normalize(medical[teacher_name]["local_features"])[local]
                        matrices[class_name][teacher_name] = proposed[teacher_name]
                    accepted[class_name] += 1; round_accepted += 1; accepted_now = True
                else:
                    row["selection_rule"] = "rejected: no coverage utility or quality floor"
            else:
                row["selection_rule"] = "rejected by quality gate or completed budget"
            row.update({"coverage_before": before_mean, "coverage_after": after_mean,
                        "excess_before": before_excess, "excess_after": after_excess})
            all_candidates.append(dict(row)); audit.append({
                "round": round_index, "candidate_index": task.index, "class_name": class_name,
                "source_image_id": task.source_image_id, "quality_gate_pass": qpass, "selected": accepted_now,
                "quality_score": float(c_quality[local]), "coverage_before": before_mean,
                "coverage_after": after_mean, "excess_before": before_excess, "excess_after": after_excess,
                "reason": row["selection_rule"],
            })
        stale_rounds = stale_rounds + 1 if round_accepted == 0 else 0
        pd.DataFrame(audit).to_csv(output / "adaptive_round_audit.csv", index=False)
        progress = {"status": "adaptive_update", "round": round_index, "round_candidates": len(tasks),
                    "round_accepted": round_accepted, "attempts": attempts, "accepted": accepted,
                    "budgets": budgets, "stale_rounds": stale_rounds,
                    "coverage": {name: coverage_state(matrices[name], radius_map[name])[:2] for name in active_classes},
                    "elapsed_minutes": (time.time() - started) / 60}
        write_json(output / "live_progress.json", progress)
        print(f"[MMOTU A4] round={round_index} candidates={len(tasks)} accepted={round_accepted} total={sum(accepted.values())}", flush=True)

    final_records = []
    for class_name in active_records:
        final_records.extend(active_records[class_name])
    final = pd.DataFrame(final_records).sort_values(["label", "image_id"]).reset_index(drop=True)
    if len(final) != len(a3) or final.groupby("label").size().to_dict() != a3.groupby("label").size().to_dict():
        raise RuntimeError("A4 changed the fixed 100% pool size or class budget")
    final.to_csv(output / "generated_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_candidates).to_csv(output / "all_candidates_manifest.csv", index=False)
    final_state = {name: {"coverage_by_teacher": coverage_state(matrices[name], radius_map[name])[0],
                          "mean_coverage": coverage_state(matrices[name], radius_map[name])[1],
                          "mean_excess": coverage_state(matrices[name], radius_map[name])[2]}
                   for name in active_classes}
    completed = {"status": "completed", "method": "A4 adaptive diagnosis-aware quality-diversity controller",
                 "test_accessed": False, "active_classes": active_classes, "accepted": accepted,
                 "budgets": budgets, "initial_state": initial_state, "final_state": final_state,
                 "candidate_count": len(all_candidates), "synthetic_pool_size": len(final),
                 "elapsed_minutes": (time.time() - started) / 60}
    write_json(output / "completed.json", completed); write_json(output / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
