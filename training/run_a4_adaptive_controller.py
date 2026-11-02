from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from metrics.generation_metrics import build_vgg16_feature_extractor
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.generate_aptos_sd025_balanced import NEGATIVE_PROMPT, PROMPTS
from training.generate_aptos_structure_guided import prepared_image, prior_tensor
from training.run_aptos_convnext_baselines import build_transform, seed_everything
from utils.fundus import preprocess_fundus_pil
from utils.image_quality import compute_pixel_stats


MINORITY = APTOS_CLASS_ORDER[1:]
ANCHOR_STRENGTH = 0.08
GATE_POWER = 0.5
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4 adaptive local-prototype coverage controller")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument(
        "--a3-lora", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/full_training/final_lora"
    )
    parser.add_argument(
        "--a3-root", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/balanced_train"
    )
    parser.add_argument(
        "--a3-quality-cache",
        default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/image_quality_fid_kid_ipr/feature_cache.npz",
    )
    parser.add_argument(
        "--discriminator",
        default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/full_training/final_online_discriminator.pt",
    )
    parser.add_argument(
        "--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv"
    )
    parser.add_argument("--output-dir", default="outputs/aptos2019/reproductions/a4_helper/balanced_train")
    parser.add_argument("--round-size", type=int, default=64)
    parser.add_argument("--max-attempt-factor", type=float, default=2.5)
    parser.add_argument("--base-replacement-fraction", type=float, default=0.10)
    parser.add_argument("--coverage-target", type=float, default=0.60)
    parser.add_argument("--coverage_gain", type=float, default=0.30)
    parser.add_argument("--max-replacement-fraction", type=float, default=0.25)
    parser.add_argument("--quality-quantile", type=float, default=0.10)
    parser.add_argument("--distance-quantile", type=float, default=0.90)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--score-image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rounds", type=int, default=0, help="Development smoke only; zero runs to stop rule")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return (values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def pairwise_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(left @ right.T, -1.0, 1.0))).astype(np.float32)


def knn_radius(values: np.ndarray, k: int = 3) -> np.ndarray:
    distance = pairwise_distance(values, values)
    np.fill_diagonal(distance, np.inf)
    return np.partition(distance, kth=k - 1, axis=1)[:, k - 1]


class ScoreDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.medical = build_transform(image_size, traditional=False)
        self.vgg = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(VGG_MEAN, VGG_STD),
            ]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        raw = Image.open(str(row.image_path)).convert("RGB")
        roi = preprocess_fundus_pil(raw, crop=True, mask_background=True).image
        prior = Image.open(str(row.prior_path)).convert("L").resize(
            (self.medical.transforms[1].size[0], self.medical.transforms[1].size[1]), Image.Resampling.BILINEAR
        )
        prior_array = np.asarray(prior, dtype=np.float32) / 255.0
        return (
            self.medical(roi),
            self.vgg(roi),
            torch.from_numpy(prior_array).unsqueeze(0),
            int(row.label),
            int(index),
        )


@torch.no_grad()
def score_frame(
    frame: pd.DataFrame,
    discriminator: A3DiagnosticDiscriminator,
    vgg: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray]:
    loader = DataLoader(
        ScoreDataset(frame, args.score_image_size),
        batch_size=args.score_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    rows: List[Dict[str, object]] = []
    vgg_features: List[np.ndarray] = []
    discriminator.eval()
    vgg.eval()
    for medical, vgg_input, prior, labels, indices in loader:
        medical = medical.to(device, non_blocking=True)
        vgg_input = vgg_input.to(device, non_blocking=True)
        prior = prior.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=True):
            mid, final_feature, logits = discriminator.encode(medical)
            class_probability = torch.softmax(logits.float(), dim=1).gather(1, labels[:, None]).flatten()
            realism = torch.sigmoid(discriminator.realism_from_features(final_feature, labels).float()).flatten()
            structure = torch.sigmoid(discriminator.structure_from_features(mid, prior).float()).flatten()
        features = vgg(vgg_input).float()
        features = F.normalize(features, dim=1).cpu().numpy().astype(np.float32)
        vgg_features.append(features)
        for local, frame_index in enumerate(indices.tolist()):
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "class_probability": float(class_probability[local].cpu()),
                    "realism_probability": float(realism[local].cpu()),
                    "structure_probability": float(structure[local].cpu()),
                    "quality_score": float(
                        0.40 * class_probability[local].cpu()
                        + 0.30 * realism[local].cpu()
                        + 0.30 * structure[local].cpu()
                    ),
                }
            )
    scored = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    return scored, np.concatenate(vgg_features, axis=0)


@dataclass(frozen=True)
class CandidateTask:
    candidate_index: int
    round_index: int
    label: int
    label_name: str
    source_image_id: str
    source_path: str
    prior_path: str
    output_path: str
    seed: int


def generate_round(
    tasks: Sequence[CandidateTask], pipeline: StableDiffusionImg2ImgPipeline, args: argparse.Namespace, output_dir: Path
) -> None:
    state: Dict[str, torch.Tensor | None] = {"source_latents": None, "gate": None}

    def anchor_callback(pipe, step_index, timestep, callback_kwargs):
        if state["source_latents"] is None or state["gate"] is None:
            raise RuntimeError("A4 anchor callback has no source state")
        latents = callback_kwargs["latents"]
        callback_kwargs["latents"] = latents + ANCHOR_STRENGTH * state["gate"] * (state["source_latents"] - latents)
        return callback_kwargs

    missing = [task for task in tasks if not Path(task.output_path).is_file()]
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        source_images = [prepared_image(task.source_path, args.image_size) for task in batch]
        priors = torch.stack([prior_tensor(task.prior_path, args.image_size, torch.device("cuda")) for task in batch]).half()
        with torch.no_grad():
            source_tensor = pipeline.image_processor.preprocess(source_images).to(device="cuda", dtype=torch.float16)
            source_latents = pipeline.vae.encode(source_tensor).latent_dist.mode() * pipeline.vae.config.scaling_factor
            state["source_latents"] = source_latents
            state["gate"] = F.interpolate(priors, size=source_latents.shape[-2:], mode="bilinear", align_corners=False).sqrt()
            images = pipeline(
                prompt=[PROMPTS[task.label_name] for task in batch],
                negative_prompt=[NEGATIVE_PROMPT] * len(batch),
                image=source_images,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=[torch.Generator(device="cuda").manual_seed(task.seed) for task in batch],
                callback_on_step_end=anchor_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images
        for task, image in zip(batch, images):
            image.save(task.output_path)
    state["source_latents"] = None
    state["gate"] = None


def make_candidate_frame(tasks: Sequence[CandidateTask]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **task.__dict__,
                "image_id": f"{task.source_image_id}__a4_adaptive__{task.candidate_index:05d}",
                "image_path": task.output_path,
                "dataset_name": "aptos2019_a4_adaptive_candidate_train_only",
            }
            for task in tasks
        ]
    )


def quality_thresholds(
    base: pd.DataFrame,
    features: np.ndarray,
    real_features: np.ndarray,
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    for class_name in MINORITY:
        mask = generated_labels == class_name
        real = real_features[real_labels == class_name]
        generated = features[mask]
        nearest = pairwise_distance(generated, real).min(axis=1)
        values = base.loc[mask].reset_index(drop=True)
        thresholds[class_name] = {
            "class_probability": float(values.class_probability.quantile(args.quality_quantile)),
            "realism_probability": float(values.realism_probability.quantile(args.quality_quantile)),
            "structure_probability": float(values.structure_probability.quantile(args.quality_quantile)),
            "nearest_real_distance": float(np.quantile(nearest, args.distance_quantile)),
        }
    return thresholds


def candidate_passes(row: pd.Series, nearest_real: float, threshold: Dict[str, float]) -> bool:
    return bool(
        row.class_probability >= threshold["class_probability"]
        and row.realism_probability >= threshold["realism_probability"]
        and row.structure_probability >= threshold["structure_probability"]
        and nearest_real <= threshold["nearest_real_distance"]
        and bool(row.valid_fundus)
        and not bool(row.near_constant)
    )


def current_coverage(distance: np.ndarray, radius: np.ndarray) -> Tuple[np.ndarray, float]:
    nearest = distance.min(axis=1)
    return nearest, float(np.mean(nearest <= radius))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4 controller requires CUDA")
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    if (output_dir.parent / "FROZEN_A4.json").is_file():
        raise RuntimeError("A4 is frozen; reproduction must use a new output root")
    completed_path = output_dir / "completed.json"
    if completed_path.is_file():
        print(completed_path.read_text(encoding="utf-8"), flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    a3_root = (ROOT / args.a3_root).resolve()
    a3 = pd.read_csv(a3_root / "generated_manifest.csv").reset_index(drop=True)
    train = pd.read_csv(a3_root / "splits/train.csv")
    priors = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(priors.image_id.astype(str), priors.prior_path.astype(str)))
    a3["prior_path"] = a3.source_image_id.astype(str).map(prior_map)
    if len(a3) != 3752 or a3.prior_path.isna().any():
        raise RuntimeError("A3 manifest or prior mapping is incomplete")
    for split in ["train", "val", "test"]:
        (output_dir / "splits").mkdir(parents=True, exist_ok=True)
        shutil.copy2(a3_root / "splits" / f"{split}.csv", output_dir / "splits" / f"{split}.csv")

    cache = np.load(ROOT / args.a3_quality_cache, allow_pickle=True)
    real_features = l2_normalize(cache["real_vgg16"])
    a3_features = l2_normalize(cache["generated_vgg16"])
    real_labels = cache["real_labels"].astype(str)
    generated_labels = cache["generated_labels"].astype(str)
    if cache["generated_paths"].astype(str).tolist() != a3.image_path.astype(str).tolist():
        raise RuntimeError("Frozen A3 VGG cache and manifest order differ")
    real_path_to_index = {str(path): index for index, path in enumerate(cache["real_paths"].astype(str))}

    discriminator = A3DiagnosticDiscriminator(num_classes=5).to(device)
    d_checkpoint = torch.load(ROOT / args.discriminator, map_location="cpu")
    if d_checkpoint.get("pretrained") is not False:
        raise RuntimeError("A3 discriminator must be weights=None")
    discriminator.load_state_dict(d_checkpoint["model_state"], strict=True)
    discriminator.freeze_all()
    discriminator.eval()
    vgg = build_vgg16_feature_extractor(device)
    write_json(output_dir / "live_progress.json", {"status": "scoring_a3", "completed": 0, "total": len(a3)})
    base_scores, scored_a3_features = score_frame(a3, discriminator, vgg, args, device)
    max_feature_delta = float(np.abs(scored_a3_features - a3_features).max())
    if max_feature_delta > 5e-3:
        raise RuntimeError(f"A3 VGG feature reproduction drift is too large: {max_feature_delta}")
    thresholds = quality_thresholds(
        base_scores, a3_features, real_features, real_labels, generated_labels, args
    )
    active_records: Dict[str, List[Dict[str, object]]] = {}
    active_features: Dict[str, np.ndarray] = {}
    active_scores: Dict[str, np.ndarray] = {}
    distances: Dict[str, np.ndarray] = {}
    radii: Dict[str, np.ndarray] = {}
    initial_coverage: Dict[str, float] = {}
    budgets: Dict[str, int] = {}
    accepted: Dict[str, int] = {name: 0 for name in MINORITY}
    attempts: Dict[str, int] = {name: 0 for name in MINORITY}
    for class_name in MINORITY:
        gmask = generated_labels == class_name
        rmask = real_labels == class_name
        records = a3.loc[gmask].to_dict("records")
        for index, record in enumerate(records):
            record.update(base_scores.loc[np.flatnonzero(gmask)[index]].to_dict())
            record["origin"] = "A3"
            record["selected"] = True
            record["selection_rule"] = "A3 initial pool"
        active_records[class_name] = records
        active_features[class_name] = a3_features[gmask].copy()
        active_scores[class_name] = base_scores.loc[gmask, "quality_score"].to_numpy(dtype=np.float32)
        real = real_features[rmask]
        radii[class_name] = knn_radius(real, k=3)
        distances[class_name] = pairwise_distance(real, active_features[class_name])
        _, coverage = current_coverage(distances[class_name], radii[class_name])
        initial_coverage[class_name] = coverage
        fraction = np.clip(
            args.base_replacement_fraction + args.coverage_gain * max(0.0, args.coverage_target - coverage),
            args.base_replacement_fraction,
            args.max_replacement_fraction,
        )
        budgets[class_name] = int(math.ceil(len(records) * fraction))
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "method": "A3 pool -> local VGG prototype deficiency -> A3 quality gate -> adaptive replacement -> recompute",
            "test_accessed": False,
            "a3_frozen": True,
            "initial_local_coverage": initial_coverage,
            "replacement_budgets": budgets,
            "quality_thresholds": thresholds,
            "a3_cache_reproduction_max_abs_delta": max_feature_delta,
        },
    )

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        (ROOT / args.model_dir).resolve(),
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(device)
    pipeline.load_lora_weights((ROOT / args.a3_lora).resolve())
    pipeline.fuse_lora(lora_scale=args.lora_scale)
    pipeline.set_progress_bar_config(disable=True)
    started = time.time()
    all_candidate_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    candidate_index = 0
    round_index = 0
    stale_rounds = 0
    while True:
        if args.max_rounds > 0 and round_index >= args.max_rounds:
            break
        remaining = {
            name: min(
                budgets[name] - accepted[name],
                int(math.ceil(args.max_attempt_factor * budgets[name])) - attempts[name],
            )
            for name in MINORITY
        }
        if max(remaining.values()) <= 0 or stale_rounds >= 3:
            break
        round_index += 1
        tasks: List[CandidateTask] = []
        eligible = [name for name in MINORITY if remaining[name] > 0]
        allocations = {name: 0 for name in MINORITY}
        for slot in range(args.round_size):
            choices = [name for name in eligible if allocations[name] < remaining[name]]
            if not choices:
                break
            class_name = choices[slot % len(choices)]
            rmask = real_labels == class_name
            real = real_features[rmask]
            nearest, _ = current_coverage(distances[class_name], radii[class_name])
            deficiency = nearest / np.maximum(radii[class_name], 1e-8)
            weights = np.maximum(deficiency - 0.85, 0.05) ** 2
            source_local = int(rng.choice(len(real), p=weights / weights.sum()))
            source_path = str(cache["real_paths"].astype(str)[np.flatnonzero(rmask)[source_local]])
            source_row = train.loc[train.image_path.astype(str) == source_path]
            if len(source_row) != 1:
                raise RuntimeError(f"Could not map selected prototype source: {source_path}")
            source = source_row.iloc[0]
            class_dir = candidate_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            path = class_dir / f"{source.image_id}__a4_adaptive__{candidate_index:05d}.png"
            tasks.append(
                CandidateTask(
                    candidate_index=candidate_index,
                    round_index=round_index,
                    label=int(source.label),
                    label_name=class_name,
                    source_image_id=str(source.image_id),
                    source_path=str(source.image_path),
                    prior_path=str(prior_map[str(source.image_id)]),
                    output_path=str(path.resolve()),
                    seed=900000 + args.seed + candidate_index,
                )
            )
            allocations[class_name] += 1
            attempts[class_name] += 1
            candidate_index += 1
        if not tasks:
            break
        write_json(
            output_dir / "live_progress.json",
            {
                "status": "generating_candidates",
                "round": round_index,
                "round_candidates": len(tasks),
                "attempts": attempts,
                "accepted": accepted,
                "budgets": budgets,
                "coverage": {name: current_coverage(distances[name], radii[name])[1] for name in MINORITY},
            },
        )
        generate_round(tasks, pipeline, args, output_dir)
        candidate_frame = make_candidate_frame(tasks)
        candidate_scores, candidate_features = score_frame(candidate_frame, discriminator, vgg, args, device)
        round_accepts = 0
        for local_index, task in enumerate(tasks):
            class_name = task.label_name
            row = candidate_frame.iloc[local_index].to_dict()
            score = candidate_scores.iloc[local_index]
            image = np.asarray(Image.open(task.output_path).convert("RGB"), dtype=np.uint8)
            pixel = compute_pixel_stats(image)
            row.update(score.to_dict())
            row.update(
                {
                    "pixel_std": float(pixel["std"]),
                    "entropy": float(pixel["entropy"]),
                    "fundus_area_ratio": float(pixel["fundus_area_ratio"]),
                    "valid_fundus": bool(pixel["is_valid_fundus"]),
                    "near_constant": bool(pixel["is_near_constant"]),
                    "selected": False,
                    "origin": "A4_candidate",
                }
            )
            rmask = real_labels == class_name
            real = real_features[rmask]
            candidate_feature = candidate_features[local_index]
            candidate_distance = pairwise_distance(real, candidate_feature[None, :]).flatten()
            nearest_real = float(candidate_distance.min())
            row["nearest_real_distance"] = nearest_real
            threshold = thresholds[class_name]
            quality_pass = candidate_passes(pd.Series(row), nearest_real, threshold)
            row["quality_gate_pass"] = quality_pass
            before_nearest, before_coverage = current_coverage(distances[class_name], radii[class_name])
            if quality_pass and accepted[class_name] < budgets[class_name]:
                nearest_owner = distances[class_name].argmin(axis=1)
                critical = set(nearest_owner.tolist())
                noncritical = [index for index in range(len(active_records[class_name])) if index not in critical]
                if noncritical:
                    remove_index = min(noncritical, key=lambda index: float(active_scores[class_name][index]))
                else:
                    # Rare case: evaluate the least damaging removals among the lowest-quality quartile.
                    ordering = np.argsort(active_scores[class_name])[: max(16, len(active_scores[class_name]) // 4)]
                    best_key = None
                    remove_index = int(ordering[0])
                    for trial in ordering:
                        trial_distance = distances[class_name].copy()
                        trial_distance[:, int(trial)] = candidate_distance
                        _, trial_coverage = current_coverage(trial_distance, radii[class_name])
                        key = (trial_coverage, -float(active_scores[class_name][int(trial)]))
                        if best_key is None or key > best_key:
                            best_key = key
                            remove_index = int(trial)
                proposed = distances[class_name].copy()
                proposed[:, remove_index] = candidate_distance
                _, after_coverage = current_coverage(proposed, radii[class_name])
                removed_quality = float(active_scores[class_name][remove_index])
                accept = after_coverage > before_coverage + 1e-12 or (
                    abs(after_coverage - before_coverage) <= 1e-12
                    and float(row["quality_score"]) >= removed_quality + 0.02
                )
                if accept:
                    removed = active_records[class_name][remove_index]
                    row["selected"] = True
                    row["selection_rule"] = "adaptive local-coverage gain or quality-preserving replacement"
                    row["replaced_image_path"] = str(removed["image_path"])
                    row["coverage_before"] = before_coverage
                    row["coverage_after"] = after_coverage
                    removed["selected"] = False
                    removed["selection_rule"] = f"replaced by {row['image_id']} under adaptive controller"
                    active_records[class_name][remove_index] = dict(row)
                    active_features[class_name][remove_index] = candidate_feature
                    active_scores[class_name][remove_index] = float(row["quality_score"])
                    distances[class_name] = proposed
                    accepted[class_name] += 1
                    round_accepts += 1
                else:
                    row["selection_rule"] = "rejected: no coverage gain and insufficient quality gain"
                    row["coverage_before"] = before_coverage
                    row["coverage_after"] = after_coverage
            else:
                row["selection_rule"] = "rejected by quality-first gate or class budget"
                row["coverage_before"] = before_coverage
                row["coverage_after"] = before_coverage
            all_candidate_rows.append(dict(row))
            audit_rows.append(
                {
                    "round": round_index,
                    "candidate_index": task.candidate_index,
                    "class_name": class_name,
                    "source_image_id": task.source_image_id,
                    "quality_gate_pass": quality_pass,
                    "selected": bool(row["selected"]),
                    "quality_score": float(row["quality_score"]),
                    "nearest_real_distance": nearest_real,
                    "coverage_before": float(row["coverage_before"]),
                    "coverage_after": float(row["coverage_after"]),
                    "reason": row["selection_rule"],
                }
            )
        stale_rounds = stale_rounds + 1 if round_accepts == 0 else 0
        pd.DataFrame(audit_rows).to_csv(output_dir / "adaptive_round_audit.csv", index=False)
        write_json(
            output_dir / "live_progress.json",
            {
                "status": "adaptive_update",
                "round": round_index,
                "round_candidates": len(tasks),
                "round_accepts": round_accepts,
                "attempts": attempts,
                "accepted": accepted,
                "budgets": budgets,
                "coverage": {name: current_coverage(distances[name], radii[name])[1] for name in MINORITY},
                "stale_rounds": stale_rounds,
                "elapsed_minutes": (time.time() - started) / 60,
            },
        )
        print(
            f"A4 round {round_index}: candidates={len(tasks)} accepted={round_accepts}; "
            f"total={sum(accepted.values())}; coverage="
            + ", ".join(f"{name}:{current_coverage(distances[name], radii[name])[1]:.3f}" for name in MINORITY),
            flush=True,
        )

    del pipeline, vgg, discriminator
    torch.cuda.empty_cache()
    selected_records = []
    for class_name in MINORITY:
        selected_records.extend(active_records[class_name])
    selected_records.sort(key=lambda row: (int(row["label"]), str(row["image_id"])))
    if len(selected_records) != 3752:
        raise RuntimeError(f"Adaptive pool size changed: {len(selected_records)}")
    class_counts = pd.Series([row["label_name"] for row in selected_records]).value_counts().to_dict()
    expected = a3.label_name.value_counts().to_dict()
    if class_counts != expected:
        raise RuntimeError(f"Adaptive class counts differ from A3: {class_counts} vs {expected}")
    for row in selected_records:
        row["dataset_name"] = "aptos2019_a4_adaptive_selected_train_only"
        row["image_path"] = str(row["image_path"])
    write_csv(output_dir / "generated_manifest.csv", selected_records)
    write_json(output_dir / "generated_manifest.json", {"items": selected_records})
    if all_candidate_rows:
        pd.DataFrame(all_candidate_rows).to_csv(output_dir / "all_candidates_manifest.csv", index=False)
    final_coverage = {name: current_coverage(distances[name], radii[name])[1] for name in MINORITY}
    result = {
        "status": "completed",
        "method": "quality-first adaptive local-prototype replacement initialized from frozen A3 pool",
        "test_accessed": False,
        "pool_size": len(selected_records),
        "class_counts": class_counts,
        "initial_local_coverage": initial_coverage,
        "final_local_coverage": final_coverage,
        "coverage_delta": {name: final_coverage[name] - initial_coverage[name] for name in MINORITY},
        "replacement_budgets": budgets,
        "attempts": attempts,
        "accepted_replacements": accepted,
        "candidate_count": len(all_candidate_rows),
        "quality_gate_pass_count": int(sum(bool(row["quality_gate_pass"]) for row in all_candidate_rows)),
        "rounds": round_index,
        "stopped_after_stale_rounds": stale_rounds,
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(completed_path, result)
    write_json(output_dir / "live_progress.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
