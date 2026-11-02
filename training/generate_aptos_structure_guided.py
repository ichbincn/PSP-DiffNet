from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.generate_aptos_sd025_balanced import NEGATIVE_PROMPT, PROMPTS
from utils.fundus import preprocess_fundus_pil
from utils.image_quality import compute_pixel_stats

ANCHOR_STRENGTH = 0.08
GATE_POWER = 0.5
VERSION_TAG = "structure_v3"


@dataclass(frozen=True)
class Task:
    output_index: int
    label: int
    label_name: str
    source_image_id: str
    source_path: str
    output_path: str
    prior_path: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the frozen APTOS A1 weak-prior augmentation without filtering")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument(
        "--lora-dir", default="outputs/aptos2019/lora_sd025_experiment/lora_training/final_lora"
    )
    parser.add_argument(
        "--prior-index",
        default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv",
    )
    parser.add_argument(
        "--baseline-data-root",
        default="outputs/aptos2019/lora_sd025_experiment/scale_sweep/scale_0p8/balanced_train",
    )
    parser.add_argument("--output-dir", default="outputs/aptos2019/structure_guidance_v3/balanced_train")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="Development smoke only; zero means all 3752")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Development mechanism check only; zero disables balanced per-class truncation",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--version-tag", default=VERSION_TAG)
    parser.add_argument("--dataset-name", default="aptos2019_a1_weak_prior_generated_train_only")
    parser.add_argument(
        "--method-name",
        default="A1 Grad-CAM sqrt(P)-gated source-latent anchoring at every denoising step",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_tasks(
    baseline_root: Path,
    prior_index_path: Path,
    output_dir: Path,
    max_images: int,
    max_images_per_class: int,
    version_tag: str,
) -> List[Task]:
    reference = json.loads((baseline_root / "generation_tasks.json").read_text(encoding="utf-8"))["items"]
    priors = pd.read_csv(prior_index_path)
    prior_paths = dict(zip(priors["image_id"].astype(str), priors["prior_path"].astype(str)))
    tasks = []
    for item in reference:
        source_id = str(item["source_image_id"])
        if source_id not in prior_paths or not Path(prior_paths[source_id]).is_file():
            raise RuntimeError(f"Missing structural prior for generation source {source_id}")
        class_dir = output_dir / "generated" / str(item["label_name"])
        class_dir.mkdir(parents=True, exist_ok=True)
        tasks.append(
            Task(
                output_index=int(item["output_index"]),
                label=int(item["label"]),
                label_name=str(item["label_name"]),
                source_image_id=source_id,
                source_path=str(Path(item["source_path"]).resolve()),
                output_path=str(
                    (class_dir / f"{source_id}__{version_tag}__{int(item['output_index']):05d}.png").resolve()
                ),
                prior_path=str(Path(prior_paths[source_id]).resolve()),
                seed=int(item["seed"]),
            )
        )
    if len(tasks) != 3752:
        raise RuntimeError(f"Expected 3752 frozen baseline tasks, got {len(tasks)}")
    if max_images_per_class > 0:
        retained = []
        per_class: Dict[str, int] = {}
        for task in tasks:
            count = per_class.get(task.label_name, 0)
            if count < max_images_per_class:
                retained.append(task)
                per_class[task.label_name] = count + 1
        tasks = retained
    elif max_images > 0:
        tasks = tasks[:max_images]
    return tasks


def prepared_image(path: str, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = preprocess_fundus_pil(image, crop=True, mask_background=True).image
    return image.resize((size, size), Image.Resampling.LANCZOS)


def prior_tensor(path: str, size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return TF.to_tensor(image).clamp(0.0, 1.0).to(device)


def record(
    task: Task,
    args: argparse.Namespace,
) -> Dict[str, object]:
    image = np.asarray(Image.open(task.output_path).convert("RGB"), dtype=np.uint8)
    stats = compute_pixel_stats(image)
    image_id = f"{task.source_image_id}__{args.version_tag}__{task.output_index:05d}"
    return {
        "output_index": task.output_index,
        "label": task.label,
        "label_name": task.label_name,
        "source_image_id": task.source_image_id,
        "source_path": task.source_path,
        "prior_path": task.prior_path,
        "output_path": task.output_path,
        "seed": task.seed,
        "image_id": image_id,
        "image_path": task.output_path,
        "dataset_name": args.dataset_name,
        "strength": float(args.strength),
        "lora_scale": float(args.lora_scale),
        "guidance_mode": "gradcam_gated_source_latent_anchor",
        "anchor_policy": "uniform",
        "anchor_strength_per_step": ANCHOR_STRENGTH,
        "gate_power": GATE_POWER,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "prompt": PROMPTS[task.label_name],
        "pixel_std": float(stats["std"]),
        "entropy": float(stats["entropy"]),
        "fundus_area_ratio": float(stats["fundus_area_ratio"]),
        "valid_fundus": bool(stats["is_valid_fundus"]),
        "near_constant": bool(stats["is_near_constant"]),
        "selected": True,
        "selection_rule": "none; every generated image enters downstream training",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Structure-guided generation requires CUDA")
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    if (output_dir.parent / "FROZEN_A1.json").is_file():
        raise RuntimeError(
            "The official A1 directory is frozen; reproduce into a new directory with "
            "scripts/reproduce_aptos_structure_a1.sh"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = (ROOT / args.baseline_data_root).resolve()
    prior_index_path = (ROOT / args.prior_index).resolve()
    train_frame = pd.read_csv(baseline_root / "splits" / "train.csv")
    train_class_counts = {
        str(name): int(count) for name, count in train_frame["label_name"].value_counts().items()
    }
    config = vars(args).copy()
    config.update(
        {
            "baseline_data_root_resolved": str(baseline_root),
            "prior_index_resolved": str(prior_index_path),
            "train_class_counts": train_class_counts,
            "anchor_strength_by_class": {
                name: ANCHOR_STRENGTH for name in train_class_counts if name != "No_DR"
            },
            "gate_power": GATE_POWER,
            "version_tag": args.version_tag,
            "method": args.method_name,
            "filtering": "none",
        }
    )
    config_path = output_dir / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing structure-guided generation directory has a different configuration")
    write_json(config_path, config)
    for split in ["train", "val", "test"]:
        frame = pd.read_csv(baseline_root / "splits" / f"{split}.csv")
        (output_dir / "splits").mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "splits" / f"{split}.csv", index=False)
    tasks = build_tasks(
        baseline_root,
        prior_index_path,
        output_dir,
        args.max_images,
        args.max_images_per_class,
        args.version_tag,
    )
    write_json(output_dir / "generation_tasks.json", {"items": [task.__dict__ for task in tasks]})
    missing = [task for task in tasks if not Path(task.output_path).is_file()]
    print(f"Structure-guided tasks: {len(tasks)}; missing: {len(missing)}; filtering: none", flush=True)
    started = time.time()
    if missing:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            (ROOT / args.model_dir).resolve(),
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=True,
        ).to(device)
        pipeline.load_lora_weights((ROOT / args.lora_dir).resolve())
        pipeline.fuse_lora(lora_scale=float(args.lora_scale))
        pipeline.set_progress_bar_config(disable=True)
        state: Dict[str, torch.Tensor | None] = {
            "priors": None,
            "source_latents": None,
            "gate": None,
        }

        def anchor_callback(pipe, step_index, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            source_latents = state["source_latents"]
            gate = state["gate"]
            if source_latents is None or gate is None:
                raise RuntimeError("A1 latent anchor callback called without source latents or gate")
            callback_kwargs["latents"] = latents + ANCHOR_STRENGTH * gate * (source_latents - latents)
            return callback_kwargs

        total_batches = (len(missing) + args.batch_size - 1) // args.batch_size
        try:
            for batch_number, start in enumerate(range(0, len(missing), args.batch_size), 1):
                batch = missing[start : start + args.batch_size]
                state["priors"] = torch.stack(
                    [prior_tensor(task.prior_path, args.image_size, device) for task in batch]
                ).to(dtype=torch.float16)
                source_images = [prepared_image(task.source_path, args.image_size) for task in batch]
                with torch.no_grad():
                    source_tensor = pipeline.image_processor.preprocess(source_images).to(
                        device=device, dtype=torch.float16
                    )
                    state["source_latents"] = (
                        pipeline.vae.encode(source_tensor).latent_dist.mode()
                        * pipeline.vae.config.scaling_factor
                    )
                    state["gate"] = torch.nn.functional.interpolate(
                        state["priors"],
                        size=state["source_latents"].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0).sqrt()
                with torch.no_grad():
                    images = pipeline(
                        prompt=[PROMPTS[task.label_name] for task in batch],
                        negative_prompt=[NEGATIVE_PROMPT] * len(batch),
                        image=source_images,
                        strength=float(args.strength),
                        guidance_scale=float(args.guidance_scale),
                        num_inference_steps=int(args.num_inference_steps),
                        generator=[torch.Generator(device="cuda").manual_seed(task.seed) for task in batch],
                        callback_on_step_end=anchor_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).images
                for task, image in zip(batch, images):
                    image.save(task.output_path)
                completed = sum(Path(task.output_path).is_file() for task in tasks)
                progress = {
                    "stage": "generation",
                    "status": "running",
                    "completed": completed,
                    "total": len(tasks),
                    "batch": batch_number,
                    "total_batches": total_batches,
                    "elapsed_minutes": (time.time() - started) / 60.0,
                    "filtering": "none",
                }
                write_json(output_dir / "live_progress.json", progress)
                if batch_number % 25 == 0 or batch_number == total_batches:
                    print(f"Structure generation: {completed}/{len(tasks)}", flush=True)
        finally:
            state["priors"] = None
            state["source_latents"] = None
            state["gate"] = None
            del pipeline
            torch.cuda.empty_cache()
    records = [record(task, args) for task in tasks]
    write_csv(output_dir / "generated_manifest.csv", records)
    write_json(output_dir / "generated_manifest.json", {"items": records})
    completed = {
        "stage": "generation",
        "status": "completed",
        "completed": len(records),
        "total": len(tasks),
        "valid_fundus": sum(bool(item["valid_fundus"]) for item in records),
        "near_constant": sum(bool(item["near_constant"]) for item in records),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "filtering": "none; all generated images are retained and used",
    }
    write_json(output_dir / "completed.json", completed)
    write_json(output_dir / "live_progress.json", completed)
    print(f"Structure-guided generation completed: {len(records)}", flush=True)


if __name__ == "__main__":
    main()
