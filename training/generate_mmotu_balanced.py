from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER, MMOTU_CLASS_SLUGS
from training.train_mmotu_lora import PROMPTS
from utils.ultrasound import preprocess_ultrasound_mask, preprocess_ultrasound_pil


NEGATIVE_PROMPT = (
    "color photograph, illustration, cartoon, painting, text-only image, empty frame, duplicated anatomy, "
    "severe geometric distortion, checkerboard artifact, low contrast blank image"
)
PRIOR_SOURCE_DESCRIPTION = "train source GT binary mask"


@dataclass(frozen=True)
class GenerationTask:
    output_index: int
    label: int
    label_name: str
    label_slug: str
    source_image_id: str
    source_path: str
    prior_path: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed MMOTU 100% balanced synthetic pool")
    parser.add_argument("--method", choices=["a0", "a1", "a3"], required=True)
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--shared-tasks", default="outputs/mmotu2d/shared/generation_tasks.json")
    parser.add_argument(
        "--task-file", default="",
        help="Optional explicit train-only generation task JSON for a frozen extension pool; default keeps the 100% shared-task protocol.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--anchor-strength", type=float, default=0.0)
    parser.add_argument("--gate-power", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
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


def build_tasks(frame: pd.DataFrame, seed: int) -> List[GenerationTask]:
    rng = np.random.default_rng(seed)
    target = int(frame.label.value_counts().max())
    tasks: List[GenerationTask] = []
    for label, (class_name, slug) in enumerate(zip(MMOTU_CLASS_ORDER, MMOTU_CLASS_SLUGS)):
        class_frame = frame[frame.label == label].reset_index(drop=True)
        need = target - len(class_frame)
        if need <= 0:
            continue
        sources = class_frame.iloc[rng.permutation(len(class_frame))].to_dict("records")
        for index in range(need):
            source = sources[index % len(sources)]
            tasks.append(
                GenerationTask(
                    output_index=index,
                    label=label,
                    label_name=class_name,
                    label_slug=slug,
                    source_image_id=str(source["image_id"]),
                    source_path=str(Path(source["image_path"]).resolve()),
                    prior_path=str(Path(source["mask_path"]).resolve()),
                    seed=seed + label * 100_000 + index,
                )
            )
    return tasks


def load_or_create_shared_tasks(frame: pd.DataFrame, path: Path, seed: int) -> List[GenerationTask]:
    expected = build_tasks(frame, seed)
    expected_payload = [asdict(task) for task in expected]
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))["items"]
        if existing != expected_payload:
            raise RuntimeError("Existing MMOTU shared generation tasks differ from the fixed train protocol")
    else:
        write_json(path, {"items": expected_payload, "test_accessed": False, "synthetic_budget": len(expected)})
    return expected


def prepared_image(path: str, size: int) -> Image.Image:
    with Image.open(path) as handle:
        return preprocess_ultrasound_pil(handle.convert("RGB"), size)


def prior_tensor(path: str, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as handle:
        prior = preprocess_ultrasound_mask(handle, size)
    return TF.to_tensor(prior).clamp(0.0, 1.0).to(device)


def pixel_stats(path: str) -> Dict[str, object]:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    gray = array.mean(axis=2).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probability = hist[hist > 0] / hist.sum()
    entropy = float(-(probability * np.log2(probability)).sum())
    return {
        "pixel_mean": float(array.mean()),
        "pixel_std": float(array.std()),
        "entropy": entropy,
        "near_constant": bool(array.std() < 5.0 or entropy < 1.0),
        "grayscale_channel_difference": float(np.abs(array.astype(np.int16) - gray[..., None]).mean()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU generation requires CUDA")
    if args.method == "a0" and args.anchor_strength != 0:
        raise ValueError("A0 must not use structural anchoring")
    if args.method in {"a1", "a3"} and args.anchor_strength <= 0:
        raise ValueError("A1/A3 generation must use a positive structural anchor")
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    if completed_path.is_file() and (output_dir / "generated_manifest.csv").is_file():
        print(f"[MMOTU {args.method}] generation already completed: {output_dir}", flush=True)
        return

    frame = pd.read_csv(ROOT / args.train_split)
    if args.task_file:
        payload = json.loads((ROOT / args.task_file).resolve().read_text(encoding="utf-8"))
        tasks = [GenerationTask(**item) for item in payload["items"]]
        train_ids = set(frame.image_id.astype(str))
        for task in tasks:
            if task.source_image_id not in train_ids:
                raise RuntimeError(f"Explicit task source is not in frozen train split: {task.source_image_id}")
            if task.label_name != MMOTU_CLASS_ORDER[task.label]:
                raise RuntimeError(f"Explicit task class mismatch: {task}")
            if Path(task.source_path).resolve() != Path(
                frame.loc[frame.image_id.astype(str).eq(task.source_image_id), "image_path"].iloc[0]
            ).resolve():
                raise RuntimeError(f"Explicit task source path mismatch: {task.source_image_id}")
        if len({(task.label, task.output_index) for task in tasks}) != len(tasks):
            raise RuntimeError("Explicit generation tasks contain duplicate class/output_index pairs")
    else:
        tasks = load_or_create_shared_tasks(frame, (ROOT / args.shared_tasks).resolve(), args.seed)
    if args.max_per_class > 0:
        retained = []
        counts: Dict[int, int] = {}
        for task in tasks:
            if counts.get(task.label, 0) < args.max_per_class:
                retained.append(task)
                counts[task.label] = counts.get(task.label, 0) + 1
        tasks = retained
    config = vars(args).copy()
    config.update(
        {
            "class_names": MMOTU_CLASS_ORDER,
            "task_count": len(tasks),
            "explicit_task_file": str((ROOT / args.task_file).resolve()) if args.task_file else "",
            "test_accessed": False,
            "filtering": "none; all generated images retained",
            "prior_source": PRIOR_SOURCE_DESCRIPTION if args.method != "a0" else "none",
        }
    )
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "generation_tasks.json", {"items": [asdict(task) for task in tasks]})

    def output_path(task: GenerationTask) -> Path:
        path = output_dir / "generated" / task.label_slug / f"{task.source_image_id}__{args.method}__{task.output_index:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    missing = [task for task in tasks if not output_path(task).is_file()]
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
        pipeline.fuse_lora(lora_scale=args.lora_scale)
        pipeline.enable_attention_slicing()
        pipeline.set_progress_bar_config(disable=True)
        state: Dict[str, torch.Tensor | None] = {"source_latents": None, "gate": None}

        def anchor_callback(_pipe, _step_index, _timestep, callback_kwargs):
            source_latents = state["source_latents"]
            gate = state["gate"]
            if source_latents is None or gate is None:
                raise RuntimeError("Structural callback missing source state")
            latents = callback_kwargs["latents"]
            callback_kwargs["latents"] = latents + args.anchor_strength * gate * (source_latents - latents)
            return callback_kwargs

        total_batches = math.ceil(len(missing) / args.batch_size)
        try:
            for batch_number, start in enumerate(range(0, len(missing), args.batch_size), 1):
                batch = missing[start : start + args.batch_size]
                source_images = [prepared_image(task.source_path, args.image_size) for task in batch]
                callback = None
                callback_inputs = None
                if args.method != "a0":
                    priors = torch.stack([prior_tensor(task.prior_path, args.image_size, device) for task in batch]).half()
                    with torch.no_grad():
                        source_tensor = pipeline.image_processor.preprocess(source_images).to(device=device, dtype=torch.float16)
                        source_latents = pipeline.vae.encode(source_tensor).latent_dist.mode() * pipeline.vae.config.scaling_factor
                        gate = F.interpolate(priors, size=source_latents.shape[-2:], mode="nearest").clamp(0, 1)
                        state["source_latents"] = source_latents
                        state["gate"] = gate.pow(args.gate_power)
                    callback = anchor_callback
                    callback_inputs = ["latents"]
                call = {
                    "prompt": [PROMPTS[task.label_name] for task in batch],
                    "negative_prompt": [NEGATIVE_PROMPT] * len(batch),
                    "image": source_images,
                    "strength": args.strength,
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "generator": [torch.Generator(device="cuda").manual_seed(task.seed) for task in batch],
                }
                if callback is not None:
                    call["callback_on_step_end"] = callback
                    call["callback_on_step_end_tensor_inputs"] = callback_inputs
                with torch.inference_mode():
                    images = pipeline(**call).images
                for task, image in zip(batch, images):
                    image.save(output_path(task))
                completed = sum(output_path(task).is_file() for task in tasks)
                progress = {
                    "status": "generating",
                    "method": args.method,
                    "completed": completed,
                    "total": len(tasks),
                    "batch": batch_number,
                    "total_batches": total_batches,
                    "elapsed_minutes": (time.time() - started) / 60,
                }
                write_json(output_dir / "live_progress.json", progress)
                if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
                    print(f"[MMOTU {args.method}] generation {completed}/{len(tasks)}", flush=True)
        finally:
            state["source_latents"] = None
            state["gate"] = None
            del pipeline
            torch.cuda.empty_cache()

    records = []
    for task in tasks:
        path = output_path(task)
        records.append(
            {
                "image_id": f"{task.source_image_id}__{args.method}__{task.output_index:05d}",
                "image_path": str(path.resolve()),
                "output_index": task.output_index,
                "label": task.label,
                "label_name": task.label_name,
                "label_slug": task.label_slug,
                "source_image_id": task.source_image_id,
                "source_path": task.source_path,
                "prior_path": task.prior_path if args.method != "a0" else "",
                "seed": task.seed,
                "method": args.method,
                "strength": args.strength,
                "lora_scale": args.lora_scale,
                "anchor_strength": args.anchor_strength,
                "gate_power": args.gate_power if args.method != "a0" else 0.0,
                "prompt": PROMPTS[task.label_name],
                "selected": True,
                "selection_rule": "none; full generated pool",
                **pixel_stats(str(path)),
            }
        )
    write_csv(output_dir / "generated_manifest.csv", records)
    write_json(output_dir / "generated_manifest.json", {"items": records})
    completed = {
        "status": "completed",
        "method": args.method,
        "count": len(records),
        "near_constant": int(sum(bool(row["near_constant"]) for row in records)),
        "elapsed_minutes": (time.time() - started) / 60,
        "test_accessed": False,
        "filtering": "none",
    }
    write_json(completed_path, completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
