from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from training.generate_aptos_sd025_balanced import NEGATIVE_PROMPT, PROMPTS
from utils.fundus import preprocess_fundus_pil
from utils.image_quality import compute_pixel_stats


@dataclass(frozen=True)
class GenerationTask:
    output_index: int
    label: int
    label_name: str
    source_image_id: str
    source_path: str
    output_path: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an APTOS balanced train set with the medical-domain LoRA")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--lora-dir", default="outputs/aptos2019/lora_sd025_experiment/lora_training/final_lora")
    parser.add_argument("--reference-data-root", default="outputs/aptos2019/sd025_balanced_train")
    parser.add_argument("--output-dir", default="outputs/aptos2019/lora_sd025_experiment/balanced_train")
    parser.add_argument("--report", default="APTOS_LoRA_SD025少数类全量补齐报告_20260807.md")
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=0, help="0 means full balancing; positive values create a fixed pilot subset")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_fixed_splits(reference_root: Path, output_dir: Path) -> Dict[str, pd.DataFrame]:
    splits: Dict[str, pd.DataFrame] = {}
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        frame = pd.read_csv(reference_root / "splits" / f"{name}.csv")
        frame.to_csv(output_dir / "splits" / f"{name}.csv", index=False)
        splits[name] = frame
    if len(splits["train"]) != 2563 or len(splits["val"]) != 549 or len(splits["test"]) != 550:
        raise RuntimeError("Unexpected fixed APTOS split sizes")
    return splits


def build_tasks(
    train_df: pd.DataFrame, output_dir: Path, target_count: int, seed: int, max_per_class: int = 0
) -> List[GenerationTask]:
    tasks: List[GenerationTask] = []
    rng = np.random.default_rng(seed)
    for label_name in APTOS_CLASS_ORDER:
        label = APTOS_CLASS_ORDER.index(label_name)
        class_df = train_df[train_df["label_name"] == label_name].reset_index(drop=True)
        need = max(target_count - len(class_df), 0)
        if max_per_class > 0:
            need = min(need, max_per_class)
        if need == 0:
            continue
        sources = class_df.iloc[rng.permutation(len(class_df))].to_dict("records")
        class_dir = output_dir / "generated" / label_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(need):
            source = sources[index % len(sources)]
            tasks.append(
                GenerationTask(
                    output_index=index,
                    label=label,
                    label_name=label_name,
                    source_image_id=str(source["image_id"]),
                    source_path=str(Path(source["image_path"]).resolve()),
                    output_path=str((class_dir / f"{source['image_id']}__lora_sd025__{index:05d}.png").resolve()),
                    seed=seed + label * 100_000 + index,
                )
            )
    return tasks


def verify_task_parity(tasks: Sequence[GenerationTask], reference_root: Path) -> None:
    reference = pd.read_csv(reference_root / "generated_manifest.csv").sort_values(
        ["label", "output_index"]
    ).reset_index(drop=True)
    current = pd.DataFrame([asdict(task) for task in tasks]).sort_values(["label", "output_index"]).reset_index(drop=True)
    keys = ["output_index", "label", "label_name", "source_image_id", "seed"]
    reference = reference.merge(current[["label", "output_index"]], on=["label", "output_index"], how="inner")
    reference = reference.sort_values(["label", "output_index"]).reset_index(drop=True)
    if len(reference) != len(current) or not current[keys].astype(str).equals(reference[keys].astype(str)):
        raise RuntimeError("LoRA generation tasks do not match the fixed ordinary-SD source/seed protocol")


def load_pipeline(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA full generation requires CUDA")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        (ROOT / args.model_dir).resolve(),
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to("cuda")
    pipeline.load_lora_weights((ROOT / args.lora_dir).resolve())
    pipeline.fuse_lora(lora_scale=float(args.lora_scale))
    pipeline.enable_attention_slicing()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def prepared_image(path: str, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = preprocess_fundus_pil(image, crop=True, mask_background=True).image
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def task_record(task: GenerationTask, args: argparse.Namespace) -> Dict[str, object]:
    image = np.asarray(Image.open(task.output_path).convert("RGB"), dtype=np.uint8)
    stats = compute_pixel_stats(image)
    return {
        **asdict(task),
        "image_id": f"{task.source_image_id}__lora_sd025__{task.output_index:05d}",
        "image_path": task.output_path,
        "dataset_name": "aptos2019_lora_sd025_generated_train_only",
        "strength": float(args.strength),
        "lora_scale": float(args.lora_scale),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "prompt": PROMPTS[task.label_name],
        "pixel_std": float(stats["std"]),
        "entropy": float(stats["entropy"]),
        "fundus_area_ratio": float(stats["fundus_area_ratio"]),
        "valid_fundus": bool(stats["is_valid_fundus"]),
        "near_constant": bool(stats["is_near_constant"]),
    }


def generate_missing(tasks: Sequence[GenerationTask], args: argparse.Namespace, output_dir: Path) -> List[Dict[str, object]]:
    missing = [task for task in tasks if not Path(task.output_path).exists()]
    pipeline = load_pipeline(args) if missing else None
    total_batches = (len(missing) + args.batch_size - 1) // args.batch_size
    for batch_number, start in enumerate(range(0, len(missing), args.batch_size), 1):
        batch = missing[start : start + args.batch_size]
        outputs = pipeline(
            prompt=[PROMPTS[task.label_name] for task in batch],
            negative_prompt=[NEGATIVE_PROMPT] * len(batch),
            image=[prepared_image(task.source_path, args.image_size) for task in batch],
            strength=float(args.strength),
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            generator=[torch.Generator(device="cuda").manual_seed(task.seed) for task in batch],
        ).images
        for task, image in zip(batch, outputs):
            image.save(task.output_path)
        completed = sum(Path(task.output_path).exists() for task in tasks)
        write_json(
            output_dir / "live_progress.json",
            {
                "status": "generating",
                "completed": completed,
                "total": len(tasks),
                "batch": batch_number,
                "total_batches": total_batches,
            },
        )
        if batch_number % 25 == 0 or batch_number == total_batches:
            print(f"LoRA generation progress: {completed}/{len(tasks)}", flush=True)
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()
    records = [task_record(task, args) for task in tasks]
    write_csv(output_dir / "generated_manifest.csv", records)
    write_json(output_dir / "generated_manifest.json", {"items": records})
    write_json(output_dir / "live_progress.json", {"status": "completed", "completed": len(tasks), "total": len(tasks)})
    return records


def write_report(
    args: argparse.Namespace,
    output_dir: Path,
    splits: Dict[str, pd.DataFrame],
    records: List[Dict[str, object]],
    elapsed_minutes: float,
) -> None:
    train_counts = splits["train"]["label_name"].value_counts().reindex(APTOS_CLASS_ORDER).astype(int)
    generated_counts = Counter(row["label_name"] for row in records)
    valid_rate = float(np.mean([row["valid_fundus"] for row in records]))
    gray_rate = float(np.mean([row["near_constant"] for row in records]))
    lines = [
        "# APTOS LoRA-SD 0.25 少数类全量补齐报告",
        "",
        "## 1. 本次目标",
        "",
        "使用已在固定 APTOS 真实训练集上完成医学域微调的 SD1.5 U-Net LoRA，在与普通 SD 完全一致的 img2img strength、提示词、源图、随机种子和补齐数量下生成少数类训练图。该输出位于独立目录，不覆盖普通 SD。",
        "",
        "## 2. 公平比较与防泄漏",
        "",
        "- train/val/test 划分直接复制普通 SD 固定切分；仅 train 少数类作为生成源。",
        "- 已逐任务核验 label、source_image_id、output_index 和 seed 与普通 SD 完全一致，唯一生成变量是是否加载医学域 LoRA。",
        "- 验证集和测试集不参与 LoRA 训练、不作为生成源，也不包含合成图。",
        "",
        "## 3. 补齐数量",
        "",
        "| 类别 | 真实训练图 | LoRA 生成图 | 补齐后 |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {int(train_counts[name])} | {int(generated_counts.get(name, 0))} | "
            f"{int(train_counts[name]) + int(generated_counts.get(name, 0))} |"
            for name in APTOS_CLASS_ORDER
        ],
        "",
        "## 4. 生成策略",
        "",
        f"- Base model：本地 Stable Diffusion v1.5；LoRA：`{Path(args.lora_dir)}`；LoRA scale={args.lora_scale}。",
        f"- Img2img strength={args.strength}；guidance scale={args.guidance_scale}；steps={args.num_inference_steps}；分辨率={args.image_size}；batch={args.batch_size}。",
        "- 类别条件提示词与普通 SD 相同；负提示词与普通 SD 相同。",
        f"- 总生成数={len(records)}；有效眼底率={valid_rate:.4f}；灰图/近常量比例={gray_rate:.4f}。",
        f"- 首次生成耗时={elapsed_minutes:.2f} 分钟；GPU={torch.cuda.get_device_name(0)}。",
        "",
        "## 5. 输出与复现",
        "",
        f"- 生成图：`{(output_dir / 'generated').relative_to(ROOT)}`",
        f"- 清单：`{(output_dir / 'generated_manifest.csv').relative_to(ROOT)}`",
        f"- 固定切分副本：`{(output_dir / 'splits').relative_to(ROOT)}`",
        "",
        "```bash",
        "bash scripts/run_aptos_lora_generation.sh",
        "```",
        "",
        "进度：`bash scripts/watch_aptos_lora_pipeline.sh`。",
        "",
        "## 6. 限制与下一步",
        "",
        "本报告只证明全量生成完成及基础文件有效性。是否解决普通 SD 的医学域差距，必须由同协议 FID/KID/IPR 与下游随机初始化 ConvNeXt 共同判断；在两方面均超过普通 SD 前，不覆盖普通 SD 图片或报告。",
        "",
        f"- Python {platform.python_version()}；PyTorch {torch.__version__}；CUDA {torch.version.cuda}。",
    ]
    (ROOT / args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = (ROOT / args.output_dir).resolve()
    freeze_markers = [output_dir / "FROZEN_BASELINE.json", output_dir.parent / "FROZEN_BASELINE.json"]
    freeze_marker = next((path for path in freeze_markers if path.exists()), None)
    if freeze_marker is not None:
        frozen = json.loads(freeze_marker.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Refusing to modify frozen baseline {frozen.get('baseline_version', 'unknown')}: {output_dir}. "
            "Use a new --output-dir for reproduction or request explicit baseline unfreezing."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_root = (ROOT / args.reference_data_root).resolve()
    lora_file = (ROOT / args.lora_dir / "pytorch_lora_weights.safetensors").resolve()
    if not lora_file.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_file}")
    config = vars(args).copy()
    config.update(
        {
            "model_dir_resolved": str((ROOT / args.model_dir).resolve()),
            "lora_dir_resolved": str((ROOT / args.lora_dir).resolve()),
            "reference_data_root_resolved": str(reference_root),
            "protocol": "exact ordinary-SD split/source/seed parity; LoRA is the only generation variable",
        }
    )
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            existing_core = {key: value for key, value in existing.items() if key not in {"max_per_class", "report"}}
            config_core = {key: value for key, value in config.items() if key not in {"max_per_class", "report"}}
            pilot_upgrade = (
                int(existing.get("max_per_class", 0)) > 0
                and int(config.get("max_per_class", 0)) == 0
                and existing_core == config_core
            )
            if not pilot_upgrade:
                raise RuntimeError("Existing LoRA generation directory has a different generation-critical configuration")
    write_json(config_path, config)
    splits = load_fixed_splits(reference_root, output_dir)
    target_count = int(splits["train"]["label_name"].value_counts().max())
    tasks = build_tasks(splits["train"], output_dir, target_count, args.seed, args.max_per_class)
    verify_task_parity(tasks, reference_root)
    write_json(output_dir / "generation_tasks.json", {"items": [asdict(task) for task in tasks]})
    print(f"Planned LoRA synthetic images: {len(tasks)}; exact ordinary-SD task parity verified", flush=True)
    records = generate_missing(tasks, args, output_dir)
    elapsed_minutes = (time.time() - started) / 60.0
    write_report(args, output_dir, splits, records, elapsed_minutes)
    print(f"LoRA generation completed: {len(records)} images", flush=True)
    print(f"Report: {(ROOT / args.report).resolve()}", flush=True)


if __name__ == "__main__":
    main()
