from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from training.run_aptos_convnext_baselines import (
    Experiment,
    make_summary_plots,
    seed_everything,
    train_experiment,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scratch ConvNeXt evaluation for the no-filter structure-guidance ablation")
    parser.add_argument("--data-root", default="outputs/aptos2019/structure_guidance_v3/balanced_train")
    parser.add_argument(
        "--baseline-root",
        default="outputs/aptos2019/lora_sd025_experiment/scale_sweep/scale_0p8/balanced_train",
    )
    parser.add_argument(
        "--baseline-metrics",
        default="outputs/aptos2019/lora_sd025_experiment/scale_sweep/scale_0p8/convnext_tiny_scratch/lora_sd025_augmented_ce/test_metrics.json",
    )
    parser.add_argument("--output-dir", default="outputs/aptos2019/structure_guidance_v3/convnext_tiny_scratch")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiment-key", default="structure_guided_augmented_ce")
    parser.add_argument("--display-name", default="LoRA 0.8 + Structure Guidance")
    parser.add_argument(
        "--guidance-description",
        default="Grad-CAM门控的逐步源latent锚定",
    )
    return parser.parse_args()


def assert_split_parity(current_root: Path, baseline_root: Path) -> Dict[str, pd.DataFrame]:
    splits = {}
    for split in ["train", "val", "test"]:
        current = pd.read_csv(current_root / "splits" / f"{split}.csv")
        baseline = pd.read_csv(baseline_root / "splits" / f"{split}.csv")
        keys = ["image_id", "label", "label_name"]
        if not current[keys].astype(str).equals(baseline[keys].astype(str)):
            raise RuntimeError(f"Structure-guidance and frozen baseline {split} splits differ")
        splits[split] = current
    return splits


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Structure-guidance ConvNeXt evaluation requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    data_root = (ROOT / args.data_root).resolve()
    baseline_root = (ROOT / args.baseline_root).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = assert_split_parity(data_root, baseline_root)
    generated = pd.read_csv(data_root / "generated_manifest.csv")
    if len(generated) != 3752:
        raise RuntimeError(f"Expected all 3752 generated images, got {len(generated)}")
    if "selected" in generated and not generated["selected"].astype(bool).all():
        raise RuntimeError("This ablation forbids post-generation filtering")
    if not generated.image_path.map(lambda value: Path(str(value)).is_file()).all():
        raise RuntimeError("A full unfiltered manifest contains a missing generated image")
    if set(generated["source_image_id"].astype(str)) - set(splits["train"]["image_id"].astype(str)):
        raise RuntimeError("Found a structure-guided source outside the fixed train split")
    generated_train = generated[["image_id", "image_path", "label", "label_name", "dataset_name"]]
    augmented = pd.concat([splits["train"], generated_train], ignore_index=True)
    counts = splits["train"]["label_name"].value_counts().reindex(APTOS_CLASS_ORDER).astype(int).tolist()
    experiment = Experiment(
        args.experiment_key,
        args.display_name,
        "real+structure_guided",
        "traditional",
        "cross_entropy",
    )
    result = train_experiment(
        experiment,
        augmented,
        splits["val"],
        splits["test"],
        counts,
        args,
        output_dir,
        device,
    )
    baseline = json.loads((ROOT / args.baseline_metrics).read_text(encoding="utf-8"))
    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auprc",
        "minority_avg_recall",
        "quadratic_weighted_kappa",
    ]
    comparison = {
        "protocol": "same real split, same 3752 source/label/seed tasks, no filtering, weights=None ConvNeXt",
        "baseline": {key: baseline[key] for key in keys},
        "structure_guidance": {key: result[key] for key in keys},
        "delta": {key: result[key] - baseline[key] for key in keys},
        "all_core_metrics_above_baseline": all(
            result[key] > baseline[key]
            for key in [
                "balanced_accuracy",
                "macro_f1",
                "macro_auprc",
                "minority_avg_recall",
                "quadratic_weighted_kappa",
            ]
        ),
    }
    write_json(output_dir / "baseline_comparison.json", comparison)
    pd.DataFrame(
        [
            {"method": "LoRA 0.8 Baseline", **comparison["baseline"]},
            {"method": args.display_name, **comparison["structure_guidance"]},
        ]
    ).to_csv(output_dir / "baseline_comparison.csv", index=False)
    make_summary_plots([baseline, result], output_dir)
    lines = [
        "# APTOS结构引导随机初始化ConvNeXt分类报告",
        "",
        "## 实验边界",
        "",
        f"本实验只在冻结LoRA 0.8生成器上加入{args.guidance_description}。没有生成后筛选、对抗损失、动态损失、采样重加权或预训练下游分类器。全部3752张结构引导图均进入训练。",
        "",
        "## 公平协议",
        "",
        "- 固定train/val/test与Baseline逐样本相同；val/test只含真实图。",
        "- 生成source、类别、数量、output index和seed与Baseline逐项相同。",
        "- ConvNeXt-Tiny使用weights=None；数据增强、AdamW、warm-up、cosine、早停和seed均与Baseline相同。",
        "",
        "## 测试结果",
        "",
        "| 方法 | Accuracy | Balanced Acc. | Macro-F1 | Macro-AUPRC | Minority Recall | QWK |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| LoRA 0.8 Baseline | {baseline['accuracy']:.5f} | {baseline['balanced_accuracy']:.5f} | {baseline['macro_f1']:.5f} | {baseline['macro_auprc']:.5f} | {baseline['minority_avg_recall']:.5f} | {baseline['quadratic_weighted_kappa']:.5f} |",
        f"| {args.display_name} | {result['accuracy']:.5f} | {result['balanced_accuracy']:.5f} | {result['macro_f1']:.5f} | {result['macro_auprc']:.5f} | {result['minority_avg_recall']:.5f} | {result['quadratic_weighted_kappa']:.5f} |",
        "",
        "## 结果边界",
        "",
        f"五项核心分类指标是否全部超过Baseline：**{'是' if comparison['all_core_metrics_above_baseline'] else '否'}**。本轮是单seed消融，只报告观察值，不宣称统计显著。",
        "",
        f"详细指标：`{output_dir / experiment.key / 'test_metrics.json'}`",
    ]
    (output_dir / "结构引导分类报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Structure-guidance classifier completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
