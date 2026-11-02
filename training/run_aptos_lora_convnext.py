from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List

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


CORE_METRICS = [
    ("balanced_accuracy", "Balanced Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("macro_auprc", "Macro-AUPRC"),
    ("minority_avg_recall", "Minority Recall"),
    ("quadratic_weighted_kappa", "QWK"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only the LoRA-SD augmented scratch ConvNeXt group")
    parser.add_argument("--data-root", default="outputs/aptos2019/lora_sd025_experiment/balanced_train")
    parser.add_argument("--fixed-results", default="outputs/aptos2019/convnext_tiny_scratch_preliminary")
    parser.add_argument("--split-reference-root", default="", help="Optional fixed ordinary-SD protocol root used only to verify split parity")
    parser.add_argument("--output-dir", default="outputs/aptos2019/lora_sd025_experiment/convnext_tiny_scratch")
    parser.add_argument("--report", default="APTOS_LoRA_SD025分类验证报告_20260807.md")
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
    return parser.parse_args()


def assert_split_parity(data_root: Path, ordinary_root: Path) -> None:
    for split in ["train", "val", "test"]:
        current = pd.read_csv(data_root / "splits" / f"{split}.csv")
        ordinary = pd.read_csv(ordinary_root / "splits" / f"{split}.csv")
        keys = ["image_id", "label", "label_name"]
        if not current[keys].astype(str).equals(ordinary[keys].astype(str)):
            raise RuntimeError(f"LoRA and ordinary-SD {split} splits differ")


def metric_table(rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| 方法 | Balanced Acc. | Macro-F1 | Macro-AUPRC | Minority Recall | QWK | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['macro_auprc']:.4f} | {row['minority_avg_recall']:.4f} | "
            f"{row['quadratic_weighted_kappa']:.4f} | {row['accuracy']:.4f} |"
        )
    return lines


def class_metric(result: Dict[str, object], class_name: str, key: str) -> float:
    return float(next(item[key] for item in result["per_class"] if item["class_name"] == class_name))


def write_report(
    args: argparse.Namespace,
    output_dir: Path,
    all_results: List[Dict[str, object]],
    ordinary_sd: Dict[str, object],
    lora: Dict[str, object],
) -> None:
    gate_items = [(key, label, float(lora[key]) > float(ordinary_sd[key])) for key, label in CORE_METRICS]
    classification_gate = all(item[2] for item in gate_items)
    lines = [
        "# APTOS LoRA-SD 0.25 随机初始化 ConvNeXt 分类验证报告",
        "",
        "## 1. 本轮任务",
        "",
        "本轮只训练 LoRA-SD 0.25 增强组。Original、Traditional Augmentation、Balanced Softmax、Focal Loss 和普通 SD 0.25 均直接读取已经固化的随机初始化结果，没有重新训练。",
        "",
        "## 2. 公平比较协议",
        "",
        "- 下游分类器：ConvNeXt-Tiny，`weights=None`，2782万参数全部从随机初始化训练；禁止载入分类预训练权重。",
        "- train/val/test 与普通 SD 实验逐样本一致；LoRA 生成源图、类别数量和 seed 也与普通 SD 一致。",
        "- LoRA 组训练集：2563 张真实图 + 3752 张 LoRA 合成图，共 6315 张；验证集549张、测试集550张，均为真实图。",
        "- 唯一比较变量：训练集合成图来自普通 SD 还是医学域 LoRA-SD。",
        "",
        "## 3. 分类训练策略",
        "",
        f"- AdamW，lr={args.learning_rate}，weight decay={args.weight_decay}；5轮线性 warm-up 后 cosine decay 至 {args.min_learning_rate}。",
        f"- 最大 {args.epochs} epoch，batch={args.batch_size}，FP16，梯度裁剪1.0；验证 Macro-F1 早停 patience={args.early_stopping_patience}。",
        "- 传统增强与普通 SD 组完全相同：眼底ROI预处理、RandomResizedCrop 224、水平翻转、±15°旋转和轻量颜色扰动。",
        f"- seed={args.seed}，固定确定性设置；最佳验证 Macro-F1 checkpoint 只在固定测试集评价一次。",
        "",
        "## 4. 固化对比结果与 LoRA 结果",
        "",
        *metric_table(all_results),
        "",
        "## 5. LoRA 相对普通 SD 的变化",
        "",
        "| 核心指标 | 普通 SD | LoRA-SD | 绝对变化 | 是否超过 |",
        "|---|---:|---:|---:|---|",
        *[
            f"| {label} | {ordinary_sd[key]:.4f} | {lora[key]:.4f} | "
            f"{lora[key]-ordinary_sd[key]:+.4f} | {'是' if passed else '否'} |"
            for key, label, passed in gate_items
        ],
        f"| Accuracy（补充） | {ordinary_sd['accuracy']:.4f} | {lora['accuracy']:.4f} | "
        f"{lora['accuracy']-ordinary_sd['accuracy']:+.4f} | {'是' if lora['accuracy'] > ordinary_sd['accuracy'] else '否'} |",
        "",
        f"分类严格门槛（五项核心指标全部高于普通 SD）：**{'通过' if classification_gate else '未通过'}**。",
        "",
        "## 6. 逐类别 Recall",
        "",
        "| 类别 | 普通 SD | LoRA-SD | 变化 |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {class_metric(ordinary_sd, name, 'recall'):.4f} | {class_metric(lora, name, 'recall'):.4f} | "
            f"{class_metric(lora, name, 'recall')-class_metric(ordinary_sd, name, 'recall'):+.4f} |"
            for name in APTOS_CLASS_ORDER
        ],
        "",
        "## 7. 输出与复现",
        "",
        f"- LoRA 分类输出：`{output_dir.relative_to(ROOT)}`",
        f"- 测试指标：`{(output_dir / 'lora_sd025_augmented_ce' / 'test_metrics.json').relative_to(ROOT)}`",
        f"- 混淆矩阵：`{(output_dir / 'lora_sd025_augmented_ce' / 'test_confusion_matrix.png').relative_to(ROOT)}`",
        f"- 逐样本预测：`{(output_dir / 'lora_sd025_augmented_ce' / 'test_predictions.csv').relative_to(ROOT)}`",
        "",
        "```bash",
        "bash scripts/run_aptos_lora_classifier.sh",
        "```",
        "",
        "进度：`bash scripts/watch_aptos_lora_pipeline.sh`。",
        "",
        "## 8. 解释限制",
        "",
        "- 本轮为单 seed 初步验证，严格门槛只用于决定是否替换当前普通 SD 工作版本，不等同于统计显著性。",
        "- 图像质量门槛还需结合 FID/KID/IPR 独立判断；分类门槛通过也不能单独触发覆盖。",
        f"- Python {platform.python_version()}；PyTorch {torch.__version__}；GPU {torch.cuda.get_device_name(0)}。",
    ]
    (ROOT / args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "classification_gate.json",
        {
            "rule": "Balanced Accuracy, Macro-F1, Macro-AUPRC, Minority Recall and QWK must all exceed ordinary SD",
            "passed": classification_gate,
            "items": {label: passed for _, label, passed in gate_items},
        },
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA ConvNeXt validation requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    data_root = (ROOT / args.data_root).resolve()
    fixed_results_dir = (ROOT / args.fixed_results).resolve()
    if args.split_reference_root:
        ordinary_data_root = (ROOT / args.split_reference_root).resolve()
    else:
        candidates = [
            ROOT / "outputs/aptos2019/sd025_balanced_train",
            ROOT / "outputs/aptos2019/frozen_experiment_records_20260809/numeric_snapshot/sd025_balanced_train",
        ]
        ordinary_data_root = next((path for path in candidates if (path / "splits/train.csv").is_file()), None)
        if ordinary_data_root is None:
            raise FileNotFoundError("No fixed split reference was found; pass --split-reference-root")
    assert_split_parity(data_root, ordinary_data_root)
    train_df = pd.read_csv(data_root / "splits/train.csv")
    val_df = pd.read_csv(data_root / "splits/val.csv")
    test_df = pd.read_csv(data_root / "splits/test.csv")
    generated = pd.read_csv(data_root / "generated_manifest.csv")
    if len(generated) != 3752 or set(generated["source_image_id"].astype(str)) - set(train_df["image_id"].astype(str)):
        raise RuntimeError("Unexpected LoRA generated manifest or non-training source")
    generated_train = generated[["image_id", "image_path", "label", "label_name", "dataset_name"]]
    augmented_df = pd.concat([train_df, generated_train], ignore_index=True)
    real_counts = train_df["label_name"].value_counts().reindex(APTOS_CLASS_ORDER).astype(int).tolist()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(
        "lora_sd025_augmented_ce", "LoRA-SD 0.25 Augmented", "real+lora_sd025", "traditional", "cross_entropy"
    )
    lora_result = train_experiment(
        experiment, augmented_df, val_df, test_df, real_counts, args, output_dir, device
    )
    fixed_results = json.loads((fixed_results_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
    detailed_fixed: List[Dict[str, object]] = []
    for row in fixed_results:
        detailed_fixed.append(
            json.loads((fixed_results_dir / row["experiment_key"] / "test_metrics.json").read_text(encoding="utf-8"))
        )
    all_results = [*detailed_fixed, lora_result]
    ordinary_sd = next(row for row in detailed_fixed if row["experiment_key"] == "sd025_augmented_ce")
    comparison_rows = [
        {
            key: row[key]
            for key in [
                "experiment_key",
                "display_name",
                "train_size",
                "epochs_ran",
                "best_val_macro_f1",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "macro_auprc",
                "minority_avg_recall",
                "quadratic_weighted_kappa",
                "training_seconds",
            ]
        }
        for row in all_results
    ]
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
    write_json(output_dir / "comparison_metrics.json", comparison_rows)
    make_summary_plots(all_results, output_dir)
    write_report(args, output_dir, all_results, ordinary_sd, lora_result)
    print(f"LoRA classifier completed. Report: {(ROOT / args.report).resolve()}", flush=True)


if __name__ == "__main__":
    main()
