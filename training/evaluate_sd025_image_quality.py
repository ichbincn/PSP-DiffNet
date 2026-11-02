from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.generation_metrics import (
    DistributionMetricConfig,
    build_inception_feature_extractor,
    build_vgg16_feature_extractor,
    compute_distribution_metrics,
    extract_features,
)


MINORITY_CLASSES = ["Mild", "Moderate", "Severe", "Proliferate_DR"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UNIT_MEAN = [0.0, 0.0, 0.0]
UNIT_STD = [1.0, 1.0, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate APTOS SD 0.25 images using FID, KID and Improved P/R")
    parser.add_argument("--manifest", default="outputs/aptos2019/sd025_balanced_train/generated_manifest.csv")
    parser.add_argument("--train-split", default="outputs/aptos2019/sd025_balanced_train/splits/train.csv")
    parser.add_argument("--output-dir", default="outputs/aptos2019/sd025_image_quality_fid_kid_ipr")
    parser.add_argument("--report", default="SD_APTOS全量图像质量评价报告.md")
    parser.add_argument("--method-name", default="普通 SD 0.25")
    parser.add_argument("--reproduce-command", default="bash scripts/run_sd025_image_quality.sh")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--ipr-k", type=int, default=3)
    parser.add_argument("--distance-chunk-size", type=int, default=256)
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


def validate_inputs(manifest: pd.DataFrame, real: pd.DataFrame) -> None:
    required_generated = {"image_path", "label_name"}
    required_real = {"image_path", "label_name"}
    if not required_generated.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required_generated - set(manifest.columns))}")
    if not required_real.issubset(real.columns):
        raise ValueError(f"Train split missing columns: {sorted(required_real - set(real.columns))}")
    if set(manifest["label_name"].unique()) != set(MINORITY_CLASSES):
        raise ValueError(f"Unexpected generated classes: {sorted(manifest['label_name'].unique())}")
    for path in list(manifest["image_path"]) + list(real["image_path"]):
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def plot_metrics(rows: List[Dict[str, object]], path: Path) -> None:
    class_rows = [row for row in rows if row["scope"] in MINORITY_CLASSES]
    names = [str(row["scope"]) for row in class_rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, [float(row["fid"]) for row in class_rows], color="#4C78A8")
    axes[0].set_title("FID ↓")
    axes[1].bar(x, [float(row["kid_mean"]) * 1000 for row in class_rows], color="#F58518")
    axes[1].set_title("KID × 1000 ↓")
    axes[2].bar(x - 0.18, [float(row["improved_precision"]) for row in class_rows], 0.36, label="Precision")
    axes[2].bar(x + 0.18, [float(row["improved_recall"]) for row in class_rows], 0.36, label="Recall")
    axes[2].set_title("Improved Precision / Recall ↑")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, names, rotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_row(row: Dict[str, object]) -> str:
    return (
        f"| {row['scope']} | {row['real_count']} | {row['generated_count']} | "
        f"{float(row['fid']):.4f} | {float(row['kid_mean']):.6f} ± {float(row['kid_std']):.6f} | "
        f"{float(row['improved_precision']):.4f} | {float(row['improved_recall']):.4f} |"
    )


def write_report(
    path: Path,
    rows: List[Dict[str, object]],
    config: DistributionMetricConfig,
    output_dir: Path,
    elapsed_minutes: float,
    device: torch.device,
    method_name: str,
    reproduce_command: str,
) -> None:
    by_scope = {str(row["scope"]): row for row in rows}
    overall = by_scope["Overall_minority"]
    macro = by_scope["Macro_class"]
    worst_fid = max((by_scope[name] for name in MINORITY_CLASSES), key=lambda row: float(row["fid"]))
    lowest_precision = min((by_scope[name] for name in MINORITY_CLASSES), key=lambda row: float(row["improved_precision"]))
    lowest_recall = min((by_scope[name] for name in MINORITY_CLASSES), key=lambda row: float(row["improved_recall"]))
    lines = [
        f"# APTOS {method_name} 全量生成图像质量评价报告（FID、KID、IPR）",
        "",
        "## 1. 本轮任务与结论",
        "",
        f"本轮对全部 {method_name} 少数类生成图，统一使用 FID、KID 和 Kynkäänniemi Improved Precision/Recall（IPR）评价，不混入旧版类别一致性、手工类内多样性、最近邻距离、Grad-CAM一致性、SSIM或PSNR。",
        "",
        f"整体少数类结果为：FID={float(overall['fid']):.4f}，KID={float(overall['kid_mean']):.6f}±{float(overall['kid_std']):.6f}，Improved Precision={float(overall['improved_precision']):.4f}，Improved Recall={float(overall['improved_recall']):.4f}。",
        "",
        f"这些指标没有跨数据集通用的合格阈值。当前结果对应 {method_name}；所有方法必须使用完全相同的真实参考集、预处理、特征网络和代码比较相对变化。",
        "",
        "## 2. 数据与比较口径",
        "",
        f"- 真实参考图：APTOS 固定训练划分中的四个少数类，共 {overall['real_count']} 张。",
        f"- 生成图：{method_name} 的全量补齐结果，共 {overall['generated_count']} 张。",
        "- Overall_minority：将四个少数类合并后计算，反映整体少数类分布。",
        "- 逐类别：每类生成图只与同类别真实训练图比较。",
        "- Macro_class：四个逐类指标的算术平均，避免 Moderate 等样本较多类别主导总结果。",
        "- No_DR 没有参与，因为本轮没有为该类生成补齐图像。",
        "",
        "## 3. 指标原理",
        "",
        "- FID：用 PyTorch-FID 的 TensorFlow兼容 Inception-v3 pool3 2048维特征分别拟合真实与生成分布的均值和协方差，再计算 Fréchet 距离；越低表示整体特征分布越接近。",
        "- KID：在同一 Inception 特征上用三次多项式核计算无偏 MMD²；越低越好。报告100次随机子集估计的均值和标准差。",
        "- Improved Precision：生成样本落入真实样本 k近邻流形的比例，主要反映 fidelity，越高越好。",
        "- Improved Recall：真实样本落入生成样本 k近邻流形的比例，主要反映 coverage，越高越好。",
        "- IPR不是分类任务的precision/recall。本实现按原始定义使用VGG-16第二个全连接层后的4096维激活，k=3。",
        "",
        "## 4. 全量结果",
        "",
        "| 范围 | 真实数 | 生成数 | FID ↓ | KID均值±标准差 ↓ | Improved Precision ↑ | Improved Recall ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        format_row(overall),
        format_row(macro),
        "",
        "## 5. 逐类别结果",
        "",
        "| 类别 | 真实数 | 生成数 | FID ↓ | KID均值±标准差 ↓ | Improved Precision ↑ | Improved Recall ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[format_row(by_scope[name]) for name in MINORITY_CLASSES],
        "",
        "## 6. 结果分析",
        "",
        f"- 宏平均为 FID={float(macro['fid']):.4f}、KID={float(macro['kid_mean']):.6f}、Precision={float(macro['improved_precision']):.4f}、Recall={float(macro['improved_recall']):.4f}。宏平均应与整体结果同时报告，因为两者回答的问题不同。",
        f"- FID最高的类别是 {worst_fid['scope']}（{float(worst_fid['fid']):.4f}），说明它在 Inception 特征空间与同类真实图的分布差异最大。",
        f"- Improved Precision最低的是 {lowest_precision['scope']}（{float(lowest_precision['improved_precision']):.4f}），它的生成样本落入真实流形的比例最低，主要提示 fidelity 问题。",
        f"- Improved Recall最低的是 {lowest_recall['scope']}（{float(lowest_recall['improved_recall']):.4f}），说明生成分布对该类真实变化的覆盖最弱。",
        "- Precision和Recall必须分开解释：高Precision、低Recall通常表示图像看起来集中且逼真但覆盖不足；低Precision、高Recall可能表示覆盖较广但包含偏离真实流形的样本。",
        "- Severe只有135张真实参考图，逐类FID和IPR对样本构成较敏感；因此不能用不同真实子集重复计算后直接横向比较。后续所有方法必须复用本次固定train.csv。",
        "- Inception-v3和VGG-16均在ImageNet自然图像上预训练，所以结果只证明相应特征空间中的分布关系，不能直接等同于医学病灶正确性。",
        "",
        "## 7. 训练与评价策略说明",
        "",
        "本轮只做离线评价，不训练或微调生成模型。所有真实图与生成图先采用同一眼底ROI裁黑边和背景遮罩，再送入固定特征网络。FID/KID使用PyTorch-FID的TensorFlow兼容Inception-v3；IPR使用ImageNet预训练VGG-16。特征网络始终eval且不更新参数。",
        "",
        f"KID设置为{subset_desc(config)}；IPR使用k={config.ipr_k}，欧氏距离和自适应k近邻半径；随机种子固定为{config.seed}。FID使用经验均值和无偏样本协方差，并通过等价低秩公式稳定计算协方差平方根迹。",
        "",
        "## 8. 输出位置",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 汇总JSON：`{output_dir / 'summary_metrics.json'}`",
        f"- 逐范围CSV：`{output_dir / 'metrics_by_scope.csv'}`",
        f"- 特征缓存：`{output_dir / 'feature_cache.npz'}`",
        f"- 指标图：`{output_dir / 'distribution_metrics_by_class.png'}`",
        f"- 运行配置：`{output_dir / 'run_config.json'}`",
        "",
        "## 9. 用户复现方法",
        "",
        "```bash",
        reproduce_command,
        "```",
        "",
        "另开终端查看实时阶段、样本进度和GPU占用：",
        "",
        "```bash",
        "bash scripts/watch_sd025_image_quality.sh",
        "```",
        "",
        "## 10. 运行环境与耗时",
        "",
        f"- Python：{platform.python_version()}；PyTorch：{torch.__version__}。",
        f"- 设备：{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}。",
        f"- 总耗时：{elapsed_minutes:.2f}分钟。",
        "- 首次运行若本地没有ImageNet权重，会下载Inception-v3和VGG-16权重；后续直接复用缓存。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def subset_desc(config: DistributionMetricConfig) -> str:
    return f"{config.kid_subsets}次随机子集，最大子集大小={config.kid_subset_size}，逐类不足时使用该类可用的较小样本数"


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "live_progress.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the full FID/KID/IPR evaluation")

    config = DistributionMetricConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        ipr_k=args.ipr_k,
        distance_chunk_size=args.distance_chunk_size,
        seed=args.seed,
    )

    def progress(stage: str, completed: int, total: int) -> None:
        write_json(
            progress_path,
            {
                "status": "running",
                "stage": stage,
                "completed": completed,
                "total": total,
                "elapsed_minutes": (time.time() - start_time) / 60.0,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    manifest = pd.read_csv(ROOT / args.manifest)
    real_all = pd.read_csv(ROOT / args.train_split)
    real = real_all[real_all["label_name"].isin(MINORITY_CLASSES)].reset_index(drop=True)
    manifest = manifest.reset_index(drop=True)
    validate_inputs(manifest, real)
    real_paths = real["image_path"].astype(str).tolist()
    generated_paths = manifest["image_path"].astype(str).tolist()

    progress("loading_inception", 0, len(real_paths) + len(generated_paths))
    inception = build_inception_feature_extractor(device)
    real_inception = extract_features(
        real_paths, inception, 299, UNIT_MEAN, UNIT_STD, device, config, "extract_real_inception", progress
    )
    generated_inception = extract_features(
        generated_paths, inception, 299, UNIT_MEAN, UNIT_STD, device, config, "extract_generated_inception", progress
    )
    del inception
    torch.cuda.empty_cache()

    progress("loading_vgg16", 0, len(real_paths) + len(generated_paths))
    vgg = build_vgg16_feature_extractor(device)
    real_vgg = extract_features(
        real_paths, vgg, 224, IMAGENET_MEAN, IMAGENET_STD, device, config, "extract_real_vgg16", progress
    )
    generated_vgg = extract_features(
        generated_paths, vgg, 224, IMAGENET_MEAN, IMAGENET_STD, device, config, "extract_generated_vgg16", progress
    )
    del vgg
    torch.cuda.empty_cache()

    np.savez_compressed(
        output_dir / "feature_cache.npz",
        real_inception=real_inception,
        generated_inception=generated_inception,
        real_vgg16=real_vgg,
        generated_vgg16=generated_vgg,
        real_labels=real["label_name"].to_numpy(),
        generated_labels=manifest["label_name"].to_numpy(),
        real_paths=np.asarray(real_paths),
        generated_paths=np.asarray(generated_paths),
    )

    rows: List[Dict[str, object]] = []
    scopes = [("Overall_minority", np.ones(len(real), dtype=bool), np.ones(len(manifest), dtype=bool))]
    for class_name in MINORITY_CLASSES:
        scopes.append(
            (
                class_name,
                real["label_name"].to_numpy() == class_name,
                manifest["label_name"].to_numpy() == class_name,
            )
        )
    for index, (scope, real_mask, generated_mask) in enumerate(scopes):
        progress(f"compute_{scope}", index, len(scopes))
        metrics = compute_distribution_metrics(
            real_inception[real_mask],
            generated_inception[generated_mask],
            real_vgg[real_mask],
            generated_vgg[generated_mask],
            device=device,
            config=config,
            seed_offset=index * 1000,
        )
        rows.append({"scope": scope, **metrics})

    class_rows = rows[1:]
    macro = {
        "scope": "Macro_class",
        "real_count": int(sum(int(row["real_count"]) for row in class_rows)),
        "generated_count": int(sum(int(row["generated_count"]) for row in class_rows)),
        "fid": float(np.mean([float(row["fid"]) for row in class_rows])),
        "kid_mean": float(np.mean([float(row["kid_mean"]) for row in class_rows])),
        "kid_std": float(np.mean([float(row["kid_std"]) for row in class_rows])),
        "kid_subset_size": "per_class",
        "improved_precision": float(np.mean([float(row["improved_precision"]) for row in class_rows])),
        "improved_recall": float(np.mean([float(row["improved_recall"]) for row in class_rows])),
    }
    rows.insert(1, macro)
    elapsed_minutes = (time.time() - start_time) / 60.0
    summary = {
        "protocol": {
            "real_reference": "fixed APTOS train split, minority classes only",
            "preprocessing": "fundus ROI crop and background mask; FID/KID inputs in [0,1]; VGG-16 inputs use ImageNet normalization",
            "fid_kid_feature_extractor": "PyTorch-FID TensorFlow-compatible Inception-v3, 2048D pool3 features",
            "ipr_feature_extractor": "torchvision VGG-16 ImageNet, 4096D activation after second FC",
            "kid_subsets": config.kid_subsets,
            "kid_max_subset_size": config.kid_subset_size,
            "ipr_k": config.ipr_k,
            "seed": config.seed,
        },
        "metrics": {str(row["scope"]): {key: value for key, value in row.items() if key != "scope"} for row in rows},
        "runtime": {
            "elapsed_minutes": elapsed_minutes,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    write_json(output_dir / "summary_metrics.json", summary)
    write_csv(output_dir / "metrics_by_scope.csv", rows)
    write_json(output_dir / "run_config.json", vars(args))
    plot_metrics(rows, output_dir / "distribution_metrics_by_class.png")
    write_report(
        (ROOT / args.report).resolve(),
        rows,
        config,
        Path(args.output_dir),
        elapsed_minutes,
        device,
        args.method_name,
        args.reproduce_command,
    )
    write_json(
        progress_path,
        {
            "status": "completed",
            "stage": "completed",
            "completed": len(scopes),
            "total": len(scopes),
            "elapsed_minutes": elapsed_minutes,
            "summary": summary["metrics"]["Overall_minority"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
