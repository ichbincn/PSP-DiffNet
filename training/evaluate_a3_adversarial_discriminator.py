from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.train_a3_discriminator_sanity import MINORITY, PriorDataset, evaluate_aux, grouped_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate online A3 D on source-group-held-out real/A1 pairs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--real-root", default="outputs/aptos2019/structure_guidance_v3/balanced_train")
    parser.add_argument("--a1-manifest", default="outputs/aptos2019/structure_guidance_v3/balanced_train/generated_manifest.csv")
    parser.add_argument("--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train = pd.read_csv(ROOT / args.real_root / "splits/train.csv")
    prior_index = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(prior_index.image_id.astype(str), prior_index.prior_path.astype(str)))
    real = train[train.label_name.isin(MINORITY)].copy()
    real["prior_path"] = real.image_id.astype(str).map(prior_map)
    fake = pd.read_csv(ROOT / args.a1_manifest, encoding="utf-8-sig").copy()
    fake["source_image_id"] = fake.source_image_id.astype(str)
    real["is_cal"] = real.image_id.astype(str).map(lambda value: grouped_calibration(value, args.seed))
    fake["is_cal"] = fake.source_image_id.map(lambda value: grouped_calibration(value, args.seed))
    real_cal = real[real.is_cal].copy()
    fake_cal = fake[fake.is_cal].copy()
    real_train_sources = set(real.loc[~real.is_cal, "image_id"].astype(str))
    cal_sources = set(real_cal.image_id.astype(str))
    fake_train_sources = set(fake.loc[~fake.is_cal, "source_image_id"].astype(str))
    fake_cal_sources = set(fake_cal.source_image_id.astype(str))
    overlap = {
        "real_train_vs_real_cal": len(real_train_sources & cal_sources),
        "fake_train_vs_fake_cal": len(fake_train_sources & fake_cal_sources),
        "real_train_vs_fake_cal": len(real_train_sources & fake_cal_sources),
        "real_cal_vs_fake_train": len(cal_sources & fake_train_sources),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Source-group leakage: {overlap}")
    real_loader = DataLoader(
        PriorDataset(real_cal, args.image_size, fake=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    fake_loader = DataLoader(
        PriorDataset(fake_cal, args.image_size, fake=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu")
    model = A3DiagnosticDiscriminator(num_classes=5).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    metrics = evaluate_aux(model, real_loader, fake_loader, torch.device("cuda"))
    payload = {
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "seed": args.seed,
        "real_calibration_images": len(real_cal),
        "fake_calibration_images": len(fake_cal),
        "source_overlap": overlap,
        "metrics": metrics,
        "checks": {
            "realism_auroc_ge_0p60": metrics["realism_auroc"] >= 0.60,
            "structure_auroc_ge_0p75": metrics["structure_auroc"] >= 0.75,
            "prior_dependency_positive": metrics["prior_shuffle_positive_score_drop"] > 0,
            "shortcut_correlation_le_0p70": metrics["realism_shortcut_max_abs_correlation"] <= 0.70,
        },
        "test_data_used": False,
    }
    payload["all_checks_pass"] = all(payload["checks"].values())
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
