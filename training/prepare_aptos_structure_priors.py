from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from training.run_aptos_convnext_baselines import IMAGENET_MEAN, IMAGENET_STD, build_model
from utils.fundus import preprocess_fundus_pil


MINORITY_CLASSES = APTOS_CLASS_ORDER[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-safe ConvNeXt Grad-CAM priors for APTOS train")
    parser.add_argument(
        "--train-split",
        default="outputs/aptos2019/lora_sd025_experiment/scale_sweep/scale_0p8/balanced_train/splits/train.csv",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default="outputs/aptos2019/convnext_tiny_scratch_preliminary/original_ce/best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-layer-index", type=int, default=5, help="ConvNeXt feature stage; 5 gives a 14x14 semantic map")
    parser.add_argument("--audit-overlays-per-class", type=int, default=4)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=MINORITY_CLASSES,
        choices=APTOS_CLASS_ORDER,
        help="Classes whose train-only Grad-CAM priors are created.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image: torch.Tensor, target_class: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[:, target_class].sum().backward()
        weights = self.gradients.float().mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations.float()).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        minimum, maximum = heatmap.min(), heatmap.max()
        return (heatmap - minimum) / (maximum - minimum + 1e-6)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS), dtype=np.float32)
    heat = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    color = np.zeros_like(base)
    color[..., 0] = heat * 255.0
    color[..., 1] = np.clip(1.0 - np.abs(heat - 0.5) * 2.0, 0.0, 1.0) * 200.0
    return Image.fromarray(np.clip(0.68 * base + 0.32 * color, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Grad-CAM prior preparation requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "prior_index.csv"
    completed_path = output_dir / "completed.json"
    requested_classes = list(args.classes)
    if completed_path.is_file() and index_path.is_file():
        existing = pd.read_csv(index_path)
        expected_existing = int(existing["label_name"].isin(requested_classes).sum())
        if (
            len(existing) == expected_existing
            and set(existing["label_name"].astype(str)) == set(requested_classes)
            and all(Path(path).is_file() for path in existing["prior_path"])
        ):
            print(f"Structure priors already completed: {output_dir}", flush=True)
            return

    split = pd.read_csv(ROOT / args.train_split)
    frame = split[split["label_name"].isin(requested_classes)].reset_index(drop=True)
    expected_count = int(split["label_name"].isin(requested_classes).sum())
    if len(frame) != expected_count or expected_count == 0:
        raise RuntimeError(f"Unexpected prior input count: expected {expected_count}, got {len(frame)}")
    checkpoint_path = (ROOT / args.classifier_checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("pretrained") is not False or checkpoint.get("experiment", {}).get("key") != "original_ce":
        raise RuntimeError("Structure priors must use the frozen scratch Original CE classifier")
    model = build_model(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if not 0 <= args.target_layer_index < len(model.features):
        raise ValueError("target-layer-index is outside ConvNeXt features")
    gradcam = GradCAM(model, model.features[args.target_layer_index])
    classifier_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    audit_ids = set()
    for class_name in requested_classes:
        rows = frame[frame["label_name"] == class_name]
        audit_ids.update(rows.sample(n=min(args.audit_overlays_per_class, len(rows)), random_state=args.seed)["image_id"].astype(str))

    records = []
    started = time.time()
    for position, row in frame.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        processed = preprocess_fundus_pil(image, crop=True, mask_background=True)
        tensor = classifier_transform(processed.image).unsqueeze(0).to(device)
        heatmap = gradcam.generate(tensor, int(row["label"])).detach().cpu().numpy()
        heatmap = np.asarray(
            Image.fromarray((heatmap * 255.0).astype(np.uint8)).resize(processed.image.size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        fundus_mask = processed.mask.astype(np.float32)
        if fundus_mask.shape != heatmap.shape:
            fundus_mask = np.asarray(
                Image.fromarray((fundus_mask * 255).astype(np.uint8)).resize(processed.image.size, Image.Resampling.NEAREST),
                dtype=np.float32,
            ) / 255.0
        heatmap *= fundus_mask
        maximum = float(heatmap.max())
        if maximum > 1e-8:
            heatmap /= maximum
        class_dir = output_dir / "maps" / str(row["label_name"])
        class_dir.mkdir(parents=True, exist_ok=True)
        prior_path = class_dir / f"{row['image_id']}_gradcam.png"
        Image.fromarray(np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)).save(prior_path)
        overlay_path = ""
        if str(row["image_id"]) in audit_ids:
            audit_dir = output_dir / "audit_overlays" / str(row["label_name"])
            audit_dir.mkdir(parents=True, exist_ok=True)
            saved_overlay = audit_dir / f"{row['image_id']}_overlay.png"
            overlay(processed.image, heatmap).save(saved_overlay)
            overlay_path = str(saved_overlay)
        positive = heatmap[fundus_mask > 0.5]
        records.append(
            {
                "image_id": str(row["image_id"]),
                "label": int(row["label"]),
                "label_name": str(row["label_name"]),
                "source_path": str(Path(row["image_path"]).resolve()),
                "prior_path": str(prior_path),
                "audit_overlay_path": overlay_path,
                "prior_mean": float(positive.mean()) if positive.size else 0.0,
                "prior_std": float(positive.std()) if positive.size else 0.0,
                "coverage_gt_0p5": float((positive > 0.5).mean()) if positive.size else 0.0,
            }
        )
        done = position + 1
        if done % 25 == 0 or done == len(frame):
            write_json(
                output_dir / "live_progress.json",
                {"stage": "priors", "status": "running", "completed": done, "total": len(frame), "elapsed_seconds": time.time() - started},
            )
            print(f"Structure prior progress: {done}/{len(frame)}", flush=True)
    gradcam.close()
    index = pd.DataFrame(records)
    index.to_csv(index_path, index=False)
    summary = {
        "status": "completed",
        "count": len(index),
        "classifier_checkpoint": str(checkpoint_path),
        "classifier_initialization": checkpoint["initialization"],
        "classifier_train_source": "fixed real train only",
        "target_layer": f"torchvision ConvNeXt-Tiny features[{args.target_layer_index}]",
        "native_cam_resolution": "14x14 when target_layer_index=5",
        "per_class": index["label_name"].value_counts().sort_index().to_dict(),
        "prior_mean": float(index["prior_mean"].mean()),
        "prior_std": float(index["prior_std"].mean()),
        "coverage_gt_0p5": float(index["coverage_gt_0p5"].mean()),
        "classes": requested_classes,
        "filtering": f"none; all {len(index)} requested train priors are retained",
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(completed_path, summary)
    write_json(output_dir / "live_progress.json", {**summary, "stage": "priors"})
    print(f"Structure priors completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
