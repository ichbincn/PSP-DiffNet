from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image
from transformers import get_cosine_schedule_with_warmup

from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.run_aptos_convnext_baselines import build_model
from training.train_diffusion_lora import PROMPTS
from utils.fundus import preprocess_fundus_pil


TEACHERS = {
    "original_ce": "outputs/aptos2019/convnext_tiny_scratch_preliminary/original_ce/best_model.pt",
    "balanced_softmax": "outputs/aptos2019/convnext_tiny_scratch_preliminary/balanced_softmax/best_model.pt",
    "focal_loss": "outputs/aptos2019/convnext_tiny_scratch_preliminary/focal_loss/best_model.pt",
}
MINORITY_LABELS = {1, 2, 3, 4}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INITIAL_WEIGHTS = {"adv": 0.05, "cls": 0.10, "struct": 0.05, "ord": 0.02}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3-static predicted-clean diagnostic consistency training")
    parser.add_argument("--mode", choices=["probe", "audit", "small", "full"], required=True)
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument(
        "--a0-lora-dir", default="outputs/aptos2019/lora_sd025_experiment/lora_training/final_lora"
    )
    parser.add_argument("--train-split", default="outputs/aptos2019/structure_guidance_v3/balanced_train/splits/train.csv")
    parser.add_argument(
        "--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv"
    )
    parser.add_argument(
        "--discriminator-checkpoint",
        default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/initialization/discriminator_seed_2026/best_discriminator.pt",
    )
    parser.add_argument("--resolved-weights", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/initialization/gradient_probe/resolved_weights.json")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--diagnostic-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=2560)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--checkpoint-steps", type=int, default=50)
    parser.add_argument("--aux-max-timestep", type=int, default=500)
    parser.add_argument("--class-margin", type=float, default=1.0)
    parser.add_argument(
        "--discriminator-class-weight",
        type=float,
        default=0.0,
        help="Optional weight of the discriminator's online real-trained class head in class/ordinal generator feedback.",
    )
    parser.add_argument("--anchor-strength", type=float, default=0.08)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--probe-timestep", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class A3Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, prior_map: Dict[str, str], tokenizer, image_size: int, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.prior_map = prior_map
        self.image_size = image_size
        self.train = train
        self.token_ids = {
            class_name: tokenizer(
                PROMPTS[class_name],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            for class_name in APTOS_CLASS_ORDER
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        image = preprocess_fundus_pil(image, crop=True, mask_background=True).image
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image_tensor = transforms.functional.to_tensor(image)
        image_id = str(row.image_id)
        prior_path = self.prior_map.get(image_id)
        if prior_path:
            prior_image = Image.open(prior_path).convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            prior = transforms.functional.to_tensor(prior_image).clamp(0.0, 1.0)
        else:
            prior = torch.zeros(1, self.image_size, self.image_size)
        flipped = False
        if self.train and torch.rand(()) < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            prior = torch.flip(prior, dims=[2])
            flipped = True
        pixel_values = image_tensor.mul(2.0).sub(1.0)
        label = int(row.label)
        return pixel_values, self.token_ids[str(row.label_name)], label, prior, image_id, flipped


def balanced_indices(frame: pd.DataFrame, samples: int, seed: int) -> List[int]:
    if samples % len(APTOS_CLASS_ORDER):
        raise ValueError("samples-per-epoch must be divisible by five")
    rng = np.random.default_rng(seed)
    per_class = samples // len(APTOS_CLASS_ORDER)
    indices: List[int] = []
    for label in range(len(APTOS_CLASS_ORDER)):
        choices = frame.index[frame.label.astype(int) == label].to_numpy()
        indices.extend(rng.choice(choices, per_class, replace=len(choices) < per_class).tolist())
    rng.shuffle(indices)
    return indices


def diagnostic_input(images_01: torch.Tensor, size: int) -> torch.Tensor:
    resized = F.interpolate(images_01, size=(size, size), mode="bilinear", align_corners=False)
    mean = resized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


def load_components(args: argparse.Namespace, device: torch.device):
    pipeline = StableDiffusionPipeline.from_pretrained(
        (ROOT / args.model_dir).resolve(),
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(device)
    pipeline.load_lora_weights((ROOT / args.a0_lora_dir).resolve(), adapter_name="a3")
    pipeline.set_adapters("a3", adapter_weights=1.0)
    unet, vae, text_encoder, tokenizer = pipeline.unet, pipeline.vae, pipeline.text_encoder, pipeline.tokenizer
    scheduler = DDPMScheduler.from_pretrained((ROOT / args.model_dir).resolve(), subfolder="scheduler", local_files_only=True)
    vae.requires_grad_(False); vae.eval()
    text_encoder.requires_grad_(False); text_encoder.eval()
    unet.requires_grad_(False)
    trainable = []
    for name, parameter in unet.named_parameters():
        if "lora_" in name.lower():
            parameter.requires_grad_(True)
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("A0 LoRA loaded but no trainable LoRA parameter was found")
    unet.enable_gradient_checkpointing()

    discriminator_checkpoint = torch.load(ROOT / args.discriminator_checkpoint, map_location="cpu")
    discriminator = A3DiagnosticDiscriminator(num_classes=5)
    discriminator.load_state_dict(discriminator_checkpoint["model_state"], strict=True)
    discriminator.requires_grad_(False); discriminator.eval(); discriminator.to(device)

    teachers = []
    for name, relative in TEACHERS.items():
        checkpoint = torch.load(ROOT / relative, map_location="cpu")
        if checkpoint.get("pretrained") is not False:
            raise RuntimeError("A3 teacher must be trained with torchvision weights=None: " + name)
        model = build_model(torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.requires_grad_(False); model.eval(); model.to(device)
        teachers.append((name, model))
    return pipeline, unet, vae, text_encoder, tokenizer, scheduler, discriminator, teachers, trainable


def predicted_clean_losses(
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    priors: torch.Tensor,
    flipped: torch.Tensor,
    teacher,
    unet,
    vae,
    text_encoder,
    scheduler,
    discriminator,
    args,
    forced_timestep: Optional[int] = None,
    return_live_details: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor]:
    with torch.no_grad(), autocast(enabled=True):
        source_latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        hidden = text_encoder(input_ids)[0]
        noise = torch.randn_like(source_latents)
        if forced_timestep is None:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (source_latents.shape[0],), device=source_latents.device).long()
        else:
            timesteps = torch.full((source_latents.shape[0],), forced_timestep, device=source_latents.device, dtype=torch.long)
        noisy = scheduler.add_noise(source_latents, noise, timesteps)
    with autocast(enabled=True):
        predicted_noise = unet(noisy, timesteps, hidden).sample
        if scheduler.config.prediction_type == "v_prediction":
            diffusion_target = scheduler.get_velocity(source_latents, noise, timesteps)
        else:
            diffusion_target = noise
        diff_loss = F.mse_loss(predicted_noise.float(), diffusion_target.float())

        alpha = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=predicted_noise.dtype)[timesteps]
        alpha = alpha.view(-1, 1, 1, 1)
        clean_latents = (noisy - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-4)
        gate = F.interpolate(priors, size=clean_latents.shape[-2:], mode="bilinear", align_corners=False).sqrt()
        anchored = clean_latents + args.anchor_strength * gate * (source_latents - clean_latents)
        decoded = vae.decode(anchored / vae.config.scaling_factor).sample
        images_01 = decoded.mul(0.5).add(0.5).clamp(0.0, 1.0)
        # A0's diffusion augmentation uses horizontal flips, while the frozen
        # structure head was calibrated in canonical fundus orientation. Undo
        # only the diagnostic view so a valid joint image/prior flip is not
        # mistaken for a structural mismatch; diffusion still trains on the
        # augmented view exactly as before.
        diagnostic_images = images_01.clone()
        diagnostic_priors = priors.clone()
        flip_mask = flipped.to(device=images_01.device, dtype=torch.bool)
        if flip_mask.any():
            diagnostic_images[flip_mask] = torch.flip(diagnostic_images[flip_mask], dims=[3])
            diagnostic_priors[flip_mask] = torch.flip(diagnostic_priors[flip_mask], dims=[3])
        d_input = diagnostic_input(diagnostic_images, args.diagnostic_size)
        mid, feature, discriminator_class_logits = discriminator.encode(d_input)
        realism_logits = discriminator.realism_from_features(feature, labels)
        prior_224 = F.interpolate(diagnostic_priors, size=(args.diagnostic_size, args.diagnostic_size), mode="bilinear", align_corners=False)
        structure_logits = discriminator.structure_from_features(mid, prior_224)
        if return_live_details:
            teacher_map = teacher.features(d_input)
            teacher_pooled = teacher.avgpool(teacher_map)
            teacher_normalized = teacher.classifier[0](teacher_pooled)
            teacher_feature = torch.flatten(teacher_normalized, 1)
            teacher_logits = teacher.classifier[2](teacher_feature)
        else:
            teacher_feature = None
            teacher_logits = teacher(d_input)

        target_logits = teacher_logits.gather(1, labels[:, None]).squeeze(1)
        other_logits = teacher_logits.masked_fill(F.one_hot(labels, 5).bool(), -1e4).max(dim=1).values
        teacher_cls_loss = F.relu(args.class_margin - (target_logits - other_logits)).mean()
        probabilities = torch.softmax(teacher_logits, dim=1)
        discriminator_class_weight = float(getattr(args, "discriminator_class_weight", 0.0))
        if discriminator_class_weight > 0.0:
            d_target = discriminator_class_logits.gather(1, labels[:, None]).squeeze(1)
            d_other = discriminator_class_logits.masked_fill(F.one_hot(labels, 5).bool(), -1e4).max(dim=1).values
            discriminator_cls_loss = F.relu(args.class_margin - (d_target - d_other)).mean()
            cls_loss = (
                (1.0 - discriminator_class_weight) * teacher_cls_loss
                + discriminator_class_weight * discriminator_cls_loss
            )
            discriminator_probabilities = torch.softmax(discriminator_class_logits, dim=1)
            probabilities = (
                (1.0 - discriminator_class_weight) * probabilities
                + discriminator_class_weight * discriminator_probabilities
            )
        else:
            discriminator_cls_loss = teacher_cls_loss.detach() * 0.0
            cls_loss = teacher_cls_loss
        grade_axis = probabilities.new_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        expected_grade = (probabilities * grade_axis[None]).sum(dim=1)
        ord_loss = F.smooth_l1_loss(expected_grade, labels.float())
        adv_loss = F.softplus(-realism_logits).mean()
        minority = torch.tensor([int(value) in MINORITY_LABELS for value in labels.tolist()], device=labels.device)
        if minority.any():
            struct_loss = F.softplus(-structure_logits[minority]).mean()
        else:
            struct_loss = structure_logits.sum() * 0.0

    losses = {"diff": diff_loss, "adv": adv_loss, "cls": cls_loss, "struct": struct_loss, "ord": ord_loss}
    stats = {
        "timestep_mean": float(timesteps.float().mean().item()),
        "aux_active_fraction": float((timesteps <= args.aux_max_timestep).float().mean().item()),
        "realism_probability": float(torch.sigmoid(realism_logits).mean().detach().item()),
        "structure_probability": float(torch.sigmoid(structure_logits).mean().detach().item()),
        "target_probability": float(probabilities.gather(1, labels[:, None]).mean().detach().item()),
        "teacher_class_loss": float(teacher_cls_loss.detach().item()),
        "discriminator_class_loss": float(discriminator_cls_loss.detach().item()),
        "predicted_clean_min": float(images_01.detach().min().item()),
        "predicted_clean_max": float(images_01.detach().max().item()),
        "predicted_clean_std": float(images_01.detach().std().item()),
        "diagnostic_canonical_unflip_fraction": float(flip_mask.float().mean().item()),
    }
    if return_live_details:
        return losses, stats, images_01.detach(), {
            "diagnostic_input": d_input,
            "teacher_feature": teacher_feature,
        }
    return losses, stats, images_01.detach()


def gradient_vector_stats(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], retain: bool):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
    norm_sq = loss.new_zeros(())
    for grad in grads:
        if grad is not None:
            norm_sq = norm_sq + grad.float().pow(2).sum()
    return grads, float(norm_sq.sqrt().item())


def gradient_cosine(left, right) -> float:
    dot = 0.0; left_sq = 0.0; right_sq = 0.0
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        af = a.float(); bf = b.float()
        dot += float((af * bf).sum().item())
        left_sq += float(af.pow(2).sum().item())
        right_sq += float(bf.pow(2).sum().item())
    return dot / max(math.sqrt(left_sq * right_sq), 1e-12)


def run_probe(args, dataset, frame, components, output_dir, device):
    _, unet, vae, text_encoder, _, scheduler, discriminator, teachers, trainable = components
    rng = np.random.default_rng(args.seed)
    # The two clinically hardest APTOS tails must be represented in the formal
    # probe. Any additional probes cycle through the remaining minority grades.
    priority_labels = [3, 4, 1, 2]
    chosen = []
    for label in priority_labels[: args.probe_batches]:
        candidates = frame.index[frame.label.astype(int) == label].to_numpy()
        chosen.append(int(rng.choice(candidates)))
    loader = DataLoader(Subset(dataset, chosen), batch_size=1, shuffle=False, num_workers=0)
    rows = []
    norm_accumulator = {key: [] for key in ["diff", "adv", "cls", "struct", "ord"]}
    for batch_index, batch in enumerate(loader):
        pixel, ids, labels, priors, image_ids, flipped = batch
        pixel = pixel.to(device, dtype=torch.float16); ids = ids.to(device); labels = labels.to(device); priors = priors.to(device, dtype=torch.float16)
        teacher_name, teacher = teachers[batch_index % len(teachers)]
        losses, stats, _ = predicted_clean_losses(
            pixel, ids, labels, priors, flipped, teacher, unet, vae, text_encoder, scheduler, discriminator, args,
            forced_timestep=args.probe_timestep,
        )
        gradients = {}
        for index, key in enumerate(["diff", "adv", "cls", "struct", "ord"]):
            gradients[key], norm = gradient_vector_stats(losses[key], trainable, retain=index < 4)
            norm_accumulator[key].append(norm)
        row = {"batch": batch_index + 1, "image_id": image_ids[0], "label": int(labels.item()), "teacher": teacher_name, **stats}
        for key, value in losses.items():
            row[key + "_loss"] = float(value.detach().item())
            row[key + "_grad_norm"] = norm_accumulator[key][-1]
            if key != "diff":
                row[key + "_diff_grad_cosine"] = gradient_cosine(gradients[key], gradients["diff"])
        rows.append(row)
        append_csv(output_dir / "gradient_probe.csv", row)
        print("[A3 probe] {}/{} label={} teacher={} diff_grad={:.6f} cls_grad={:.6f}".format(
            batch_index + 1, len(chosen), int(labels.item()), teacher_name,
            norm_accumulator["diff"][-1], norm_accumulator["cls"][-1]), flush=True)
        del losses, gradients
        torch.cuda.empty_cache()
    mean_norms = {key: float(np.mean(values)) for key, values in norm_accumulator.items()}
    resolved = dict(INITIAL_WEIGHTS)
    for key in resolved:
        weighted = resolved[key] * mean_norms[key]
        limit = 0.20 * mean_norms["diff"]
        if weighted > limit and weighted > 0:
            resolved[key] *= limit / weighted
    payload = {
        "initial_weights": INITIAL_WEIGHTS,
        "resolved_weights": resolved,
        "mean_raw_gradient_norms": mean_norms,
        "rule": "each weighted auxiliary gradient capped once at 20% of diffusion gradient; no upward scaling",
        "probe_timestep": args.probe_timestep,
        "probe_batches": len(rows),
        "test_data_used": False,
        "rows": rows,
    }
    write_json(output_dir / "resolved_weights.json", payload)
    write_json(output_dir / "live_progress.json", {"status": "completed", **payload})
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def run_timestep_audit(args, dataset, frame, components, output_dir, device):
    _, unet, vae, text_encoder, _, scheduler, discriminator, teachers, _ = components
    rng = np.random.default_rng(args.seed)
    timesteps = [0, 25, 50, 100, 200, 300, 500]
    rows = []
    for label, teacher_index in [(3, 0), (4, 1)]:
        candidates = frame.index[frame.label.astype(int) == label].to_numpy()
        selected = int(rng.choice(candidates))
        batch = next(iter(DataLoader(Subset(dataset, [selected]), batch_size=1, shuffle=False, num_workers=0)))
        pixel, ids, labels, priors, image_ids, flipped = batch
        pixel = pixel.to(device, dtype=torch.float16); ids = ids.to(device); labels = labels.to(device)
        priors = priors.to(device, dtype=torch.float16)
        image_grid = [pixel.detach().float().mul(0.5).add(0.5).clamp(0.0, 1.0)]
        for timestep in timesteps:
            torch.manual_seed(args.seed + label * 1000 + timestep)
            torch.cuda.manual_seed_all(args.seed + label * 1000 + timestep)
            with torch.no_grad():
                losses, stats, predicted = predicted_clean_losses(
                    pixel, ids, labels, priors, flipped, teachers[teacher_index][1], unet, vae, text_encoder,
                    scheduler, discriminator, args, forced_timestep=timestep,
                )
            row = {
                "image_id": image_ids[0], "label": label, "label_name": APTOS_CLASS_ORDER[label],
                "teacher": teachers[teacher_index][0], "timestep": timestep, **stats,
                **{key + "_loss": float(value.detach().item()) for key, value in losses.items()},
            }
            rows.append(row)
            image_grid.append(predicted.detach().float())
        save_image(
            torch.cat(image_grid, dim=0),
            output_dir / ("{}_source_t0_t25_t50_t100_t200_t300_t500.png".format(APTOS_CLASS_ORDER[label])),
            nrow=len(image_grid),
        )
    pd.DataFrame(rows).to_csv(output_dir / "timestep_audit.csv", index=False)
    payload = {"status": "completed", "timesteps": timesteps, "rows": rows, "test_data_used": False}
    write_json(output_dir / "timestep_audit.json", payload)
    write_json(output_dir / "live_progress.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def save_lora(unet, directory: Path) -> None:
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name="a3"))
    StableDiffusionPipeline.save_lora_weights(directory, unet_lora_layers=state)


def save_training_checkpoint(path, unet, optimizer, lr_scheduler, scaler, epoch, batch, step, history):
    torch.save(
        {
            "lora_state": get_peft_model_state_dict(unet, adapter_name="a3"),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "global_step": step,
            "history": history,
        },
        path,
    )


def run_training(args, dataset, frame, components, output_dir, device):
    completed_path = output_dir / "completed.json"
    final_lora_path = output_dir / "final_lora/pytorch_lora_weights.safetensors"
    if completed_path.is_file() and final_lora_path.is_file():
        print("[A3 {}] completed result already exists: {}".format(args.mode, output_dir), flush=True)
        return
    _, unet, vae, text_encoder, _, scheduler, discriminator, teachers, trainable = components
    resolved_payload = json.loads((ROOT / args.resolved_weights).read_text(encoding="utf-8"))
    weights = {key: float(value) for key, value in resolved_payload["resolved_weights"].items()}
    micro_batches = math.ceil(args.samples_per_epoch / args.batch_size)
    steps_per_epoch = math.ceil(micro_batches / args.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * args.epochs
    if args.max_optimizer_steps:
        planned_steps = min(planned_steps, args.max_optimizer_steps)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(args.warmup_steps, max(planned_steps // 4, 1)), num_training_steps=planned_steps
    )
    scaler = GradScaler(enabled=True)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0; micro_step = 0; history = []; accumulation_records = []
    start_epoch = 0; start_batch = 0
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(unet, checkpoint["lora_state"], adapter_name="a3")
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch"])
        global_step = int(checkpoint["global_step"])
        micro_step = global_step * args.gradient_accumulation_steps
        history = list(checkpoint["history"])
        step_history_path = output_dir / "step_history.csv"
        if step_history_path.is_file():
            persisted = pd.read_csv(step_history_path)
            persisted = persisted[persisted.global_step.astype(int) <= global_step]
            persisted.to_csv(step_history_path, index=False, encoding="utf-8-sig")
        print(
            "[A3 {}] resume epoch={} batch={} global_step={}".format(
                args.mode, start_epoch + 1, start_batch, global_step
            ),
            flush=True,
        )
    started = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        loader = DataLoader(
            Subset(dataset, balanced_indices(frame, args.samples_per_epoch, args.seed + epoch)),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        unet.train()
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            pixel, ids, labels, priors, image_ids, flipped = batch
            pixel = pixel.to(device, dtype=torch.float16, non_blocking=True)
            ids = ids.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            priors = priors.to(device, dtype=torch.float16, non_blocking=True)
            teacher_name, teacher = teachers[micro_step % len(teachers)]
            losses, stats, predicted_images = predicted_clean_losses(
                pixel, ids, labels, priors, flipped, teacher, unet, vae, text_encoder, scheduler, discriminator, args
            )
            aux_active = stats["timestep_mean"] <= args.aux_max_timestep
            total = losses["diff"]
            if aux_active:
                for key in ["adv", "cls", "struct", "ord"]:
                    total = total + weights[key] * losses[key]
            scaler.scale(total / args.gradient_accumulation_steps).backward()
            micro_record = {**stats, "total_loss": float(total.detach().item())}
            for key, value in losses.items():
                micro_record[key + "_loss"] = float(value.detach().item())
                micro_record[key + "_weighted"] = (
                    float(value.detach().item())
                    * (1.0 if key == "diff" else weights[key])
                    * int(aux_active or key == "diff")
                )
            accumulation_records.append(micro_record)
            micro_step += 1
            boundary = micro_step % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm).item())
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); lr_scheduler.step()
                global_step += 1
                averaged = {
                    key: float(np.mean([record[key] for record in accumulation_records]))
                    for key in accumulation_records[0]
                }
                row = {
                    "global_step": global_step, "epoch": epoch + 1, "batch": batch_index + 1,
                    "image_id": image_ids[0], "label": int(labels[0].item()), "teacher": teacher_name,
                    "lora_grad_norm_before_clip": grad_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]), "gpu_peak_gib": float(torch.cuda.max_memory_allocated() / 2**30),
                    **averaged,
                }
                append_csv(output_dir / "step_history.csv", row)
                history.append(row)
                accumulation_records = []
                sample_interval = 20 if args.mode == "small" else 100
                if global_step == 1 or global_step % sample_interval == 0:
                    sample_dir = output_dir / "predicted_clean_samples"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    source_01 = pixel.detach().float().mul(0.5).add(0.5).clamp(0.0, 1.0)
                    save_image(
                        torch.cat([source_01, predicted_images.detach().float()], dim=0),
                        sample_dir / "step_{:05d}_source_then_predicted.png".format(global_step),
                        nrow=max(source_01.shape[0], 1),
                    )
                progress = {"status": "training", "mode": args.mode, "global_step": global_step, "planned_steps": planned_steps,
                            "epoch": epoch + 1, "loss": row["total_loss"], "elapsed_minutes": (time.time() - started) / 60.0}
                write_json(output_dir / "live_progress.json", progress)
                if global_step == 1 or global_step % 10 == 0:
                    print("[A3 {}] step={}/{} loss={:.5f} diff={:.5f} cls={:.5f} real={:.3f} target_p={:.3f} vram={:.2f}GiB".format(
                        args.mode, global_step, planned_steps, row["total_loss"], row["diff_loss"], row["cls_loss"],
                        row["realism_probability"], row["target_probability"], row["gpu_peak_gib"]), flush=True)
                if global_step % args.checkpoint_steps == 0:
                    save_training_checkpoint(checkpoint_path, unet, optimizer, lr_scheduler, scaler,
                                             epoch, batch_index + 1, global_step, history)
                if global_step >= planned_steps:
                    stop = True
            del losses, total
            if stop:
                break
        if stop:
            break
        start_batch = 0
    final_lora = output_dir / "final_lora"
    save_lora(unet, final_lora)
    completed = {
        "status": "completed", "mode": args.mode, "global_step": global_step, "planned_steps": planned_steps,
        "elapsed_minutes": (time.time() - started) / 60.0, "resolved_weights": weights,
        "final_lora": str(final_lora.resolve()), "test_data_used": False,
    }
    write_json(completed_path, completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A3-static requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    default_outputs = {
        "probe": "outputs/aptos2019/reproductions/a3_static_helper/gradient_probe",
        "audit": "outputs/aptos2019/reproductions/a3_static_helper/timestep_audit",
        "small": "outputs/aptos2019/reproductions/a3_static_helper/small_sanity",
        "full": "outputs/aptos2019/reproductions/a3_static_helper/joint_training",
    }
    output_dir = (ROOT / (args.output_dir or default_outputs[args.mode])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"output_dir_resolved": str(output_dir), "device": torch.cuda.get_device_name(0), "test_data_used": False})
    write_json(output_dir / "run_config.json", config)
    frame = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    prior_frame = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(prior_frame.image_id.astype(str), prior_frame.prior_path.astype(str)))
    components = load_components(args, device)
    dataset = A3Dataset(frame, prior_map, components[4], args.image_size, train=True)
    write_json(output_dir / "model_audit.json", {
        "sd_backbone_frozen": True, "vae_frozen": True, "text_encoder_frozen": True,
        "lora_trainable_parameters": sum(p.numel() for p in components[-1]),
        "class_branch": "one frozen member per microbatch, deterministic round-robin expectation over three weights=None teachers",
        "diagnostic_aux_checkpoint": str((ROOT / args.discriminator_checkpoint).resolve()),
        "structure_prior": "A1 Grad-CAM weak prior; sqrt(P), alpha=0.08",
    })
    if args.mode == "probe":
        run_probe(args, dataset, frame, components, output_dir, device)
    elif args.mode == "audit":
        run_timestep_audit(args, dataset, frame, components, output_dir, device)
    else:
        run_training(args, dataset, frame, components, output_dir, device)


if __name__ == "__main__":
    main()
