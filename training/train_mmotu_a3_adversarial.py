from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.run_mmotu_convnext import build_model
from training.train_mmotu_lora import PROMPTS, balanced_indices
from utils.ultrasound import preprocess_ultrasound_mask, preprocess_ultrasound_pil


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = len(MMOTU_CLASS_ORDER)
STRUCTURE_SUPERVISION_DESCRIPTION = "all classes with valid train GT masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMOTU A3 online alternating diagnostic-adversarial LoRA")
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--a0-lora-dir", default="outputs/mmotu2d/a0_lora/lora_training/final_lora")
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--prior-index", default="outputs/mmotu2d/a1_structure/gt_mask_priors/prior_index.csv")
    parser.add_argument("--teacher-root", default="outputs/mmotu2d/convnext_real_baseline/formal")
    parser.add_argument(
        "--teacher-checkpoints", default="",
        help="Optional comma-separated train/validation-selected teacher checkpoints; overrides teacher-root/seed_*.",
    )
    parser.add_argument("--teacher-initialization", choices=["scratch", "imagenet"], default="scratch")
    parser.add_argument("--teacher-aggregation", choices=["rotate", "mean_logits"], default="rotate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--diagnostic-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--aux-max-timestep", type=int, default=100)
    parser.add_argument("--separate-aux-interval", type=int, default=4)
    parser.add_argument("--class-margin", type=float, default=1.0)
    parser.add_argument("--anchor-strength", type=float, default=0.08)
    parser.add_argument("--gate-power", type=float, default=0.5)
    parser.add_argument("--adv-weight", type=float, default=0.05)
    parser.add_argument("--class-weight", type=float, default=0.04)
    parser.add_argument("--structure-weight", type=float, default=0.05)
    parser.add_argument("--discriminator-class-weight", type=float, default=0.25)
    parser.add_argument("--d-head-lr", type=float, default=5e-5)
    parser.add_argument("--d-backbone-lr", type=float, default=2e-6)
    parser.add_argument("--d-weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-update-interval", type=int, default=4)
    parser.add_argument("--d-warmup-g-steps", type=int, default=25)
    parser.add_argument("--d-batch-size", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=64)
    parser.add_argument("--d-class-weight", type=float, default=0.25)
    parser.add_argument("--d-structure-fake-weight", type=float, default=0.5)
    parser.add_argument("--r1-gamma", type=float, default=1e-4)
    parser.add_argument("--r1-interval", type=int, default=16)
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


def calibration_source(image_id: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 == 0


def diagnostic_input(images_01: torch.Tensor, size: int) -> torch.Tensor:
    resized = F.interpolate(images_01, size=(size, size), mode="bilinear", align_corners=False)
    mean = resized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


class MMOTUA3Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, prior_map: Dict[str, str], tokenizer, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.prior_map = prior_map
        self.image_size = image_size
        self.token_ids = {
            name: tokenizer(
                PROMPTS[name], max_length=tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt"
            ).input_ids[0]
            for name in MMOTU_CLASS_ORDER
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as handle:
            image = preprocess_ultrasound_pil(handle.convert("RGB"), self.image_size)
        image_tensor = transforms.functional.to_tensor(image)
        image_id = str(row.image_id)
        prior_path = self.prior_map[image_id]
        with Image.open(prior_path) as handle:
            prior = transforms.functional.to_tensor(preprocess_ultrasound_mask(handle, self.image_size)).clamp(0, 1)
        flipped = bool(torch.rand(()) < 0.5)
        if flipped:
            image_tensor = torch.flip(image_tensor, dims=[2])
            prior = torch.flip(prior, dims=[2])
        return (
            image_tensor.mul(2).sub(1), self.token_ids[str(row.label_name)], int(row.label),
            prior, image_id, flipped,
        )


def load_components(args: argparse.Namespace, device: torch.device):
    model_dir = (ROOT / args.model_dir).resolve()
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_dir, variant="fp16", torch_dtype=torch.float16, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False, local_files_only=True,
    ).to(device)
    pipeline.load_lora_weights((ROOT / args.a0_lora_dir).resolve(), adapter_name="a3")
    pipeline.set_adapters("a3", adapter_weights=1.0)
    unet, vae = pipeline.unet, pipeline.vae
    text_encoder, tokenizer = pipeline.text_encoder, pipeline.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)
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
        raise RuntimeError("A0 LoRA was loaded, but no trainable LoRA parameter was found")
    unet.enable_gradient_checkpointing()

    teacher_root = (ROOT / args.teacher_root).resolve()
    if args.teacher_checkpoints:
        teacher_paths = [(ROOT / value.strip()).resolve() for value in args.teacher_checkpoints.split(",") if value.strip()]
    else:
        teacher_paths = [teacher_root / f"seed_{seed}/best_model.pt" for seed in (2026, 2027, 2028)]
    if len(teacher_paths) != 3:
        raise RuntimeError(f"MMOTU A3 requires exactly three frozen teachers, got {len(teacher_paths)}")
    expected_pretrained = args.teacher_initialization == "imagenet"
    teachers = []
    for path in teacher_paths:
        checkpoint = torch.load(path, map_location="cpu")
        if bool(checkpoint.get("pretrained")) != expected_pretrained or checkpoint.get("num_classes") != NUM_CLASSES:
            raise RuntimeError(f"Invalid {args.teacher_initialization} MMOTU teacher checkpoint: {path}")
        teacher = build_model(torch.device("cpu"))
        teacher.load_state_dict(checkpoint["model_state"], strict=True)
        teacher.requires_grad_(False); teacher.eval(); teacher.to(device)
        teachers.append((path.parent.name, teacher))

    discriminator = A3DiagnosticDiscriminator(num_classes=NUM_CLASSES)
    discriminator.load_medical_teacher(torch.load(teacher_paths[0], map_location="cpu")["model_state"])
    discriminator.to(device)
    discriminator.requires_grad_(False); discriminator.eval()
    return pipeline, unet, vae, text_encoder, tokenizer, noise_scheduler, discriminator, teachers, trainable


def set_generator_view(discriminator: torch.nn.Module) -> None:
    discriminator.requires_grad_(False)
    discriminator.eval()


def set_discriminator_view(discriminator: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    discriminator.requires_grad_(False)
    head_modules = (
        discriminator.realism_label_embedding, discriminator.realism_head,
        discriminator.structure_head, discriminator.backbone.classifier,
    )
    for module in (*head_modules, discriminator.backbone.features[6], discriminator.backbone.features[7]):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    heads: List[torch.nn.Parameter] = []
    for module in head_modules:
        heads.extend(module.parameters())
    backbone = list(discriminator.backbone.features[6].parameters()) + list(discriminator.backbone.features[7].parameters())
    discriminator.train()
    for index in range(6):
        discriminator.backbone.features[index].eval()
    return heads, backbone


def predicted_clean(
    pixels: torch.Tensor, ids: torch.Tensor, labels: torch.Tensor, priors: torch.Tensor,
    flipped: torch.Tensor, teacher_models: Sequence[torch.nn.Module], unet: torch.nn.Module,
    vae: torch.nn.Module, text_encoder: torch.nn.Module, scheduler, discriminator: torch.nn.Module,
    args: argparse.Namespace, forced_timestep: Optional[int] = None,
):
    with torch.no_grad(), autocast(enabled=True):
        source_latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        hidden = text_encoder(ids)[0]
        noise = torch.randn_like(source_latents)
        if forced_timestep is None:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (len(pixels),), device=pixels.device).long()
        else:
            timesteps = torch.full((len(pixels),), forced_timestep, device=pixels.device, dtype=torch.long)
        noisy = scheduler.add_noise(source_latents, noise, timesteps)
    with autocast(enabled=True):
        predicted_noise = unet(noisy, timesteps, hidden).sample
        target = scheduler.get_velocity(source_latents, noise, timesteps) if scheduler.config.prediction_type == "v_prediction" else noise
        diffusion_loss = F.mse_loss(predicted_noise.float(), target.float())
        alpha = scheduler.alphas_cumprod.to(timesteps.device, predicted_noise.dtype)[timesteps].view(-1, 1, 1, 1)
        clean = (noisy - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-4)
        gate = F.interpolate(priors, size=clean.shape[-2:], mode="nearest").clamp(0, 1).pow(args.gate_power)
        anchored = clean + args.anchor_strength * gate * (source_latents - clean)
        images = vae.decode(anchored / vae.config.scaling_factor).sample.mul(0.5).add(0.5).clamp(0, 1)

        canonical_images, canonical_priors = images.clone(), priors.clone()
        flip_mask = flipped.to(device=images.device, dtype=torch.bool)
        if flip_mask.any():
            canonical_images[flip_mask] = torch.flip(canonical_images[flip_mask], dims=[3])
            canonical_priors[flip_mask] = torch.flip(canonical_priors[flip_mask], dims=[3])
        d_input = diagnostic_input(canonical_images, args.diagnostic_size)
        mid, feature, d_class_logits = discriminator.encode(d_input)
        realism_logits = discriminator.realism_from_features(feature, labels)
        prior_224 = F.interpolate(canonical_priors, (args.diagnostic_size, args.diagnostic_size), mode="bilinear", align_corners=False)
        structure_logits = discriminator.structure_from_features(mid, prior_224)
        teacher_logits = torch.stack([teacher(d_input) for teacher in teacher_models], dim=0).mean(dim=0)

        one_hot = F.one_hot(labels, NUM_CLASSES).bool()
        teacher_target = teacher_logits.gather(1, labels[:, None]).squeeze(1)
        teacher_other = teacher_logits.masked_fill(one_hot, -1e4).max(1).values
        teacher_class = F.relu(args.class_margin - (teacher_target - teacher_other)).mean()
        d_target = d_class_logits.gather(1, labels[:, None]).squeeze(1)
        d_other = d_class_logits.masked_fill(one_hot, -1e4).max(1).values
        d_class = F.relu(args.class_margin - (d_target - d_other)).mean()
        class_loss = (1 - args.discriminator_class_weight) * teacher_class + args.discriminator_class_weight * d_class
        probabilities = (
            (1 - args.discriminator_class_weight) * torch.softmax(teacher_logits, 1)
            + args.discriminator_class_weight * torch.softmax(d_class_logits, 1)
        )
        losses = {
            "diff": diffusion_loss,
            "adv": F.softplus(-realism_logits).mean(),
            "cls": class_loss,
            "struct": F.softplus(-structure_logits).mean(),
        }
        stats = {
            "timestep_mean": float(timesteps.float().mean().item()),
            "realism_probability": float(torch.sigmoid(realism_logits).mean().detach().item()),
            "structure_probability": float(torch.sigmoid(structure_logits).mean().detach().item()),
            "target_probability": float(probabilities.gather(1, labels[:, None]).mean().detach().item()),
            "teacher_class_loss": float(teacher_class.detach().item()),
            "discriminator_class_loss": float(d_class.detach().item()),
        }
    return losses, stats, images.detach()


def diffusion_only(pixels, ids, unet, vae, text_encoder, scheduler):
    """Uniform-timestep diffusion objective without an unnecessary diagnostic decode."""
    with torch.no_grad(), autocast(enabled=True):
        source_latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        hidden = text_encoder(ids)[0]
        noise = torch.randn_like(source_latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (len(pixels),), device=pixels.device
        ).long()
        noisy = scheduler.add_noise(source_latents, noise, timesteps)
    with autocast(enabled=True):
        predicted_noise = unet(noisy, timesteps, hidden).sample
        target = scheduler.get_velocity(source_latents, noise, timesteps) if scheduler.config.prediction_type == "v_prediction" else noise
        loss = F.mse_loss(predicted_noise.float(), target.float())
    return loss, float(timesteps.float().mean().item())


def canonical_replay(pixels, fake, labels, priors, flipped) -> Dict[str, torch.Tensor]:
    real = pixels.detach().float().mul(0.5).add(0.5).clamp(0, 1)
    fake = fake.detach().float().clamp(0, 1)
    prior = priors.detach().float().clamp(0, 1)
    mask = flipped.to(dtype=torch.bool)
    if mask.any():
        real[mask] = torch.flip(real[mask], dims=[3])
        fake[mask] = torch.flip(fake[mask], dims=[3])
        prior[mask] = torch.flip(prior[mask], dims=[3])
    return {"real": real.cpu().half(), "fake": fake.cpu().half(), "labels": labels.cpu(), "priors": prior.cpu().half()}


def sample_replay(replay: Sequence[Dict[str, torch.Tensor]], size: int, rng, device):
    picks = rng.choice(len(replay), size=size, replace=len(replay) < size)
    chosen = [replay[int(i)] for i in picks]
    return (
        torch.cat([x["real"] for x in chosen]).to(device=device, dtype=torch.float32),
        torch.cat([x["fake"] for x in chosen]).to(device=device, dtype=torch.float32),
        torch.cat([x["labels"] for x in chosen]).to(device=device, dtype=torch.long),
        torch.cat([x["priors"] for x in chosen]).to(device=device, dtype=torch.float32),
    )


def discriminator_step(discriminator, optimizer, replay, rng, args, device, d_step):
    heads, backbone = set_discriminator_view(discriminator)
    trainable = heads + backbone
    real, fake, labels, priors = sample_replay(replay, args.d_batch_size, rng, device)
    do_r1 = args.r1_gamma > 0 and (d_step + 1) % args.r1_interval == 0
    real.requires_grad_(do_r1)
    real_input, fake_input = diagnostic_input(real, args.diagnostic_size), diagnostic_input(fake, args.diagnostic_size)
    optimizer.zero_grad(set_to_none=True)
    real_mid, real_feature, real_class = discriminator.encode(real_input)
    fake_mid, fake_feature, _ = discriminator.encode(fake_input)
    real_logits = discriminator.realism_from_features(real_feature, labels)
    fake_logits = discriminator.realism_from_features(fake_feature, labels)
    prior_224 = F.interpolate(priors, (args.diagnostic_size, args.diagnostic_size), mode="bilinear", align_corners=False)
    mismatch = torch.roll(prior_224, shifts=(31, 47), dims=(2, 3))
    real_structure = discriminator.structure_from_features(real_mid, prior_224)
    mismatch_structure = discriminator.structure_from_features(real_mid, mismatch)
    fake_structure = discriminator.structure_from_features(fake_mid, prior_224)
    realism_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    structure_loss = (
        F.softplus(-real_structure).mean() + 0.5 * F.softplus(mismatch_structure).mean()
        + args.d_structure_fake_weight * F.softplus(fake_structure).mean()
    )
    class_loss = F.cross_entropy(real_class, labels)
    total = realism_loss + structure_loss + args.d_class_weight * class_loss
    r1 = total.new_zeros(())
    if do_r1:
        gradient = torch.autograd.grad(real_logits.sum(), real, create_graph=True, retain_graph=True)[0]
        r1 = gradient.flatten(1).pow(2).sum(1).mean()
        total = total + 0.5 * args.r1_gamma * args.r1_interval * r1
    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
    optimizer.step()
    result = {
        "d_step": d_step + 1, "d_total_loss": float(total.detach()),
        "d_realism_loss": float(realism_loss.detach()), "d_structure_loss": float(structure_loss.detach()),
        "d_class_loss": float(class_loss.detach()), "d_r1_penalty": float(r1.detach()),
        "d_real_probability": float(torch.sigmoid(real_logits).mean().detach()),
        "d_fake_probability": float(torch.sigmoid(fake_logits).mean().detach()),
        "d_real_structure_probability": float(torch.sigmoid(real_structure).mean().detach()),
        "d_fake_structure_probability": float(torch.sigmoid(fake_structure).mean().detach()),
        "d_grad_norm_before_clip": grad_norm,
    }
    set_generator_view(discriminator)
    return result


def save_lora(unet: torch.nn.Module, directory: Path) -> None:
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet, adapter_name="a3"))
    StableDiffusionPipeline.save_lora_weights(directory, unet_lora_layers=state)


def save_checkpoint(path, unet, discriminator, g_optimizer, d_optimizer, g_scheduler, scaler, epoch, batch, g_step, d_step):
    torch.save({
        "lora_state": get_peft_model_state_dict(unet, adapter_name="a3"),
        "discriminator_state": discriminator.state_dict(), "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(), "g_scheduler": g_scheduler.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "batch": batch,
        "g_step": g_step, "d_step": d_step,
    }, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU A3 requires CUDA")
    if args.samples_per_epoch % NUM_CLASSES:
        raise ValueError(f"samples-per-epoch must be divisible by {NUM_CLASSES}")
    seed_everything(args.seed)
    device = torch.device("cuda")
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_lora = output / "final_lora"
    if (output / "completed.json").is_file() and (final_lora / "pytorch_lora_weights.safetensors").is_file():
        print(f"[MMOTU A3] completed: {output}", flush=True)
        return
    config = vars(args).copy()
    config.update({
        "test_accessed": False, "pretrained_classifier": False, "ordinal_loss_weight": 0.0,
        "structure_supervision_classes": STRUCTURE_SUPERVISION_DESCRIPTION,
        "adversarial_definition": "online delayed replay, alternating D then G through current frozen-D view",
        "source_holdout": "deterministic 20% hash partition excluded from D replay",
    })
    write_json(output / "run_config.json", config)

    frame = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    prior_frame = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(prior_frame.image_id.astype(str), prior_frame.prior_path.astype(str)))
    if frame.empty or set(frame.image_id.astype(str)) - set(prior_map):
        raise RuntimeError("Fixed medical train split/prior index mismatch")
    components = load_components(args, device)
    pipeline, unet, vae, text_encoder, tokenizer, noise_scheduler, discriminator, teachers, g_parameters = components
    dataset = MMOTUA3Dataset(frame, prior_map, tokenizer, args.image_size)

    d_heads, d_backbone = set_discriminator_view(discriminator)
    d_optimizer = torch.optim.AdamW([
        {"params": d_heads, "lr": args.d_head_lr}, {"params": d_backbone, "lr": args.d_backbone_lr},
    ], weight_decay=args.d_weight_decay, betas=(0.0, 0.99))
    set_generator_view(discriminator)
    g_optimizer = torch.optim.AdamW(g_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    planned_steps = math.ceil(args.samples_per_epoch / args.batch_size / args.gradient_accumulation_steps) * args.epochs
    if args.max_optimizer_steps:
        planned_steps = min(planned_steps, args.max_optimizer_steps)
    g_scheduler = get_cosine_schedule_with_warmup(
        g_optimizer, num_warmup_steps=min(args.warmup_steps, max(planned_steps // 4, 1)),
        num_training_steps=planned_steps,
    )
    scaler = GradScaler(enabled=True)
    write_json(output / "model_audit.json", {
        "sd_backbone_frozen": True, "vae_frozen": True, "text_encoder_frozen": True,
        "lora_trainable_parameters": sum(x.numel() for x in g_parameters),
        "discriminator_initialization": f"{args.teacher_initialization} real-only ConvNeXt-Tiny first frozen teacher checkpoint",
        "teachers": [name for name, _ in teachers], "teacher_pretrained": args.teacher_initialization == "imagenet",
        "teacher_aggregation": args.teacher_aggregation,
        "online_d_trainable": ["three heads", "class head", "ConvNeXt feature blocks 6-7"],
        "generator_gradient_through_discriminator": True, "ordinal_head_or_loss": False,
        "weights": {"adv": args.adv_weight, "class": args.class_weight, "structure": args.structure_weight},
        "test_accessed": False,
    })

    checkpoint_path = output / "checkpoint_latest.pt"
    start_epoch = start_batch = g_step = d_step = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(unet, checkpoint["lora_state"], adapter_name="a3")
        discriminator.load_state_dict(checkpoint["discriminator_state"], strict=True)
        g_optimizer.load_state_dict(checkpoint["g_optimizer"]); d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        g_scheduler.load_state_dict(checkpoint["g_scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, start_batch = int(checkpoint["epoch"]), int(checkpoint["batch"])
        g_step, d_step = int(checkpoint["g_step"]), int(checkpoint["d_step"])
        print(f"[MMOTU A3] resume epoch={start_epoch + 1} batch={start_batch} g={g_step} d={d_step}", flush=True)

    replay: Deque[Dict[str, torch.Tensor]] = deque(maxlen=args.replay_capacity)
    rng = np.random.default_rng(args.seed + 7341)
    g_optimizer.zero_grad(set_to_none=True)
    micro_step = g_step * args.gradient_accumulation_steps
    accumulation: List[Dict[str, float]] = []
    last_d = {key: 0.0 for key in (
        "d_step", "d_total_loss", "d_realism_loss", "d_structure_loss", "d_class_loss", "d_r1_penalty",
        "d_real_probability", "d_fake_probability", "d_real_structure_probability",
        "d_fake_structure_probability", "d_grad_norm_before_clip",
    )}
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
            if (micro_step % args.gradient_accumulation_steps == 0 and g_step >= args.d_warmup_g_steps
                    and g_step % args.d_update_interval == 0 and len(replay) >= args.d_batch_size):
                last_d = discriminator_step(discriminator, d_optimizer, list(replay), rng, args, device, d_step)
                d_step += 1
                append_csv(output / "discriminator_history.csv", last_d)
            pixels, ids, labels, priors, image_ids, flipped = batch
            pixels = pixels.to(device, dtype=torch.float16, non_blocking=True)
            ids = ids.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            priors = priors.to(device, dtype=torch.float16, non_blocking=True); flipped = flipped.to(device)
            if args.teacher_aggregation == "mean_logits":
                teacher_name = "mean_logits(" + ",".join(name for name, _ in teachers) + ")"
                teacher_models = [teacher for _, teacher in teachers]
            else:
                teacher_name, teacher = teachers[micro_step % len(teachers)]
                teacher_models = [teacher]
            separate_aux = args.separate_aux_interval > 0 and micro_step % args.separate_aux_interval == 0
            if args.separate_aux_interval > 0:
                diffusion_loss, diffusion_timestep = diffusion_only(
                    pixels, ids, unet, vae, text_encoder, noise_scheduler
                )
            else:
                diff_losses, diff_stats, images = predicted_clean(
                    pixels, ids, labels, priors, flipped, teacher_models, unet, vae, text_encoder,
                    noise_scheduler, discriminator, args,
                )
                diffusion_loss, diffusion_timestep = diff_losses["diff"], diff_stats["timestep_mean"]
            if separate_aux:
                forced = int((args.seed + 73 * micro_step) % (args.aux_max_timestep + 1))
                aux_losses, aux_stats, images = predicted_clean(
                    pixels, ids, labels, priors, flipped, teacher_models, unet, vae, text_encoder,
                    noise_scheduler, discriminator, args, forced_timestep=forced,
                )
                losses = {"diff": diffusion_loss, **{key: aux_losses[key] for key in ("adv", "cls", "struct")}}
                stats = {**aux_stats, "diffusion_timestep_mean": diffusion_timestep}
                aux_active = True
            elif args.separate_aux_interval > 0:
                zero = diffusion_loss.detach() * 0.0
                losses = {"diff": diffusion_loss, "adv": zero, "cls": zero, "struct": zero}
                stats = {
                    "timestep_mean": diffusion_timestep, "diffusion_timestep_mean": diffusion_timestep,
                    "realism_probability": 0.0, "structure_probability": 0.0,
                    "target_probability": 0.0, "teacher_class_loss": 0.0,
                    "discriminator_class_loss": 0.0,
                }
                aux_active = False
            else:
                losses, stats = diff_losses, diff_stats
                aux_active = args.separate_aux_interval <= 0 and stats["timestep_mean"] <= args.aux_max_timestep
            total = losses["diff"]
            if aux_active:
                total = total + args.adv_weight * losses["adv"] + args.class_weight * losses["cls"] + args.structure_weight * losses["struct"]
                replay_mask = torch.tensor(
                    [not calibration_source(str(image_id), args.seed) for image_id in image_ids],
                    device=device, dtype=torch.bool,
                )
                if replay_mask.any():
                    replay.append(canonical_replay(
                        pixels[replay_mask], images[replay_mask], labels[replay_mask], priors[replay_mask], flipped[replay_mask]
                    ))
            scaler.scale(total / args.gradient_accumulation_steps).backward()
            accumulation.append({
                **stats, "total_loss": float(total.detach()), "diff_loss": float(losses["diff"].detach()),
                "adv_loss": float(losses["adv"].detach()), "cls_loss": float(losses["cls"].detach()),
                "struct_loss": float(losses["struct"].detach()), "aux_active": float(aux_active),
            })
            micro_step += 1
            boundary = micro_step % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if boundary:
                scaler.unscale_(g_optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(g_parameters, args.max_grad_norm).item())
                scaler.step(g_optimizer); scaler.update(); g_optimizer.zero_grad(set_to_none=True); g_scheduler.step()
                g_step += 1
                averaged = {key: float(np.mean([row[key] for row in accumulation])) for key in accumulation[0]}
                row = {
                    "global_step": g_step, "d_step": d_step, "epoch": epoch + 1, "batch": batch_index + 1,
                    "teacher": teacher_name, "replay_size": len(replay), "g_grad_norm_before_clip": grad_norm,
                    "g_learning_rate": float(g_optimizer.param_groups[0]["lr"]),
                    "gpu_peak_gib": float(torch.cuda.max_memory_allocated() / 2**30), **averaged, **last_d,
                }
                append_csv(output / "step_history.csv", row); accumulation = []
                progress = {
                    "status": "training", "mode": args.mode, "global_step": g_step, "planned_steps": planned_steps,
                    "d_step": d_step, "replay_size": len(replay), "loss": row["total_loss"],
                    "G_realism_probability": row["realism_probability"], "D_real_probability": row["d_real_probability"],
                    "D_fake_probability": row["d_fake_probability"], "target_probability": row["target_probability"],
                    "elapsed_minutes": (time.time() - started) / 60,
                }
                write_json(output / "live_progress.json", progress)
                if g_step == 1 or g_step % 10 == 0:
                    print(
                        f"[MMOTU A3] g={g_step}/{planned_steps} d={d_step} total={row['total_loss']:.5f} "
                        f"diff={row['diff_loss']:.5f} aux={row['aux_active']:.2f} Greal={row['realism_probability']:.3f} "
                        f"Dreal={row['d_real_probability']:.3f} Dfake={row['d_fake_probability']:.3f} "
                        f"target={row['target_probability']:.3f} vram={row['gpu_peak_gib']:.2f}GiB", flush=True,
                    )
                if g_step % args.checkpoint_steps == 0:
                    save_checkpoint(checkpoint_path, unet, discriminator, g_optimizer, d_optimizer, g_scheduler,
                                    scaler, epoch, batch_index + 1, g_step, d_step)
                if g_step >= planned_steps:
                    stop = True
            if stop:
                break
        if stop:
            break
        start_batch = 0

    save_lora(unet, final_lora)
    torch.save({
        "model_state": discriminator.state_dict(), "online_d_steps": d_step, "pretrained": False,
        "num_classes": NUM_CLASSES, "class_names": MMOTU_CLASS_ORDER, "test_data_used": False,
    }, output / "final_online_discriminator.pt")
    completed = {
        "status": "completed", "mode": args.mode, "global_step": g_step, "planned_steps": planned_steps,
        "d_step": d_step, "elapsed_minutes": (time.time() - started) / 60,
        "final_lora": str(final_lora.resolve()),
        "final_discriminator": str((output / "final_online_discriminator.pt").resolve()),
        "test_accessed": False, "ordinal_loss_weight": 0.0,
    }
    write_json(output / "completed.json", completed); write_json(output / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)
    del pipeline


if __name__ == "__main__":
    main()
