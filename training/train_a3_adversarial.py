from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_a3_static import (
    A3Dataset,
    INITIAL_WEIGHTS,
    balanced_indices,
    diagnostic_input,
    load_components,
    predicted_clean_losses,
    save_lora,
)
from training.train_a3_discriminator_sanity import grouped_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 online alternating diagnostic-adversarial LoRA training")
    parser.add_argument("--mode", choices=["small", "full"], required=True)
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--a0-lora-dir", default="outputs/aptos2019/lora_sd025_experiment/lora_training/final_lora")
    parser.add_argument("--train-split", default="outputs/aptos2019/structure_guidance_v3/balanced_train/splits/train.csv")
    parser.add_argument("--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv")
    parser.add_argument(
        "--discriminator-checkpoint",
        default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/initialization/discriminator_seed_2026/best_discriminator.pt",
    )
    parser.add_argument(
        "--resolved-weights",
        default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/initialization/gradient_probe/resolved_weights.json",
    )
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--aux-max-timestep", type=int, default=100)
    parser.add_argument(
        "--separate-aux-interval",
        type=int,
        default=0,
        help=(
            "If positive, evaluate an additional deterministic low-noise predicted-clean view "
            "every N micro-batches for diagnostic losses/replay while keeping the diffusion "
            "view uniformly sampled. Zero preserves the original single-view protocol."
        ),
    )
    parser.add_argument("--class-margin", type=float, default=1.0)
    parser.add_argument("--anchor-strength", type=float, default=0.08)
    parser.add_argument("--discriminator-class-weight", type=float, default=0.25)
    parser.add_argument("--adv-weight", type=float, default=-1.0)
    parser.add_argument("--class-weight", type=float, default=-1.0)
    parser.add_argument("--structure-weight", type=float, default=-1.0)
    parser.add_argument("--ordinal-weight", type=float, default=-1.0)
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


def set_generator_view(discriminator: torch.nn.Module) -> None:
    discriminator.requires_grad_(False)
    discriminator.eval()


def set_discriminator_view(discriminator: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    discriminator.requires_grad_(False)
    for module in (
        discriminator.realism_label_embedding,
        discriminator.realism_head,
        discriminator.structure_head,
        discriminator.backbone.classifier,
        discriminator.backbone.features[6],
        discriminator.backbone.features[7],
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    head_parameters = list(discriminator.realism_label_embedding.parameters())
    head_parameters += list(discriminator.realism_head.parameters())
    head_parameters += list(discriminator.structure_head.parameters())
    head_parameters += list(discriminator.backbone.classifier.parameters())
    backbone_parameters = list(discriminator.backbone.features[6].parameters())
    backbone_parameters += list(discriminator.backbone.features[7].parameters())
    discriminator.train()
    for index in range(6):
        discriminator.backbone.features[index].eval()
    return head_parameters, backbone_parameters


def canonical_replay_item(
    source_minus1_1: torch.Tensor,
    fake_01: torch.Tensor,
    labels: torch.Tensor,
    priors: torch.Tensor,
    flipped: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    real = source_minus1_1.detach().float().mul(0.5).add(0.5).clamp(0.0, 1.0)
    fake = fake_01.detach().float().clamp(0.0, 1.0)
    prior = priors.detach().float().clamp(0.0, 1.0)
    flip_mask = flipped.to(dtype=torch.bool)
    if flip_mask.any():
        real[flip_mask] = torch.flip(real[flip_mask], dims=[3])
        fake[flip_mask] = torch.flip(fake[flip_mask], dims=[3])
        prior[flip_mask] = torch.flip(prior[flip_mask], dims=[3])
    return {
        "real": real.cpu().half(),
        "fake": fake.cpu().half(),
        "labels": labels.detach().cpu().long(),
        "priors": prior.cpu().half(),
    }


def sample_replay(
    replay: Sequence[Dict[str, torch.Tensor]], batch_size: int, rng: np.random.Generator, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = rng.choice(len(replay), size=batch_size, replace=len(replay) < batch_size)
    chosen = [replay[int(index)] for index in indices]
    real = torch.cat([item["real"] for item in chosen], dim=0).to(device=device, dtype=torch.float32)
    fake = torch.cat([item["fake"] for item in chosen], dim=0).to(device=device, dtype=torch.float32)
    labels = torch.cat([item["labels"] for item in chosen], dim=0).to(device=device)
    priors = torch.cat([item["priors"] for item in chosen], dim=0).to(device=device, dtype=torch.float32)
    return real, fake, labels, priors


def discriminator_step(
    discriminator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[Dict[str, torch.Tensor]],
    rng: np.random.Generator,
    args: argparse.Namespace,
    device: torch.device,
    d_step: int,
) -> Dict[str, float]:
    head_parameters, backbone_parameters = set_discriminator_view(discriminator)
    trainable = head_parameters + backbone_parameters
    real_01, fake_01, labels, priors = sample_replay(replay, args.d_batch_size, rng, device)
    do_r1 = args.r1_gamma > 0 and (d_step + 1) % args.r1_interval == 0
    real_01.requires_grad_(do_r1)
    real_input = diagnostic_input(real_01, args.diagnostic_size)
    fake_input = diagnostic_input(fake_01, args.diagnostic_size)
    optimizer.zero_grad(set_to_none=True)

    real_mid, real_feature, real_class_logits = discriminator.encode(real_input)
    fake_mid, fake_feature, _ = discriminator.encode(fake_input)
    real_logits = discriminator.realism_from_features(real_feature, labels)
    fake_logits = discriminator.realism_from_features(fake_feature, labels)
    prior_224 = F.interpolate(priors, size=(args.diagnostic_size, args.diagnostic_size), mode="bilinear", align_corners=False)
    mismatch = torch.roll(prior_224, shifts=(31, 47), dims=(2, 3))
    real_structure = discriminator.structure_from_features(real_mid, prior_224)
    mismatch_structure = discriminator.structure_from_features(real_mid, mismatch)
    fake_structure = discriminator.structure_from_features(fake_mid, prior_224)

    realism_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    structure_mask = labels != 0
    if structure_mask.any():
        structure_loss = F.softplus(-real_structure[structure_mask]).mean()
        structure_loss = structure_loss + 0.5 * F.softplus(mismatch_structure[structure_mask]).mean()
        structure_loss = structure_loss + args.d_structure_fake_weight * F.softplus(fake_structure[structure_mask]).mean()
    else:
        structure_loss = real_structure.sum() * 0.0
    class_loss = F.cross_entropy(real_class_logits, labels)
    total = realism_loss + structure_loss + args.d_class_weight * class_loss
    r1_penalty = total.new_zeros(())
    if do_r1:
        gradients = torch.autograd.grad(real_logits.sum(), real_01, create_graph=True, retain_graph=True)[0]
        r1_penalty = gradients.flatten(1).pow(2).sum(1).mean()
        total = total + 0.5 * args.r1_gamma * args.r1_interval * r1_penalty
    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
    optimizer.step()
    stats = {
        "d_step": d_step + 1,
        "d_total_loss": float(total.detach().item()),
        "d_realism_loss": float(realism_loss.detach().item()),
        "d_structure_loss": float(structure_loss.detach().item()),
        "d_class_loss": float(class_loss.detach().item()),
        "d_r1_penalty": float(r1_penalty.detach().item()),
        "d_real_probability": float(torch.sigmoid(real_logits).mean().detach().item()),
        "d_fake_probability": float(torch.sigmoid(fake_logits).mean().detach().item()),
        "d_real_structure_probability": float(torch.sigmoid(real_structure[structure_mask]).mean().detach().item()) if structure_mask.any() else 0.0,
        "d_fake_structure_probability": float(torch.sigmoid(fake_structure[structure_mask]).mean().detach().item()) if structure_mask.any() else 0.0,
        "d_grad_norm_before_clip": grad_norm,
    }
    set_generator_view(discriminator)
    return stats


def resolve_generator_weights(args: argparse.Namespace) -> Dict[str, float]:
    payload = json.loads((ROOT / args.resolved_weights).read_text(encoding="utf-8"))
    weights = {key: float(value) for key, value in payload.get("resolved_weights", INITIAL_WEIGHTS).items()}
    overrides = {
        "adv": args.adv_weight,
        "cls": args.class_weight,
        "struct": args.structure_weight,
        "ord": args.ordinal_weight,
    }
    for key, value in overrides.items():
        if value >= 0:
            weights[key] = float(value)
    return weights


def save_checkpoint(
    path: Path,
    unet: torch.nn.Module,
    discriminator: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    batch: int,
    global_step: int,
    d_step: int,
) -> None:
    torch.save(
        {
            "lora_state": get_peft_model_state_dict(unet, adapter_name="a3"),
            "discriminator_state": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "global_step": global_step,
            "d_step": d_step,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A3 adversarial training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    final_lora_path = output_dir / "final_lora/pytorch_lora_weights.safetensors"
    if completed_path.is_file() and final_lora_path.is_file():
        print(f"[A3-adv] completed result already exists: {output_dir}", flush=True)
        return

    config = vars(args).copy()
    config.update(
        {
            "device": torch.cuda.get_device_name(0),
            "output_dir_resolved": str(output_dir),
            "test_data_used": False,
            "adversarial_definition": "delayed replay alternating D update followed by G update through current frozen-D view",
        }
    )
    write_json(output_dir / "run_config.json", config)
    frame = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    prior_frame = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(prior_frame.image_id.astype(str), prior_frame.prior_path.astype(str)))
    components = load_components(args, device)
    _, unet, vae, text_encoder, tokenizer, scheduler, discriminator, teachers, g_parameters = components
    dataset = A3Dataset(frame, prior_map, tokenizer, args.image_size, train=True)

    head_parameters, backbone_parameters = set_discriminator_view(discriminator)
    d_optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.d_head_lr},
            {"params": backbone_parameters, "lr": args.d_backbone_lr},
        ],
        weight_decay=args.d_weight_decay,
        betas=(0.0, 0.99),
    )
    set_generator_view(discriminator)
    g_optimizer = torch.optim.AdamW(g_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    micro_batches = math.ceil(args.samples_per_epoch / args.batch_size)
    steps_per_epoch = math.ceil(micro_batches / args.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * args.epochs
    if args.max_optimizer_steps:
        planned_steps = min(planned_steps, args.max_optimizer_steps)
    g_scheduler = get_cosine_schedule_with_warmup(
        g_optimizer,
        num_warmup_steps=min(args.warmup_steps, max(planned_steps // 4, 1)),
        num_training_steps=planned_steps,
    )
    scaler = GradScaler(enabled=True)
    weights = resolve_generator_weights(args)
    write_json(
        output_dir / "model_audit.json",
        {
            "sd_backbone_frozen": True,
            "vae_frozen": True,
            "text_encoder_frozen": True,
            "lora_trainable_parameters": sum(parameter.numel() for parameter in g_parameters),
            "discriminator_online_trainable": [
                "realism_label_embedding",
                "realism_head",
                "structure_head",
                "class_head",
                "convnext_features_6_7",
            ],
            "discriminator_frozen_during_generator_backward": True,
            "generator_receives_gradient_through_discriminator_input": True,
            "generator_weights": weights,
            "test_data_used": False,
        },
    )

    checkpoint_path = output_dir / "checkpoint_latest.pt"
    global_step = 0
    d_step = 0
    start_epoch = 0
    start_batch = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(unet, checkpoint["lora_state"], adapter_name="a3")
        discriminator.load_state_dict(checkpoint["discriminator_state"], strict=True)
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch"])
        global_step = int(checkpoint["global_step"])
        d_step = int(checkpoint["d_step"])
        print(f"[A3-adv] resume epoch={start_epoch + 1} batch={start_batch} g={global_step} d={d_step}", flush=True)

    replay: Deque[Dict[str, torch.Tensor]] = deque(maxlen=args.replay_capacity)
    rng = np.random.default_rng(args.seed + 7341)
    g_optimizer.zero_grad(set_to_none=True)
    micro_step = global_step * args.gradient_accumulation_steps
    accumulation: List[Dict[str, float]] = []
    last_d_stats = {
        "d_step": float(d_step),
        "d_total_loss": 0.0,
        "d_realism_loss": 0.0,
        "d_structure_loss": 0.0,
        "d_class_loss": 0.0,
        "d_r1_penalty": 0.0,
        "d_real_probability": 0.0,
        "d_fake_probability": 0.0,
        "d_real_structure_probability": 0.0,
        "d_fake_structure_probability": 0.0,
        "d_grad_norm_before_clip": 0.0,
    }
    started = time.time()
    stop = False

    for epoch in range(start_epoch, args.epochs):
        loader = DataLoader(
            Subset(dataset, balanced_indices(frame, args.samples_per_epoch, args.seed + epoch)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        unet.train()
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            at_accumulation_start = micro_step % args.gradient_accumulation_steps == 0
            if (
                at_accumulation_start
                and global_step >= args.d_warmup_g_steps
                and global_step % args.d_update_interval == 0
                and len(replay) >= args.d_batch_size
            ):
                last_d_stats = discriminator_step(
                    discriminator, d_optimizer, list(replay), rng, args, device, d_step
                )
                d_step += 1
                append_csv(output_dir / "discriminator_history.csv", last_d_stats)

            pixel, ids, labels, priors, image_ids, flipped = batch
            pixel = pixel.to(device, dtype=torch.float16, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            priors = priors.to(device, dtype=torch.float16, non_blocking=True)
            flipped = flipped.to(device)
            teacher_name, teacher = teachers[micro_step % len(teachers)]
            losses, stats, predicted_images = predicted_clean_losses(
                pixel,
                ids,
                labels,
                priors,
                flipped,
                teacher,
                unet,
                vae,
                text_encoder,
                scheduler,
                discriminator,
                args,
            )
            diffusion_timestep_mean = float(stats["timestep_mean"])
            separate_aux_probe = (
                args.separate_aux_interval > 0
                and micro_step % args.separate_aux_interval == 0
            )
            if separate_aux_probe:
                if args.aux_max_timestep < 0:
                    raise ValueError("--aux-max-timestep must be non-negative with a separate auxiliary probe")
                # Cycle deterministically through the complete trusted low-noise range.
                # This adds diagnostic exposure without biasing the uniformly sampled
                # diffusion MSE timestep distribution.
                forced_aux_timestep = int(
                    (args.seed + 73 * micro_step) % (args.aux_max_timestep + 1)
                )
                aux_losses, aux_stats, aux_predicted_images = predicted_clean_losses(
                    pixel,
                    ids,
                    labels,
                    priors,
                    flipped,
                    teacher,
                    unet,
                    vae,
                    text_encoder,
                    scheduler,
                    discriminator,
                    args,
                    forced_timestep=forced_aux_timestep,
                )
                losses = {"diff": losses["diff"], **{key: aux_losses[key] for key in ("adv", "cls", "struct", "ord")}}
                stats = aux_stats
                predicted_images = aux_predicted_images
                aux_active = True
                aux_probe_timestep_mean = float(forced_aux_timestep)
            else:
                aux_active = (
                    args.separate_aux_interval <= 0
                    and stats["timestep_mean"] <= args.aux_max_timestep
                )
                aux_probe_timestep_mean = float(stats["timestep_mean"]) if aux_active else -1.0
            stats = {
                **stats,
                "diffusion_timestep_mean": diffusion_timestep_mean,
                "aux_probe_timestep_mean": aux_probe_timestep_mean,
                "separate_aux_probe": float(separate_aux_probe),
            }
            total = losses["diff"]
            if aux_active:
                for key in ("adv", "cls", "struct", "ord"):
                    total = total + weights[key] * losses[key]
                # Keep the original source-group calibration partition unseen by
                # online D updates. The generator may still learn diffusion and
                # receive D feedback on these sources; only D replay is excluded.
                d_train_mask = torch.tensor(
                    [not grouped_calibration(str(image_id), args.seed) for image_id in image_ids],
                    device=pixel.device,
                    dtype=torch.bool,
                )
                if d_train_mask.any():
                    replay.append(
                        canonical_replay_item(
                            pixel[d_train_mask],
                            predicted_images[d_train_mask],
                            labels[d_train_mask],
                            priors[d_train_mask],
                            flipped[d_train_mask],
                        )
                    )
            scaler.scale(total / args.gradient_accumulation_steps).backward()
            record = {**stats, "total_loss": float(total.detach().item()), "aux_active": float(aux_active)}
            for key, value in losses.items():
                record[f"{key}_loss"] = float(value.detach().item())
            accumulation.append(record)
            micro_step += 1
            boundary = micro_step % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if boundary:
                scaler.unscale_(g_optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(g_parameters, args.max_grad_norm).item())
                scaler.step(g_optimizer)
                scaler.update()
                g_optimizer.zero_grad(set_to_none=True)
                g_scheduler.step()
                global_step += 1
                averaged = {key: float(np.mean([item[key] for item in accumulation])) for key in accumulation[0]}
                row = {
                    "global_step": global_step,
                    "d_step": d_step,
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                    "teacher": teacher_name,
                    "replay_size": len(replay),
                    "g_grad_norm_before_clip": grad_norm,
                    "g_learning_rate": float(g_optimizer.param_groups[0]["lr"]),
                    "d_head_learning_rate": float(d_optimizer.param_groups[0]["lr"]),
                    "gpu_peak_gib": float(torch.cuda.max_memory_allocated() / 2**30),
                    **averaged,
                    **{key: float(value) for key, value in last_d_stats.items()},
                }
                append_csv(output_dir / "step_history.csv", row)
                accumulation = []
                progress = {
                    "status": "training",
                    "mode": args.mode,
                    "global_step": global_step,
                    "planned_steps": planned_steps,
                    "d_step": d_step,
                    "replay_size": len(replay),
                    "loss": row["total_loss"],
                    "realism_probability": row["realism_probability"],
                    "target_probability": row["target_probability"],
                    "d_real_probability": row["d_real_probability"],
                    "d_fake_probability": row["d_fake_probability"],
                    "elapsed_minutes": (time.time() - started) / 60.0,
                }
                write_json(output_dir / "live_progress.json", progress)
                if global_step == 1 or global_step % 10 == 0:
                    print(
                        "[A3-adv] g={}/{} d={} total={:.5f} diff={:.5f} aux={:.2f} "
                        "Greal={:.3f} Dreal={:.3f} Dfake={:.3f} target={:.3f} vram={:.2f}GiB".format(
                            global_step,
                            planned_steps,
                            d_step,
                            row["total_loss"],
                            row["diff_loss"],
                            row["aux_active"],
                            row["realism_probability"],
                            row["d_real_probability"],
                            row["d_fake_probability"],
                            row["target_probability"],
                            row["gpu_peak_gib"],
                        ),
                        flush=True,
                    )
                if global_step % args.checkpoint_steps == 0:
                    save_checkpoint(
                        checkpoint_path,
                        unet,
                        discriminator,
                        g_optimizer,
                        d_optimizer,
                        g_scheduler,
                        scaler,
                        epoch,
                        batch_index + 1,
                        global_step,
                        d_step,
                    )
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
    torch.save(
        {
            "model_state": discriminator.state_dict(),
            "source_checkpoint": str((ROOT / args.discriminator_checkpoint).resolve()),
            "online_d_steps": d_step,
            "pretrained": False,
            "test_data_used": False,
        },
        output_dir / "final_online_discriminator.pt",
    )
    completed = {
        "status": "completed",
        "mode": args.mode,
        "global_step": global_step,
        "planned_steps": planned_steps,
        "d_step": d_step,
        "elapsed_minutes": (time.time() - started) / 60.0,
            "generator_weights": weights,
            "online_discriminator_source_group_calibration_excluded": True,
        "final_lora": str(final_lora.resolve()),
        "final_discriminator": str((output_dir / "final_online_discriminator.pt").resolve()),
        "test_data_used": False,
    }
    write_json(completed_path, completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
