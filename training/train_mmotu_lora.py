from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER
from utils.ultrasound import preprocess_ultrasound_pil


PROMPTS: Dict[str, str] = {
    "Chocolate cyst": "high quality grayscale transvaginal ultrasound image of an ovarian chocolate cyst, clinically plausible anatomy",
    "Serous cystadenoma": "high quality grayscale transvaginal ultrasound image of ovarian serous cystadenoma, clinically plausible anatomy",
    "Teratoma": "high quality grayscale transvaginal ultrasound image of an ovarian teratoma, clinically plausible anatomy",
    "Theca cell tumor": "high quality grayscale transvaginal ultrasound image of an ovarian theca cell tumor, clinically plausible anatomy",
    "Simple cyst": "high quality grayscale transvaginal ultrasound image of a simple ovarian cyst, clinically plausible anatomy",
    "Normal ovary": "high quality grayscale transvaginal ultrasound image of a normal ovary, clinically plausible anatomy",
    "Mucinous cystadenoma": "high quality grayscale transvaginal ultrasound image of ovarian mucinous cystadenoma, clinically plausible anatomy",
    "High-grade serous carcinoma": "high quality grayscale transvaginal ultrasound image of high grade serous ovarian carcinoma, clinically plausible anatomy",
}
PREPROCESSING_DESCRIPTION = "aspect-preserving resize-pad; horizontal flip only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train class-conditioned SD1.5 LoRA on fixed MMOTU real train")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--output-dir", default="outputs/mmotu2d/a0_lora/lora_training")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


class MMOTULoRADataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: CLIPTokenizer, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.token_ids = {
            class_name: tokenizer(
                PROMPTS[class_name],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            for class_name in MMOTU_CLASS_ORDER
        }
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as handle:
            image = preprocess_ultrasound_pil(handle.convert("RGB"), self.image_size)
        return self.transform(image), self.token_ids[str(row.label_name)], str(row.label_name)


def balanced_indices(frame: pd.DataFrame, samples: int, seed: int) -> List[int]:
    if samples % len(MMOTU_CLASS_ORDER):
        raise ValueError("samples-per-epoch must be divisible by eight")
    rng = np.random.default_rng(seed)
    per_class = samples // len(MMOTU_CLASS_ORDER)
    indices: List[int] = []
    for class_name in MMOTU_CLASS_ORDER:
        candidates = frame.index[frame.label_name == class_name].to_numpy()
        indices.extend(rng.choice(candidates, per_class, replace=len(candidates) < per_class).tolist())
    rng.shuffle(indices)
    return indices


def save_checkpoint(path, unet, optimizer, scheduler, scaler, epoch, batch, step, history) -> None:
    torch.save(
        {
            "lora_state": get_peft_model_state_dict(unet),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "history": history,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU LoRA training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    started = time.time()
    model_dir = (ROOT / args.model_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_lora_dir = output_dir / "final_lora"
    completed_path = output_dir / "completed.json"
    if completed_path.is_file() and (final_lora_dir / "pytorch_lora_weights.safetensors").is_file():
        print(f"[MMOTU LoRA] completed: {final_lora_dir}", flush=True)
        return

    frame = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    if frame.empty or set(frame.label_name) != set(MMOTU_CLASS_ORDER):
        raise RuntimeError("Unexpected fixed medical train split or class set")
    config = vars(args).copy()
    config.update(
        {
            "train_only": True,
            "test_accessed": False,
            "class_names": MMOTU_CLASS_ORDER,
            "prompts": PROMPTS,
            "class_balanced_exposure": args.samples_per_epoch // len(MMOTU_CLASS_ORDER),
            "preprocessing": PREPROCESSING_DESCRIPTION,
        }
    )
    write_json(output_dir / "run_config.json", config)

    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_dir, subfolder="vae", variant="fp16", torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_dir, subfolder="unet", variant="fp16", torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    unet.enable_gradient_checkpointing()
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    for parameter in trainable_parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    micro_batches = math.ceil(args.samples_per_epoch / args.batch_size)
    steps_per_epoch = math.ceil(micro_batches / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=True)
    dataset = MMOTULoRADataset(frame, tokenizer, args.image_size)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    history = []
    start_epoch = 0
    start_batch = 0
    global_step = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(unet, checkpoint["lora_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch"])
        global_step = int(checkpoint["step"])
        history = list(checkpoint["history"])
        print(f"[MMOTU LoRA] resume epoch={start_epoch + 1} batch={start_batch} step={global_step}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        loader = DataLoader(
            Subset(dataset, balanced_indices(frame, args.samples_per_epoch, args.seed + epoch)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_seed,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        losses = []
        unet.train()
        for batch_index, (pixels, input_ids, _class_names) in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            pixels = pixels.to(device, dtype=torch.float16, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            with torch.no_grad(), autocast(enabled=True):
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                hidden = text_encoder(input_ids)[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(latents),), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            with autocast(enabled=True):
                prediction = unet(noisy, timesteps, hidden).sample
                target = noise_scheduler.get_velocity(latents, noise, timesteps) if noise_scheduler.config.prediction_type == "v_prediction" else noise
                loss = F.mse_loss(prediction.float(), target.float())
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.item()))
            boundary = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                progress = {
                    "status": "training",
                    "epoch": epoch + 1,
                    "epochs": args.epochs,
                    "batch": batch_index + 1,
                    "batches_per_epoch": len(loader),
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "loss": float(loss.item()),
                    "mean_epoch_loss": float(np.mean(losses)),
                    "elapsed_minutes": (time.time() - started) / 60,
                }
                write_json(output_dir / "live_progress.json", progress)
                if global_step == 1 or global_step % 25 == 0:
                    print(
                        f"[MMOTU LoRA] epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                        f"loss={loss.item():.5f} mean={np.mean(losses):.5f}",
                        flush=True,
                    )
                if global_step % args.checkpoint_steps == 0:
                    save_checkpoint(checkpoint_path, unet, optimizer, scheduler, scaler, epoch, batch_index + 1, global_step, history)
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "global_step": global_step})
        write_json(output_dir / "training_history.json", history)
        save_checkpoint(checkpoint_path, unet, optimizer, scheduler, scaler, epoch + 1, 0, global_step, history)
        start_batch = 0

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(final_lora_dir, unet_lora_layers=state)
    completed = {
        "status": "completed",
        "global_step": global_step,
        "elapsed_minutes": (time.time() - started) / 60,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "final_lora": str(final_lora_dir.resolve()),
        "test_accessed": False,
        "history": history,
    }
    write_json(completed_path, completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
