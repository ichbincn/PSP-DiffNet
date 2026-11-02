from __future__ import annotations

import argparse
import json
import math
import os
import platform
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

from data.aptos_dataset import APTOS_CLASS_ORDER
from utils.fundus import preprocess_fundus_pil


PROMPTS = {
    "No_DR": "high quality color fundus photograph, healthy retina, no diabetic retinopathy, clinically plausible retinal anatomy",
    "Mild": "high quality color fundus photograph, mild diabetic retinopathy, clinically plausible retinal anatomy",
    "Moderate": "high quality color fundus photograph, moderate diabetic retinopathy, clinically plausible retinal anatomy",
    "Severe": "high quality color fundus photograph, severe diabetic retinopathy, clinically plausible retinal anatomy",
    "Proliferate_DR": "high quality color fundus photograph, proliferative diabetic retinopathy, clinically plausible retinal anatomy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a class-conditioned SD1.5 UNet LoRA on the fixed APTOS train split")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--train-split", default="outputs/aptos2019/sd025_balanced_train/splits/train.csv")
    parser.add_argument("--output-dir", default="outputs/aptos2019/lora_sd025_experiment/lora_training")
    parser.add_argument("--report", default="APTOS_LoRA医学域微调训练报告_20260807.md")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=2560)
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
    torch.backends.cuda.matmul.allow_tf32 = False


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


class AptosLoRADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: CLIPTokenizer, image_size: int):
        self.dataframe = dataframe.reset_index(drop=True)
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
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = preprocess_fundus_pil(image, crop=True, mask_background=True).image
        return self.transform(image), self.token_ids[str(row["label_name"])], str(row["label_name"])


def balanced_epoch_indices(dataframe: pd.DataFrame, samples_per_epoch: int, seed: int) -> List[int]:
    if samples_per_epoch % len(APTOS_CLASS_ORDER) != 0:
        raise ValueError("samples_per_epoch must be divisible by the number of classes")
    rng = np.random.default_rng(seed)
    per_class = samples_per_epoch // len(APTOS_CLASS_ORDER)
    indices: List[int] = []
    for class_name in APTOS_CLASS_ORDER:
        candidates = dataframe.index[dataframe["label_name"] == class_name].to_numpy()
        if len(candidates) == 0:
            raise RuntimeError(f"No training images for {class_name}")
        indices.extend(rng.choice(candidates, size=per_class, replace=len(candidates) < per_class).tolist())
    rng.shuffle(indices)
    return indices


def save_checkpoint(
    path: Path,
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch_index: int,
    next_batch_index: int,
    global_step: int,
    history: List[Dict[str, object]],
) -> None:
    torch.save(
        {
            "lora_state": get_peft_model_state_dict(unet),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch_index": epoch_index,
            "next_batch_index": next_batch_index,
            "global_step": global_step,
            "history": history,
        },
        path,
    )


def write_report(
    args: argparse.Namespace,
    output_dir: Path,
    dataframe: pd.DataFrame,
    trainable: int,
    total: int,
    history: List[Dict[str, object]],
    elapsed_minutes: float,
) -> None:
    counts = dataframe["label_name"].value_counts().reindex(APTOS_CLASS_ORDER).astype(int)
    final_loss = float(history[-1]["mean_loss"])
    best_loss = min(float(row["mean_loss"]) for row in history)
    lines = [
        "# APTOS Stable Diffusion v1.5 LoRA 医学域微调训练报告",
        "",
        "## 1. 任务定位",
        "",
        "本轮训练论文规划中的基础 diffusion+LoRA 版本：以 Stable Diffusion v1.5 为 backbone，冻结 VAE、文本编码器及 U-Net 原始参数，只训练 U-Net 注意力层中的 LoRA 参数，使通用自然图像扩散模型适配眼底医学图像域。该版本不加入结构先验、质量筛选或对抗损失。",
        "",
        "## 2. 数据与防泄漏",
        "",
        f"- 训练数据只使用固定 APTOS train split，共 {len(dataframe)} 张真实图像；未使用验证集、测试集或普通 SD 合成图。",
        "- 固定 train/val/test 切分与普通 SD 实验完全一致，seed=2026。",
        "- 每个 epoch 固定抽取 2560 个训练样本，每类精确抽取 512 次；样本不足类别采用有放回抽样，使 LoRA 同时学习医学域和类别条件，而不被 No_DR 主导。",
        "",
        "| 类别 | 真实训练图 | 每 epoch 暴露次数 |",
        "|---|---:|---:|",
        *[f"| {name} | {int(counts[name])} | {args.samples_per_epoch // len(APTOS_CLASS_ORDER)} |" for name in APTOS_CLASS_ORDER],
        "",
        "## 3. LoRA 结构与原理",
        "",
        f"- 注入位置：U-Net 所有 cross/self-attention 的 `to_q`、`to_k`、`to_v`、`to_out.0` 线性层。",
        f"- Rank={args.rank}，alpha={args.lora_alpha}；可训练参数 {trainable:,}/{total:,}（{100.0 * trainable / total:.4f}%）。",
        "- 对冻结权重 W 不直接更新，只学习低秩增量 ΔW=BA，并以 W+α/r·BA 参与前向传播，从而以较少参数完成医学域适配。",
        "- 训练目标为标准 latent diffusion 噪声预测 MSE：真实眼底图经冻结 VAE 编码到 latent，随机加入扩散噪声，再由带 LoRA 的 U-Net 根据类别文本条件预测噪声。",
        "",
        "## 4. 完整训练策略",
        "",
        f"- 输入：眼底 ROI 裁黑边与背景遮罩，resize={args.image_size}，仅使用 p=0.5 水平翻转；不做旋转和颜色扰动。",
        f"- Epoch={args.epochs}；batch size={args.batch_size}；梯度累积={args.gradient_accumulation_steps}；有效 batch={args.batch_size * args.gradient_accumulation_steps}。",
        f"- AdamW：lr={args.learning_rate}，weight decay={args.weight_decay}；warm-up={args.warmup_steps} optimizer steps，随后 cosine decay。",
        f"- FP16 混合精度，梯度裁剪 max norm={args.max_grad_norm}，gradient checkpointing；随机种子={args.seed}。",
        "- U-Net backbone、VAE、CLIP text encoder 全部冻结；没有训练分类器，也没有引入对抗判别器。",
        "",
        "## 5. 训练结果",
        "",
        f"- 完成 optimizer steps：{history[-1]['global_step']}。",
        f"- 最后一轮平均 denoising loss：{final_loss:.6f}；最低轮平均 loss：{best_loss:.6f}。",
        f"- 总耗时：{elapsed_minutes:.2f} 分钟；GPU：{torch.cuda.get_device_name(0)}。",
        "",
        "| Epoch | 平均 denoising loss | optimizer step |",
        "|---:|---:|---:|",
        *[f"| {row['epoch']} | {float(row['mean_loss']):.6f} | {row['global_step']} |" for row in history],
        "",
        "## 6. 输出与复现",
        "",
        f"- LoRA 权重：`{output_dir / 'final_lora'}`",
        f"- 训练历史：`{output_dir / 'training_history.json'}`",
        f"- 实时进度：`{output_dir / 'live_progress.json'}`",
        f"- 断点：`{output_dir / 'checkpoint_latest.pt'}`",
        "",
        "```bash",
        "bash scripts/run_aptos_lora_training.sh",
        "```",
        "",
        "另开终端查看进度：",
        "",
        "```bash",
        "bash scripts/watch_aptos_lora_pipeline.sh",
        "```",
        "",
        "## 7. 限制",
        "",
        "- Denoising loss 只能证明 LoRA 学会训练目标，不能单独证明生成图医学真实性或下游有效性。后续必须使用与普通 SD 完全一致的 FID/KID/IPR 和随机初始化 ConvNeXt 协议验证。",
        "- 这是第一版单 seed LoRA baseline，不包含结构先验和诊断一致性判别器。",
        "",
        "## 8. 环境",
        "",
        f"- Python {platform.python_version()}；PyTorch {torch.__version__}；CUDA {torch.version.cuda}。",
    ]
    (ROOT / args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA")
    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8")
    seed_everything(args.seed)
    device = torch.device("cuda")
    model_dir = (ROOT / args.model_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_lora_dir = output_dir / "final_lora"
    completed_path = output_dir / "completed.json"
    if completed_path.exists() and (final_lora_dir / "pytorch_lora_weights.safetensors").exists():
        print(f"LoRA training already completed: {final_lora_dir}", flush=True)
        return

    dataframe = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    if len(dataframe) != 2563 or set(dataframe["label_name"]) != set(APTOS_CLASS_ORDER):
        raise RuntimeError("Unexpected fixed APTOS train split")
    config = vars(args).copy()
    config.update(
        {
            "model_dir_resolved": str(model_dir),
            "train_split_resolved": str((ROOT / args.train_split).resolve()),
            "protocol": "fixed real train split only; exact class-balanced exposure per epoch; frozen SD1.5 backbone",
        }
    )
    config_path = output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing LoRA output directory has a different configuration")
    write_json(config_path, config)

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
    trainable = sum(parameter.numel() for parameter in trainable_parameters)
    total = sum(parameter.numel() for parameter in unet.parameters())
    print(f"LoRA trainable parameters: {trainable:,}/{total:,} ({100.0 * trainable / total:.4f}%)", flush=True)

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    micro_batches_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_optimizer_steps
    )
    scaler = GradScaler(enabled=True)
    dataset = AptosLoRADataset(dataframe, tokenizer, args.image_size)
    history: List[Dict[str, object]] = []
    start_epoch = 0
    start_batch = 0
    global_step = 0
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(unet, checkpoint["lora_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch_index"])
        start_batch = int(checkpoint["next_batch_index"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
        print(f"Resuming epoch={start_epoch + 1}, batch={start_batch}, global_step={global_step}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    for epoch_index in range(start_epoch, args.epochs):
        indices = balanced_epoch_indices(dataframe, args.samples_per_epoch, args.seed + epoch_index)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_seed,
            generator=torch.Generator().manual_seed(args.seed + epoch_index),
        )
        epoch_losses: List[float] = []
        unet.train()
        for batch_index, (pixel_values, input_ids, _) in enumerate(loader):
            if epoch_index == start_epoch and batch_index < start_batch:
                continue
            pixel_values = pixel_values.to(device, dtype=torch.float16, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            with torch.no_grad(), autocast(enabled=True):
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with autocast(enabled=True):
                prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            epoch_losses.append(float(loss.item()))
            accumulation_boundary = (batch_index + 1) % args.gradient_accumulation_steps == 0
            final_batch = batch_index + 1 == len(loader)
            if accumulation_boundary or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                progress = {
                    "status": "training",
                    "epoch": epoch_index + 1,
                    "epochs": args.epochs,
                    "batch": batch_index + 1,
                    "batches_per_epoch": len(loader),
                    "global_step": global_step,
                    "total_optimizer_steps": total_optimizer_steps,
                    "loss": float(loss.item()),
                    "mean_epoch_loss": float(np.mean(epoch_losses)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_minutes": (time.time() - started) / 60.0,
                }
                write_json(output_dir / "live_progress.json", progress)
                if global_step % 25 == 0 or global_step == 1:
                    print(
                        f"epoch {epoch_index + 1}/{args.epochs} batch {batch_index + 1}/{len(loader)} "
                        f"step {global_step}/{total_optimizer_steps} loss={loss.item():.5f} "
                        f"mean={np.mean(epoch_losses):.5f}",
                        flush=True,
                    )
                if global_step % args.checkpoint_steps == 0:
                    save_checkpoint(
                        checkpoint_path,
                        unet,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch_index,
                        batch_index + 1,
                        global_step,
                        history,
                    )
        history.append(
            {"epoch": epoch_index + 1, "mean_loss": float(np.mean(epoch_losses)), "global_step": global_step}
        )
        write_json(output_dir / "training_history.json", history)
        save_checkpoint(
            checkpoint_path,
            unet,
            optimizer,
            scheduler,
            scaler,
            epoch_index + 1,
            0,
            global_step,
            history,
        )
        start_batch = 0
        print(f"epoch {epoch_index + 1} completed: mean_loss={history[-1]['mean_loss']:.6f}", flush=True)

    final_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(final_lora_dir, unet_lora_layers=final_lora_state)
    elapsed_minutes = (time.time() - started) / 60.0
    write_json(
        completed_path,
        {
            "status": "completed",
            "global_step": global_step,
            "elapsed_minutes": elapsed_minutes,
            "trainable_parameters": trainable,
            "total_unet_parameters_with_lora": total,
            "final_lora_dir": str(final_lora_dir),
        },
    )
    write_json(
        output_dir / "live_progress.json",
        {
            "status": "completed",
            "epoch": args.epochs,
            "epochs": args.epochs,
            "global_step": global_step,
            "total_optimizer_steps": total_optimizer_steps,
            "elapsed_minutes": elapsed_minutes,
        },
    )
    write_report(args, output_dir, dataframe, trainable, total, history, elapsed_minutes)
    print(f"LoRA training completed: {final_lora_dir}", flush=True)
    print(f"Report: {(ROOT / args.report).resolve()}", flush=True)


if __name__ == "__main__":
    main()
