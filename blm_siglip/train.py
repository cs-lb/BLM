# -*- coding: utf-8 -*-
"""
训练主入口（SigLIP 版，8×H800 DDP）
======================================
单卡冒烟：
    python train.py --config configs/default.yaml --max-steps 200

8 卡正式训练（负例跨卡同步）：
    torchrun --nproc_per_node=8 train.py --config configs/default.yaml

断点续训：
    python train.py --config configs/default.yaml --resume outputs/blm_vit_v1/ckpt_step2000.pt
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler

# 复用 blm_clip 的数据层与 binpack（同级工程，见 README「工程关系」）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blm_clip"))

from blm_clip.data import ClipCollator, ImageTextDataset  # noqa: E402
from blm_siglip.engine import Trainer  # noqa: E402
from blm_siglip.models import BLMSiglipModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="BLM 视觉基座预训练（SigLIP）")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def pick_device() -> torch.device:
    """设备选择：CUDA > MPS（Apple Silicon）> CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir

    # ---- 分布式环境（torchrun 注入 RANK/WORLD_SIZE）----
    is_dist = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        rank = 0
        device = pick_device()

    seed = cfg["seed"] + rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if rank == 0:
        print(f"[config] {cfg}")

    # ---- BGE-M3 分词器（SentencePiece，与 Qwen 不同体系，别混用）----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_name"])

    collator = ClipCollator(
        tokenizer=tokenizer,
        max_text_len=96,
        patch_size=cfg["data"]["patch_size"],
        merge_size=cfg["data"]["merge_size"],
        min_pixels=cfg["data"]["min_pixels"],
        max_pixels=cfg["data"]["max_pixels"],
        vision_type=cfg["model"]["vision_type"],
    )

    train_set = ImageTextDataset(cfg["data"]["train_jsonl"], cfg["data"]["image_root"])

    if cfg["data"].get("use_binpack", False):
        # ===== binpack 模式：一个 bin = 一个 step；各 rank 交错取 bin =====
        from blm_clip.binpack import build_bins, build_token_nums
        from blm_clip.data import BinCollator, BinDataset

        token_nums = build_token_nums(
            train_set.samples, cfg["data"]["image_root"], cfg["data"],
            cache_path=cfg["data"]["train_jsonl"] + ".tokencache.json",
        )
        bins = build_bins(train_set.samples, token_nums,
                          strategy=cfg["data"].get("bin_strategy", "online"),
                          bin_token=cfg["data"]["bin_token"], seed=cfg["seed"])
        if is_dist:
            bins = bins[rank::dist.get_world_size()]
        if rank == 0:
            n = sum(len(b) for b in bins)
            print(f"[binpack] {len(bins)} bins, {n} samples, "
                  f"avg {n / max(1, len(bins)):.1f} samples/bin")
        train_loader = DataLoader(
            BinDataset(bins), batch_size=1, shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=BinCollator(collator, cfg["data"]["image_root"]),
            pin_memory=True,
        )
    else:
        sampler = DistributedSampler(train_set, shuffle=True) if is_dist else None
        train_loader = DataLoader(
            train_set, batch_size=cfg["train"]["batch_size"],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=cfg["data"]["num_workers"], collate_fn=collator,
            drop_last=True, pin_memory=True,
        )

    eval_loader = None
    if cfg["data"].get("eval_jsonl") and os.path.exists(cfg["data"]["eval_jsonl"]):
        eval_set = ImageTextDataset(cfg["data"]["eval_jsonl"], cfg["data"]["image_root"])
        eval_loader = DataLoader(eval_set, batch_size=cfg["train"]["batch_size"],
                                 shuffle=False, num_workers=cfg["data"]["num_workers"],
                                 collate_fn=collator, pin_memory=True)

    # ---- 模型 ----
    model = BLMSiglipModel(cfg)

    # MPS/CPU 环境：bf16 权重与算子支持不全，强制 fp32（M1 Pro 本地测试场景）
    if device.type in ("mps", "cpu") and cfg["model"]["vision_type"] == "qwenvit":
        model.visual.to(torch.float32)
        if rank == 0:
            print("[env] 非 CUDA 设备，视觉塔已转为 fp32")

    # 梯度检查点：24GB 级显存（如双卡 4090）省激活内存，代价约 +30% 计算
    # DDP 下必须 use_reentrant=False，且 embedding 层需显式开输入梯度
    if cfg["train"].get("grad_checkpointing", False):
        ckpt_kwargs = {"use_reentrant": False}
        model.visual.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=ckpt_kwargs)
        model.text.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=ckpt_kwargs)
        model.text.backbone.enable_input_require_grads()
        if hasattr(model.visual.visual, "enable_input_require_grads"):
            model.visual.visual.enable_input_require_grads()
        if rank == 0:
            print("[env] 已开启梯度检查点（视觉塔 + 文本塔，use_reentrant=False）")

    model.to(device)
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        if rank == 0:
            print(f"[resume] 从 {args.resume} 恢复（step {start_step}，LR 调度同步快进）")

    trainer = Trainer(raw_model, cfg, train_loader, eval_loader, device,
                      is_dist=is_dist, rank=rank, start_step=start_step)
    if args.resume and "optimizer" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
    trainer.train()

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
