# -*- coding: utf-8 -*-
"""
训练主入口
============
单卡：
    python train.py --config configs/default.yaml

多卡（DDP，负例跨卡同步）：
    torchrun --nproc_per_node=8 train.py --config configs/default.yaml

可选参数覆盖（不改 yaml 直接调试）：
    python train.py --config configs/default.yaml --batch-size 32 --max-steps 500
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from blm_clip.data import ClipCollator, ImageTextDataset
from blm_clip.engine import Trainer
from blm_clip.models import BLMClipModel


def parse_args():
    p = argparse.ArgumentParser(description="BLM ViT 单阶段图文对比学习训练")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    # 常用覆盖项
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行覆盖配置
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg["seed"] + rank)
    if rank == 0:
        print(f"[config] {cfg}")

    # ---- 分词器（决定文本塔词表大小与 pad_id）----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 默认无 pad，用 eos 代替

    # ---- 数据 ----
    collator = ClipCollator(
        tokenizer=tokenizer,
        max_text_len=cfg["model"]["text_max_len"],
        patch_size=cfg["data"]["patch_size"],
        merge_size=cfg["data"]["merge_size"],
        min_pixels=cfg["data"]["min_pixels"],
        max_pixels=cfg["data"]["max_pixels"],
        vision_type=cfg["model"]["vision_type"],
    )
    train_set = ImageTextDataset(cfg["data"]["train_jsonl"], cfg["data"]["image_root"])

    if cfg["data"].get("use_binpack", False):
        # ===== binpack 模式：DataLoader 遍历 bin 列表，一个 bin = 一个 step =====
        from blm_clip.binpack import build_bins, build_token_nums
        from blm_clip.data import BinCollator, BinDataset

        # 1) 离线统计全部样本的 patch token 数（带磁盘缓存，二次启动秒读）
        token_nums = build_token_nums(
            train_set.samples, cfg["data"]["image_root"], cfg["data"],
            cache_path=cfg["data"]["train_jsonl"] + ".tokencache.json",
        )
        # 2) 装箱：online（业务数据，尺寸有上限）/ offline（开源长尾，全局装箱）
        bins = build_bins(
            train_set.samples, token_nums,
            strategy=cfg["data"].get("bin_strategy", "online"),
            bin_token=cfg["data"]["bin_token"],
            seed=cfg["seed"],
        )
        # 3) DDP：各 rank 交错取 bin（每桶 token ≈ bin_token，各卡 step 时间一致）
        if is_dist:
            bins = bins[rank::dist.get_world_size()]
        if rank == 0:
            n_samples = sum(len(b) for b in bins)
            print(f"[binpack] {len(bins)} bins, {n_samples} samples, "
                  f"avg {n_samples / max(1, len(bins)):.1f} samples/bin")

        # 4) DataLoader 以 bin 为元素（batch_size=1），BinCollator 对整个 bin 做 packing
        #    注意：此模式下 train.batch_size 失效，有效 batch = bin 的实际填充量
        train_loader = DataLoader(
            BinDataset(bins),
            batch_size=1,
            shuffle=True,                     # 每个 epoch 打乱 bin 的顺序
            num_workers=cfg["data"]["num_workers"],
            collate_fn=BinCollator(collator, cfg["data"]["image_root"]),
            pin_memory=True,
        )
    else:
        # ===== 常规模式：固定 batch_size，DDP 用 DistributedSampler =====
        train_loader = DataLoader(
            train_set,
            batch_size=cfg["train"]["batch_size"],
            shuffle=not is_dist,
            sampler=DistributedSampler(train_set) if is_dist else None,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=collator,
            drop_last=True,
            pin_memory=True,
        )

    eval_loader = None
    if cfg["data"].get("eval_jsonl") and os.path.exists(cfg["data"]["eval_jsonl"]):
        eval_set = ImageTextDataset(cfg["data"]["eval_jsonl"], cfg["data"]["image_root"])
        eval_loader = DataLoader(
            eval_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
            num_workers=cfg["data"]["num_workers"], collate_fn=collator, pin_memory=True,
        )

    # ---- 模型 ----
    model = BLMClipModel(cfg, vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id)
    model.to(device)
    if is_dist:
        # 无 BN 层，broadcast_buffers 关闭；find_unused 关闭以省通信
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )
        raw_model = model.module
    else:
        raw_model = model

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"])
        if rank == 0:
            print(f"[resume] 从 {args.resume} 恢复（step {ckpt.get('step')}）")

    # ---- 训练 ----
    trainer = Trainer(raw_model, cfg, train_loader, eval_loader, device, is_dist=is_dist, rank=rank)
    if args.resume and "optimizer" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
    trainer.train()

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
