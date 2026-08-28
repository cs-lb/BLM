# -*- coding: utf-8 -*-
"""
三元损失 + 蒸馏训练入口（v1 → v2）
======================================
从 v1 checkpoint 继续训练：
    python train.py --config configs/default.yaml --v1-ckpt <blm_siglip的ckpt>

依赖：复用 blm_siglip 的 BLMSiglipModel 与 blm_clip 的数据层。
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blm_clip"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blm_siglip"))

from blm_clip.data import ClipCollator  # noqa: E402
from blm_siglip.models import BLMSiglipModel  # noqa: E402
from blm_triplet.data import TripletCollator, TripletDataset  # noqa: E402
from blm_triplet.engine import TripletTrainer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="BLM ViT v1→v2：三元损失 + Reranker 蒸馏")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--v1-ckpt", type=str, default=None, help="v1（blm_siglip）checkpoint")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_name"])

    # ---- 数据 ----
    clip_collator = ClipCollator(
        tokenizer=tokenizer, max_text_len=96,
        patch_size=cfg["data"]["patch_size"], merge_size=cfg["data"]["merge_size"],
        min_pixels=cfg["data"]["min_pixels"], max_pixels=cfg["data"]["max_pixels"],
        vision_type=cfg["model"]["vision_type"],
    )
    train_set = TripletDataset(cfg["data"]["triplet_jsonl"])
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=TripletCollator(clip_collator, tokenizer, cfg["data"]["image_root"]),
        pin_memory=True, drop_last=True,
    )

    # ---- 模型：从 v1 继续（v2 = v1 + 三元/蒸馏微调）----
    model = BLMSiglipModel(cfg)
    if args.v1_ckpt:
        ckpt = torch.load(args.v1_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[init] 从 v1 ckpt 继续训练（step {ckpt.get('step')}）")
    model.to(device)

    trainer = TripletTrainer(model, cfg, train_loader, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
