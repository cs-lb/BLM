# -*- coding: utf-8 -*-
"""
独立评估入口
==============
加载训练好的 checkpoint，在评估集上输出检索指标：

    python eval.py --config configs/default.yaml --ckpt outputs/blm_vit_v1/ckpt_step20000.pt
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from blm_clip.data import ClipCollator, ImageTextDataset
from blm_clip.engine import Evaluator
from blm_clip.models import BLMClipModel


def main():
    p = argparse.ArgumentParser(description="BLM ViT 检索评估")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = ClipCollator(
        tokenizer=tokenizer,
        max_text_len=cfg["model"]["text_max_len"],
        patch_size=cfg["data"]["patch_size"],
        merge_size=cfg["data"]["merge_size"],
        min_pixels=cfg["data"]["min_pixels"],
        max_pixels=cfg["data"]["max_pixels"],
        vision_type=cfg["model"]["vision_type"],
    )
    eval_set = ImageTextDataset(cfg["data"]["eval_jsonl"], cfg["data"]["image_root"])
    eval_loader = DataLoader(
        eval_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collator, pin_memory=True,
    )

    model = BLMClipModel(cfg, vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"[load] {args.ckpt}（step {ckpt.get('step')}）")

    metrics = Evaluator(model, eval_loader, device).run()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
