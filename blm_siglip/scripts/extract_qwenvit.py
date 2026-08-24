# -*- coding: utf-8 -*-
"""
视觉塔权重抽取：从 Qwen2.5-VL-7B 全模中抽出 model.visual 单独存盘
====================================================================
为什么需要：7B 全模 bf16 约 15GB，每次训练启动都加载整模再取视觉塔，
浪费内存与时间。一次性抽取后（约 1.4GB），训练时 _from_config +
load_state_dict 秒级加载（见 models.py::QwenViTVisionTower._load_extracted）。

用法：
    python scripts/extract_qwenvit.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --out assets/qwenvit_7b_visual.pt
"""

import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--out", default="assets/qwenvit_7b_visual.pt")
    args = p.parse_args()

    try:
        from transformers import Qwen2_5_VLModel as _Base
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Base

    print(f"[load] {args.model}（仅这一次需要加载全模）...")
    base = _Base.from_pretrained(args.model, low_cpu_mem_usage=True)
    visual = base.visual if hasattr(base, "visual") else base.model.visual

    payload = {
        "config": visual.config.to_dict(),       # vision_config，重建结构用
        "state_dict": visual.state_dict(),       # 视觉塔全部权重
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    size_gb = os.path.getsize(args.out) / 1024 ** 3
    print(f"[done] {args.out}（{size_gb:.2f} GB）")
    print("之后把 configs/default.yaml 的 model.visual_weights 指向该文件即可秒载")


if __name__ == "__main__":
    main()
