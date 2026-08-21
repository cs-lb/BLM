# -*- coding: utf-8 -*-
"""
细粒度图文对齐训练入口（方案第 6 章）
========================================
在 blm_clip 的 BLMClipModel 基础上，加入区域级监督：
    L = L_global（CLIP InfoNCE）+ λ1 * L_region（区域 InfoNCE）+ λ2 * L_hard（难负例 margin）

数据：fg_data 生产线产出的 fg_train.jsonl
    {"image", "caption", "regions": [{"expr", "bbox", "conf", "hard": [...]}]}

用法：
    python train_fg.py --config configs/fg_config.yaml
"""

import argparse
import math
import os
import random
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 复用 blm_clip 工程（同级目录）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blm_clip"))

from blm_clip.data import ClipCollator, ImageTextDataset, move_batch_to_device  # noqa: E402
from blm_clip.engine import Evaluator, cosine_warmup_lambda  # noqa: E402
from blm_clip.losses import ClipLoss  # noqa: E402
from blm_clip.models import BLMClipModel  # noqa: E402

from fg_clip.region import (  # noqa: E402
    encode_regions, hard_negative_margin_loss, region_infonce_loss,
)
from fg_data.common import read_jsonl  # noqa: E402


# ---------------------------------------------------------------
# 数据集：图文对 + 区域监督
# ---------------------------------------------------------------
class FGDataset(Dataset):
    def __init__(self, jsonl_path: str, max_regions: int):
        self.samples = read_jsonl(jsonl_path)
        self.max_regions = max_regions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        regions = s["regions"][: self.max_regions]   # 每图区域数上限（控制文本塔开销）
        return {"image": s["image"], "text": s["caption"], "regions": regions}


class FGCollator:
    """全局图文对走 ClipCollator；区域/难负例文本单独 tokenize。"""

    def __init__(self, clip_collator: ClipCollator, tokenizer, image_root: str, max_text_len: int):
        self.clip_collator = clip_collator
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.max_text_len = max_text_len

    def _tokenize_list(self, texts: list[str]) -> dict:
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_text_len, return_tensors="pt")

    def __call__(self, batch: list[dict]) -> dict:
        # ---- 全局图文（复用 blm_clip 的 packing collator）----
        global_batch = self.clip_collator(
            [{"image": self._load(b), "text": b["text"]} for b in batch])

        # ---- 区域监督：拍平所有图的区域与难负例 ----
        regions_per_image, pos_texts, hard_texts, hard_region_idx = [], [], [], []
        for b in batch:
            bboxes = []
            for r in b["regions"]:
                region_id = len(pos_texts)
                bboxes.append(r["bbox"])
                pos_texts.append(r["expr"])
                for h in r["hard"]:
                    hard_texts.append(h)
                    hard_region_idx.append(region_id)
            regions_per_image.append(bboxes)

        pos_enc = self._tokenize_list(pos_texts) if pos_texts else None
        hard_enc = self._tokenize_list(hard_texts) if hard_texts else None

        return {
            **global_batch,
            "regions_per_image": regions_per_image,
            "pos_ids": pos_enc["input_ids"] if pos_enc else None,
            "pos_mask": pos_enc["attention_mask"] if pos_enc else None,
            "hard_ids": hard_enc["input_ids"] if hard_enc else None,
            "hard_mask": hard_enc["attention_mask"] if hard_enc else None,
            "hard_region_idx": torch.tensor(hard_region_idx, dtype=torch.long),
        }

    def _load(self, sample: dict):
        from blm_clip.data import load_pil_image
        return load_pil_image(sample, self.image_root)


# ---------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps

    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 数据 ----
    clip_collator = ClipCollator(
        tokenizer=tokenizer,
        max_text_len=cfg["model"]["text_max_len"],
        patch_size=cfg["data"]["patch_size"],
        merge_size=cfg["data"]["merge_size"],
        min_pixels=cfg["data"]["min_pixels"],
        max_pixels=cfg["data"]["max_pixels"],
        vision_type=cfg["model"]["vision_type"],
    )
    train_set = FGDataset(cfg["data"]["fg_train_jsonl"], tc["max_regions_per_image"])
    train_loader = DataLoader(
        train_set, batch_size=tc["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=FGCollator(clip_collator, tokenizer, cfg["data"]["image_root"],
                              cfg["model"]["text_max_len"]),
        pin_memory=True, drop_last=True,
    )
    eval_loader = None
    if os.path.exists(cfg["data"]["eval_jsonl"]):
        eval_set = ImageTextDataset(cfg["data"]["eval_jsonl"], cfg["data"]["image_root"])
        eval_loader = DataLoader(eval_set, batch_size=tc["batch_size"], shuffle=False,
                                 num_workers=cfg["data"]["num_workers"],
                                 collate_fn=clip_collator, pin_memory=True)

    # ---- 模型 + 区域投影层 ----
    model = BLMClipModel(cfg, vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id)
    if cfg["model"].get("base_ckpt"):
        ckpt = torch.load(cfg["model"]["base_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[load] 从 {cfg['model']['base_ckpt']} 继续训练")

    # merger 前 hidden 维（qwenvit 1280 / timm 768）→ 共享嵌入空间
    if cfg["model"]["vision_type"] == "qwenvit":
        region_in_dim = model.visual.visual.config.hidden_size
    else:
        region_in_dim = model.visual.vit.num_features
    region_proj = torch.nn.Linear(region_in_dim, cfg["model"]["embed_dim"], bias=False)
    model.to(device)
    region_proj.to(device)

    # ---- 优化器：复用判别式 LR 分组，区域投影并入 proj 组 ----
    groups = model.param_groups(tc["lr_vision"], tc["lr_text"], tc["lr_proj"], tc["weight_decay"])
    groups.append({"params": list(region_proj.parameters()),
                   "lr": tc["lr_proj"], "weight_decay": 0.0, "name": "region_proj"})
    optimizer = torch.optim.AdamW(groups, betas=tuple(tc["betas"]), eps=tc["eps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_warmup_lambda(s, tc["warmup_steps"], tc["max_steps"]))

    clip_loss_fn = ClipLoss(use_dist=False)
    evaluator = Evaluator(model, eval_loader, device) if eval_loader else None
    os.makedirs(tc["output_dir"], exist_ok=True)
    use_bf16 = tc["bf16"] and device.type == "cuda"

    # ---- 训练循环 ----
    model.train()
    step, accum = 0, 0
    log_buf = {"g": 0.0, "r": 0.0, "h": 0.0, "n": 0}
    pbar = tqdm(total=tc["max_steps"], desc="fg-train")

    while step < tc["max_steps"]:
        for batch in train_loader:
            if step >= tc["max_steps"]:
                break
            regions_per_image = batch.pop("regions_per_image")
            hard_region_idx = batch.pop("hard_region_idx").to(device)
            pos_ids = batch.pop("pos_ids").to(device)
            pos_mask = batch.pop("pos_mask").to(device)
            hard_ids = batch.pop("hard_ids").to(device)
            hard_mask = batch.pop("hard_mask").to(device)
            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                # 全局损失
                img_z, txt_z, scale = model(batch)
                l_global = clip_loss_fn(img_z, txt_z, scale)

                # 区域损失（区域特征需要 merger 前 token，与全局前向分开执行——
                # 共享视觉塔权重，但多一次视觉前向；GPU 上可接受，学习版调小 max_regions）
                region_z = torch.nn.functional.normalize(
                    encode_regions(model, batch["image_inputs"], regions_per_image, region_proj),
                    dim=-1)
                pos_z = model.encode_text(pos_ids, pos_mask)
                l_region = region_infonce_loss(region_z, pos_z, model.clamped_logit_scale())

                # 难负例 margin 损失
                hard_z = model.encode_text(hard_ids, hard_mask)
                l_hard = hard_negative_margin_loss(
                    region_z, pos_z, hard_z, hard_region_idx, margin=tc["hard_margin"])

                loss = (l_global + tc["lambda_region"] * l_region
                        + tc["lambda_hard"] * l_hard) / tc["accum_steps"]

            loss.backward()
            log_buf["g"] += l_global.item(); log_buf["r"] += l_region.item()
            log_buf["h"] += l_hard.item();  log_buf["n"] += 1
            accum += 1

            if accum % tc["accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(region_proj.parameters()), tc["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)

                if step % tc["log_every"] == 0 and log_buf["n"] > 0:
                    pbar.set_postfix(
                        L_g=f"{log_buf['g']/log_buf['n']:.3f}",
                        L_r=f"{log_buf['r']/log_buf['n']:.3f}",
                        L_h=f"{log_buf['h']/log_buf['n']:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.1e}")
                    log_buf = {"g": 0.0, "r": 0.0, "h": 0.0, "n": 0}

                if evaluator and step % tc["eval_every"] == 0:
                    metrics = evaluator.run()
                    tqdm.write(f"[step {step}] eval: {metrics}  "
                               f"（重点看全局 hit@5 是否回退，方案 7 评估要求）")

                if step % tc["save_every"] == 0:
                    path = os.path.join(tc["output_dir"], f"ckpt_step{step}.pt")
                    torch.save({"step": step, "model": model.state_dict(),
                                "region_proj": region_proj.state_dict(), "config": cfg}, path)
                    tqdm.write(f"[save] {path}")

    pbar.close()
    path = os.path.join(tc["output_dir"], f"ckpt_step{step}.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "region_proj": region_proj.state_dict(), "config": cfg}, path)
    print(f"[done] 最终 checkpoint: {path}")


if __name__ == "__main__":
    main()
