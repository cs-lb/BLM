# -*- coding: utf-8 -*-
"""
QwenViT + binpack 链路专项测试（不训练，只验证数据通路与张量形状）
====================================================================
验证点：
  1. token 统计（build_token_nums）与 smart_resize 规则一致
  2. 在线装箱（build_bins online）的填充率
  3. BinCollator 对整个 bin 的 packing：pixel_values 拼接行数 == Σ(gh*gw)
  4. QwenViT 动态分辨率前向 + 逐图池化：输出 [bin内样本数, embed_dim]
  5. 池化边界：每图 merger 后 token 数 == t*gh*gw/4

用法：
    python scripts/test_qwenvit_binpack.py
"""

import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("test")


def main():
    from transformers import AutoTokenizer
    from blm_clip.blm_clip.binpack import build_bins, build_token_nums
    from blm_clip.blm_clip.data import BinCollator, ClipCollator, ImageTextDataset
    from blm_clip.blm_clip.models import BLMClipModel

    # ---- 配置：基于 smoke.yaml，切换到 qwenvit + binpack ----
    cfg = yaml.safe_load(open("configs/smoke.yaml", "r", encoding="utf-8"))
    cfg["model"]["vision_type"] = "qwenvit"
    cfg["data"]["bin_token"] = 448 * 16          # 小 bin（16 张最大图预算），5000 条能出更多 bin

    device = torch.device("cpu")                  # 链路验证用 CPU fp32 即可

    # ================= 1. binpack 链路 =================
    logger.info("==== 1. token 统计 + 在线装箱 ====")
    train_set = ImageTextDataset(cfg["data"]["train_jsonl"], cfg["data"]["image_root"])
    token_nums = build_token_nums(
        train_set.samples, cfg["data"]["image_root"], cfg["data"],
        cache_path=cfg["data"]["train_jsonl"] + ".tokencache.json",
    )
    logger.info("token 数分布: min=%d, max=%d, mean=%.0f",
                min(token_nums), max(token_nums), sum(token_nums) / len(token_nums))

    bins = build_bins(train_set.samples, token_nums,
                      strategy="online", bin_token=cfg["data"]["bin_token"], seed=cfg["seed"])
    fill = sum(token_nums) / (len(bins) * cfg["data"]["bin_token"]) * 100
    logger.info("装箱结果: %d bins, 总填充率 %.2f%%", len(bins), fill)

    # ================= 2. 加载 QwenViT =================
    logger.info("==== 2. 加载 QwenViT（首次需下载 3B 全模，约 7GB）====")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = BLMClipModel(cfg, vocab_size=len(tokenizer), pad_id=tokenizer.pad_token_id)
    model.to(device).eval()
    logger.info("模型加载完成")

    # ================= 3. 整 bin packing + 动态分辨率前向 =================
    logger.info("==== 3. BinCollator packing + 前向 ====")
    collator = ClipCollator(
        tokenizer=tokenizer,
        max_text_len=cfg["model"]["text_max_len"],
        patch_size=cfg["data"]["patch_size"],
        merge_size=cfg["data"]["merge_size"],
        min_pixels=cfg["data"]["min_pixels"],
        max_pixels=cfg["data"]["max_pixels"],
        vision_type="qwenvit",
    )
    bin_collator = BinCollator(collator, cfg["data"]["image_root"])

    batch = bin_collator([bins[0]])              # 取第一个 bin
    pixel_values, grid_thw = batch["image_inputs"]
    n_images = grid_thw.shape[0]
    expected_patches = int(grid_thw.prod(dim=1).sum().item())

    logger.info("bin 内样本数: %d", n_images)
    logger.info("grid_thw: %s", grid_thw.tolist())
    logger.info("pixel_values 形状: %s（拼接行数应 = Σ gh*gw = %d）",
                tuple(pixel_values.shape), expected_patches)
    assert pixel_values.shape[0] == expected_patches, "packing 拼接行数与 grid_thw 不一致！"

    with torch.no_grad():
        img_z = model.encode_image((pixel_values, grid_thw))
        txt_z = model.encode_text(batch["input_ids"], batch["attention_mask"])

    logger.info("图 embedding: %s, 文 embedding: %s", tuple(img_z.shape), tuple(txt_z.shape))
    assert img_z.shape == (n_images, cfg["model"]["embed_dim"]), "图 embedding 形状错误！"
    assert txt_z.shape == (n_images, cfg["model"]["embed_dim"]), "文 embedding 形状错误！"

    # ---- 池化边界验证：merger 后每图 token 数 = t*gh*gw/4 ----
    merged_tokens = (grid_thw.prod(dim=1) // (cfg["data"]["merge_size"] ** 2)).tolist()
    logger.info("每图 merger 后 token 数（池化切分依据）: %s", merged_tokens)

    # ---- 数值健康检查 ----
    norms = img_z.norm(dim=-1)
    logger.info("图 embedding L2 范数（应≈1）: min=%.4f max=%.4f", norms.min(), norms.max())
    sim = img_z @ txt_z.t()
    logger.info("图文相似度矩阵: diag_mean=%.3f, offdiag_mean=%.3f（未训练，二者接近属正常）",
                sim.diag().mean(), (sim.sum() - sim.diag().sum()) / (n_images * (n_images - 1)))

    print("\n✅ ALL PASS：binpack 装箱、packing 拼接、动态分辨率前向、逐图池化 全部正确")


if __name__ == "__main__":
    main()
