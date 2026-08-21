# -*- coding: utf-8 -*-
"""
模型模块：BLM CLIP 双塔模型
================================
结构（对齐实际落地方案）：
    图像 -> QwenViT（动态分辨率，取 Qwen2.5-VL 的 visual）-> 逐图均值池化 -> 投影层 -> 图 embedding
    文本 -> Transformer Encoder（EOS 池化）-> 投影层 -> 文 embedding
    两侧 L2 归一化到同一 embed_dim 空间，可学习温度 logit_scale 做 InfoNCE

关键点：
1. QwenViT 原生支持动态分辨率（2D-RoPE + 窗口注意力），输入是
   「拼接的 patch 序列 pixel_values + 每图网格 grid_thw」，天然适合 packing；
2. 视觉塔输出经过 merger 后每图 token 数 = t * gh * gw / merge_size^2，
   需要按 grid_thw 逐图切分后池化，得到每图一个全局向量；
3. 单阶段方案：双塔联合训练（不冻结 ViT），视觉塔用小 LR 保护预训练表征。
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# 视觉塔：QwenViT 封装
# ---------------------------------------------------------------
class QwenViTVisionTower(nn.Module):
    """加载 Qwen2.5-VL 的视觉塔（含 patch_embed + blocks + merger），
    再接一个线性投影层映射到共享嵌入维度。

    输入格式与 Qwen2.5-VL 官方一致：
        pixel_values: [sum_i(t_i*gh_i*gw_i), C * temporal * patch * patch]
                      一个 batch 内所有图片的 patch 按顺序拼接
        grid_thw:     [B, 3]，每张图的 (temporal, grid_h, grid_w)
    """

    def __init__(self, model_name: str, embed_dim: int, merge_size: int = 2):
        super().__init__()
        self.merge_size = merge_size

        # 只取视觉塔。Qwen2_5_VLForConditionalGeneration.visual 即完整视觉通路。
        # 加载整模再取子模块会占用一些内存（low_cpu_mem_usage 缓解）；
        # 首次加载后建议把 visual 单独存盘（见 README「加速加载」）。
        try:
            from transformers import Qwen2_5_VLModel as _Base
        except ImportError:  # 旧版 transformers 无独立 BaseModel
            from transformers import Qwen2_5_VLForConditionalGeneration as _Base

        base = _Base.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.visual = base.visual if hasattr(base, "visual") else base.model.visual

        vit_out_dim = self.visual.config.out_hidden_size  # merger 输出维度（= LLM hidden）
        self.proj = nn.Linear(vit_out_dim, embed_dim, bias=False)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """返回 [B, embed_dim]，每图一个 L2 归一化前的全局表征。"""
        # 视觉塔前向：输出 [sum_i(tokens_i), vit_out_dim]
        # tokens_i = t_i * gh_i * gw_i / merge_size^2（merger 已做 2x2 合并）
        feats = self.visual(pixel_values, grid_thw=grid_thw)
        # transformers>=5.x 的视觉塔返回 BaseModelOutputWithPooling：
        #   last_hidden_state = merger 前 token（Σ gh*gw，hidden_size）
        #   pooler_output     = merger 后 token（Σ gh*gw/4，out_hidden_size）← 取这个
        if not torch.is_tensor(feats):
            feats = feats.pooler_output

        # 按 grid_thw 计算每张图占用的 token 数，逐图切分 + 均值池化
        tokens_per_image = (grid_thw.prod(dim=1) // (self.merge_size ** 2)).tolist()
        chunks = torch.split(feats, tokens_per_image, dim=0)
        pooled = torch.stack([c.mean(dim=0) for c in chunks], dim=0)  # [B, vit_out_dim]
        # 视觉塔可能按 checkpoint 原生 bf16 加载，与 fp32 投影层对齐
        pooled = pooled.to(self.proj.weight.dtype)
        return self.proj(pooled)


class TimmVisionTower(nn.Module):
    """备用视觉塔：timm 固定分辨率 ViT（调试/对照用，输入需 resize 到 224）。"""

    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        import timm
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        vit_out_dim = self.vit.num_features
        self.proj = nn.Linear(vit_out_dim, embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, 224, 224]"""
        return self.proj(self.vit(images))


# ---------------------------------------------------------------
# 文本塔：Transformer Encoder（EOS 池化，CLIP 风格）
# ---------------------------------------------------------------
class TextTower(nn.Module):
    """从零训练的 Transformer Encoder。

    设计对齐 CLIP 文本塔：
    - token embedding + 可学习位置编码；
    - pre-norm Transformer Encoder；
    - 取每条文本最后一个非 padding token（EOS 位）的表征作为整句向量；
    - 词表直接复用 Qwen BPE 分词器，便于后续 LLM Continue-Training 阶段衔接。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        max_len: int = 96,
        pad_id: int = 0,
    ):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.empty(max_len, width))

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln_final = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim, bias=False)

        # 初始化（CLIP 风格）
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.01)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, L]，attention_mask: [B, L]（1=有效，0=padding）"""
        B, L = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:L].unsqueeze(0)
        # nn.TransformerEncoder 的 padding mask 语义：True = 需要屏蔽的位置
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.ln_final(x)

        # EOS 池化：取每条序列最后一个有效 token
        eos_idx = attention_mask.sum(dim=1) - 1                      # [B]
        pooled = x[torch.arange(B, device=x.device), eos_idx]        # [B, width]
        return self.proj(pooled)


# ---------------------------------------------------------------
# 双塔 CLIP 模型
# ---------------------------------------------------------------
class BLMClipModel(nn.Module):
    """图文双塔 + 可学习温度，输出归一化 embedding 供 InfoNCE 使用。"""

    def __init__(self, cfg: dict, vocab_size: int, pad_id: int):
        super().__init__()
        m = cfg["model"]

        # ---- 视觉塔 ----
        if m["vision_type"] == "qwenvit":
            self.visual = QwenViTVisionTower(
                m["vision_name"], m["embed_dim"], merge_size=cfg["data"]["merge_size"]
            )
        else:
            self.visual = TimmVisionTower(m["timm_name"], m["embed_dim"])
        if m.get("freeze_vision", False):
            for p in self.visual.parameters():
                p.requires_grad_(False)

        # ---- 文本塔 ----
        self.text = TextTower(
            vocab_size=vocab_size,
            embed_dim=m["embed_dim"],
            width=m["text_width"],
            layers=m["text_layers"],
            heads=m["text_heads"],
            max_len=m["text_max_len"],
            pad_id=pad_id,
        )

        # ---- 可学习温度：存储 log(1/tau)，前向时 exp 并 clamp 上限 ----
        init_tau = m["init_logit_scale"]
        self.max_logit_scale = math.log(m["max_logit_scale"])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    def clamped_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=self.max_logit_scale)

    def encode_image(self, image_inputs) -> torch.Tensor:
        z = self.visual(*image_inputs) if isinstance(image_inputs, (list, tuple)) else self.visual(image_inputs)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        return F.normalize(self.text(input_ids, attention_mask), dim=-1)

    def forward(self, batch: dict):
        """返回 (图embedding [B,D], 文embedding [B,D], 标量温度)"""
        img_z = self.encode_image(batch["image_inputs"])
        txt_z = self.encode_text(batch["input_ids"], batch["attention_mask"])
        return img_z, txt_z, self.clamped_logit_scale().exp()

    # ---------- 优化器参数分组（判别式学习率）----------
    def param_groups(self, lr_vision: float, lr_text: float, lr_proj: float, weight_decay: float):
        groups = [
            {"params": [p for p in self.visual.parameters() if p.requires_grad],
             "lr": lr_vision, "weight_decay": weight_decay, "name": "vision"},
            {"params": list(self.text.parameters()),
             "lr": lr_text, "weight_decay": weight_decay, "name": "text"},
            {"params": [self.logit_scale],
             "lr": lr_proj, "weight_decay": 0.0, "name": "logit_scale"},
        ]
        return [g for g in groups if len(g["params"]) > 0]
