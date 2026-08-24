# -*- coding: utf-8 -*-
"""
模型模块：QwenViT-7B 视觉塔 + BGE-M3 文本塔 + SigLIP 双标量
================================================================
    图像 → QwenViT（动态分辨率 packing）→ 逐图池化 → proj(3584→1024) → normalize
    文本 → BGE-M3（XLM-R-large）→ CLS → proj(1024→1024) → normalize
    标量：logit_scale（init ln10）、logit_bias（init −10），均可学习、免 weight decay。

工程要点：
- 视觉权重单独抽取：7B 全模 bf16 ~15GB，启动加载整模浪费内存与时间，
  先用 scripts/extract_qwenvit.py 抽出 model.visual 存盘（~1.4GB），之后秒级加载；
- transformers>=5.x 视觉塔返回 BaseModelOutputWithPooling：
  last_hidden_state = merger 前（Σgh·gw, 1280），pooler_output = merger 后（Σgh·gw/4, 3584），
  全局表征取 pooler_output；
- BGE-M3 是 SentencePiece 词表（250,002），与 Qwen 分词器不同体系，别混用。
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# 视觉塔：Qwen2.5-VL-7B 的 ViT
# ---------------------------------------------------------------
class QwenViTVisionTower(nn.Module):
    """输入为 packing 后的 patch 序列 + grid_thw（见 data.py），输出每图一个向量。"""

    def __init__(self, model_name: str, embed_dim: int, merge_size: int = 2,
                 visual_weights: str = None):
        super().__init__()
        self.merge_size = merge_size

        if visual_weights and os.path.exists(visual_weights):
            # 快速路径：加载抽取好的视觉塔权重（extract_qwenvit.py 产物）
            self._load_extracted(visual_weights)
        else:
            # 兜底路径：从 7B 全模取视觉塔（首次/未抽取时，慢但可用）
            try:
                from transformers import Qwen2_5_VLModel as _Base
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as _Base
            base = _Base.from_pretrained(model_name, low_cpu_mem_usage=True)
            self.visual = base.visual if hasattr(base, "visual") else base.model.visual

        # merger 输出维（7B 为 3584）→ 共享嵌入空间
        self.proj = nn.Linear(self.visual.config.out_hidden_size, embed_dim, bias=False)

    def _load_extracted(self, ckpt_path: str):
        """从抽取文件恢复：{'config': vision_config_dict, 'state_dict': ...}"""
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLVisionTransformerPretrainedModel,
        )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vision_cfg = Qwen2_5_VLVisionConfig(**payload["config"])
        self.visual = Qwen2_5_VLVisionTransformerPretrainedModel(vision_cfg)
        self.visual.load_state_dict(payload["state_dict"])

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values: [Σpatches, 1176]；grid_thw: [B, 3]。返回 [B, embed_dim]。"""
        feats = self.visual(pixel_values, grid_thw=grid_thw)
        if not torch.is_tensor(feats):
            feats = feats.pooler_output          # merger 后 token；别拿 last_hidden_state

        # 按 grid_thw 切回每图（merger 后 token 数 = t·gh·gw/4），均值池化
        tokens_per_image = (grid_thw.prod(dim=1) // (self.merge_size ** 2)).tolist()
        chunks = torch.split(feats, tokens_per_image, dim=0)
        pooled = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        pooled = pooled.to(self.proj.weight.dtype)   # 视觉塔 bf16 → 投影层 fp32 对齐
        return self.proj(pooled)


class TimmVisionTower(nn.Module):
    """备用视觉塔：timm 固定分辨率 ViT（冒烟/对调用，输入 resize 224）。"""

    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        import timm
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.vit.num_features, embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.vit(images))


# ---------------------------------------------------------------
# 文本塔：BGE-M3（CLS 池化）
# ---------------------------------------------------------------
class BgeM3TextTower(nn.Module):
    """BGE-M3 本体已是检索向编码器，语义空间天然适合余弦检索——
    这正是之前"冻结强塔做 LiT"失败教训的正面应用：文本塔已在检索友好空间。"""

    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.backbone.config.hidden_size, embed_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]        # BGE-M3 dense 检索标准取法：CLS 位
        return self.proj(cls)


# ---------------------------------------------------------------
# 双塔模型
# ---------------------------------------------------------------
class BLMSiglipModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        if m["vision_type"] == "qwenvit":
            self.visual = QwenViTVisionTower(
                m["vision_name"], m["embed_dim"],
                merge_size=cfg["data"]["merge_size"],
                visual_weights=m.get("visual_weights"),
            )
        else:
            self.visual = TimmVisionTower(m["timm_name"], m["embed_dim"])
        if m.get("freeze_vision", False):
            for p in self.visual.parameters():
                p.requires_grad_(False)

        self.text = BgeM3TextTower(m["text_name"], m["embed_dim"])

        # SigLIP 双标量（SigLIP 论文口径；均免 weight_decay）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(m["siglip_init_t"])))   # ln(10)
        self.logit_bias = nn.Parameter(torch.tensor(float(m["siglip_init_b"])))       # −10

    def encode_image(self, image_inputs) -> torch.Tensor:
        z = self.visual(*image_inputs) if isinstance(image_inputs, (list, tuple)) \
            else self.visual(image_inputs)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        return F.normalize(self.text(input_ids, attention_mask), dim=-1)

    def forward(self, batch: dict):
        img_z = self.encode_image(batch["image_inputs"])
        txt_z = self.encode_text(batch["input_ids"], batch["attention_mask"])
        return img_z, txt_z, self.logit_scale, self.logit_bias

    def param_groups(self, lr_vision, lr_text, lr_proj, weight_decay):
        """判别式学习率：双塔均预训练 → 对称小 LR；标量参数免 wd。"""
        groups = [
            {"params": [p for p in self.visual.parameters() if p.requires_grad],
             "lr": lr_vision, "weight_decay": weight_decay, "name": "vision"},
            {"params": list(self.text.parameters()),
             "lr": lr_text, "weight_decay": weight_decay, "name": "text"},
            {"params": [self.logit_scale, self.logit_bias],
             "lr": lr_proj, "weight_decay": 0.0, "name": "siglip_scalars"},
        ]
        return [g for g in groups if len(g["params"]) > 0]
