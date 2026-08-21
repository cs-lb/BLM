# -*- coding: utf-8 -*-
"""
数据模块：图文对数据集 + 动态分辨率预处理
==============================================
数据格式（jsonl，每行一条）：
    {"image": "relative/path.jpg", "text": "一张红色的球在草地上"}

动态分辨率方案（与 Qwen2.5-VL 官方预处理逐一对齐）：
    1. smart_resize：把图片缩放到 28 的倍数，且 merged-token 数
       （H*W/28²）落在 [min_tokens, max_tokens] 预算内 —— 保留长宽比；
    2. image_to_qwen_patches：按 Qwen 官方的网格重排方式把图片转成
       patch 序列（temporal 维复制 + merge 分组），并给出 grid_thw；
    3. Collator 把一个 batch 所有图片的 patch **直接拼接**（packing），
       配合视觉塔内部的 cu_seqlens 注意力，无需 padding，无计算浪费。
"""

import io
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Qwen2.5-VL 官方使用的 CLIP 归一化常数
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------
# 动态分辨率：smart_resize（移植自 qwen_vl_utils）
# ---------------------------------------------------------------
def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """把 (h, w) 调整到 factor 的整数倍，且总像素夹在 [min_pixels, max_pixels]。

    factor = patch_size * merge_size = 14 * 2 = 28，
    即约束「merged token 数 = h*w/factor²」落在预算内。
    """
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)

    if h_bar * w_bar > max_pixels:          # 超出预算：等比缩小
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(math.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(math.floor(width / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:        # 小于下限：等比放大
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil(height * beta / factor)) * factor
        w_bar = int(math.ceil(width * beta / factor)) * factor
    return h_bar, w_bar


def image_to_qwen_patches(
    img: torch.Tensor,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把归一化后的图片张量转成 Qwen 视觉塔的输入格式。

    与 transformers 的 Qwen2VLImageProcessor 完全一致的网格重排：
      输入  img: [C, H, W]，H、W 均为 patch_size*merge_size 的倍数
      输出  patches: [gh*gw, C*temporal*patch*patch]
            grid_thw: [3] = (1, gh, gw)
    """
    C, H, W = img.shape
    gh, gw = H // patch_size, W // patch_size

    # 图片的 temporal 维为 1，复制到 temporal_patch_size 以适配 Conv3d
    x = img.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)      # [T, C, H, W]

    grid_t = 1
    x = x.reshape(
        grid_t, temporal_patch_size, C,
        gh // merge_size, merge_size, patch_size,
        gw // merge_size, merge_size, patch_size,
    )
    # 重排为 (t, gh/m, gw/m, m, m, C, T, p, p)，与官方 processor 一致
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = x.reshape(grid_t * gh * gw, C * temporal_patch_size * patch_size * patch_size)

    grid_thw = torch.tensor([grid_t, gh, gw], dtype=torch.long)
    return patches, grid_thw


# ---------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------
class ImageTextDataset(Dataset):
    """jsonl 图文对数据集。学习版规模（万~百万级）用 jsonl 足够；
    工业级（千万+）建议转 webdataset tar 分片，接口保持不变。"""

    def __init__(self, jsonl_path: str, image_root: str):
        self.image_root = image_root
        self.samples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {"image": load_pil_image(s, self.image_root), "text": s["text"]}


def load_pil_image(sample: Dict, image_root: str) -> Image.Image:
    """按 jsonl 中的路径字段加载图片（BytesIO 读取，防止文件句柄泄漏）。"""
    path = sample["image"] if os.path.isabs(sample["image"]) else os.path.join(image_root, sample["image"])
    with open(path, "rb") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


# ---------------------------------------------------------------
# bin 粒度数据集与 collator（配合 binpack 使用）
# ---------------------------------------------------------------
class BinDataset(Dataset):
    """每个元素是一个 bin（样本 dict 的列表），即一个训练 step 的 batch。

    图片不在 __getitem__ 里解码，而是在 BinCollator 中惰性加载——
    这样 num_workers 子进程只传递轻量的路径 dict，解码发生在 collator。
    """

    def __init__(self, bins: List[List[Dict]]):
        self.bins = bins

    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, idx: int) -> List[Dict]:
        return self.bins[idx]


class BinCollator:
    """DataLoader(batch_size=1) 的 collate_fn：取出一个 bin，
    解码其中所有图片后，交给 ClipCollator 对整个 bin 做 packing。

    最终产出的 batch 结构与 ClipCollator 完全一致：
    pixel_values 是整个 bin 内所有图 patch 的拼接，grid_thw 是样本数行。
    """

    def __init__(self, clip_collator: "ClipCollator", image_root: str):
        self.clip_collator = clip_collator
        self.image_root = image_root

    def __call__(self, bin_batch: List[List[Dict]]) -> Dict:
        samples = bin_batch[0]  # batch_size=1，取出一个 bin
        loaded = [{"image": load_pil_image(s, self.image_root), "text": s["text"]} for s in samples]
        return self.clip_collator(loaded)


# ---------------------------------------------------------------
# Collator：动态分辨率 + packing
# ---------------------------------------------------------------
class ClipCollator:
    def __init__(
        self,
        tokenizer,
        max_text_len: int,
        patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
        vision_type: str = "qwenvit",
    ):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.vision_type = vision_type
        self.factor = patch_size * merge_size

    def _process_image_qwenvit(self, img: Image.Image):
        """动态分辨率路径：smart_resize -> 归一化 -> Qwen patch 格式"""
        w, h = img.size
        rh, rw = smart_resize(h, w, self.factor, self.min_pixels, self.max_pixels)
        img = img.resize((rw, rh), Image.BICUBIC)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGE_MEAN)) / np.array(IMAGE_STD)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # [C, H, W]
        return image_to_qwen_patches(tensor, self.patch_size, self.merge_size)

    def _process_image_timm(self, img: Image.Image):
        """备用路径：固定 224 分辨率"""
        import torchvision.transforms as T
        tf = T.Compose([
            T.Resize(224, interpolation=Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
        return tf(img)

    def __call__(self, batch: List[Dict]) -> Dict:
        texts = [b["text"] for b in batch]

        # ---- 文本：动态 padding 到 batch 内最长 ----
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # ---- 图像 ----
        if self.vision_type == "qwenvit":
            patch_list, grid_list = [], []
            for b in batch:
                patches, grid_thw = self._process_image_qwenvit(b["image"])
                patch_list.append(patches)
                grid_list.append(grid_thw)
            # packing：所有图的 patch 直接拼接，无 padding
            image_inputs = (torch.cat(patch_list, dim=0), torch.stack(grid_list, dim=0))
        else:
            image_inputs = torch.stack([self._process_image_timm(b["image"]) for b in batch])

        return {
            "image_inputs": image_inputs,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


def move_batch_to_device(batch: Dict, device) -> Dict:
    """把 collator 产出的异构 batch 搬到 GPU。"""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(x.to(device, non_blocking=True) for x in v)
        else:
            out[k] = v
    return out
