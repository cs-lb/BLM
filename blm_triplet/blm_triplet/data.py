# -*- coding: utf-8 -*-
"""
三元组数据集与 collator
==========================
训练数据（scripts/build_triplets.py 从裁判打标结果构建）：

    {"image": "x.jpg",
     "candidates": [
        {"text": "R03 人物着装暴露……", "label": 1, "teacher_score": 0.93, "conf": 0.95},
        {"text": "R01 内容包含联系方式……", "label": 0, "teacher_score": 0.06, "conf": 0.88},
        ...   # 每张图 3~7 个候选（1~2 正 + 2~5 负，难负例优先）
     ]}

collator 复用 blm_clip 的 ClipCollator 做图像 packing；
文本侧把 [n_anchor × K] 个候选文本一次 tokenize，训练时 reshape 回 [A, K]。
"""

import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, jsonl_path: str):
        import json
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TripletCollator:
    """图像走 ClipCollator（动态分辨率 packing）；候选文本统一 tokenize。

    产出：
        image_inputs:  packing 后的图像输入（同 blm_clip）
        cand_ids/cand_mask: [A*K, L]  候选文本（训练时 reshape [A, K, L]）
        labels:         [A, K] float（1=相关正例，0=不相关负例）
        teacher_scores: [A, K] float ∈ [0,1]
        conf_weights:   [A, K] float
    """

    def __init__(self, clip_collator, tokenizer, image_root: str, max_text_len: int = 96):
        self.clip_collator = clip_collator
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.max_text_len = max_text_len

    def __call__(self, batch: list[dict]) -> dict:
        from blm_clip.data import load_pil_image

        # ---- 图像：沿用全局 packing collator（caption 字段用占位，训练不取）----
        global_batch = self.clip_collator(
            [{"image": load_pil_image(b, self.image_root), "text": ""} for b in batch])

        # ---- 候选文本：拍平 [A*K] 后一次 tokenize ----
        texts, labels, scores, confs, k_per_anchor = [], [], [], [], []
        for b in batch:
            cands = b["candidates"]
            k_per_anchor.append(len(cands))
            for c in cands:
                texts.append(c["text"])
                labels.append(float(c["label"]))
                scores.append(float(c["teacher_score"]))
                confs.append(float(c.get("conf", 1.0)))

        # 每个 anchor 的候选数一致才能 reshape；取 batch 内最小 K 截断（构建时已统一，双保险）
        k_min = min(k_per_anchor)
        if any(k != k_min for k in k_per_anchor):
            texts, labels, scores, confs = [], [], [], []
            for b in batch:
                for c in b["candidates"][:k_min]:
                    texts.append(c["text"])
                    labels.append(float(c["label"]))
                    scores.append(float(c["teacher_score"]))
                    confs.append(float(c.get("conf", 1.0)))

        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_text_len, return_tensors="pt")
        a, k = len(batch), k_min
        return {
            "image_inputs": global_batch["image_inputs"],
            "cand_ids": enc["input_ids"].view(a, k, -1),
            "cand_mask": enc["attention_mask"].view(a, k, -1),
            "labels": torch.tensor(labels).view(a, k),
            "teacher_scores": torch.tensor(scores).view(a, k),
            "conf_weights": torch.tensor(confs).view(a, k),
        }


def move_to_device(batch: dict, device) -> dict:
    from blm_clip.data import _move_tensor
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = _move_tensor(v, device)
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(_move_tensor(x, device) for x in v)
        else:
            out[k] = v
    return out
