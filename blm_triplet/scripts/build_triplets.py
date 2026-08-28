# -*- coding: utf-8 -*-
"""
三元组构建脚本：裁判打标结果 → 训练用三元组 jsonl
====================================================
输入（Judged MLLM 对候选对的判定结果）：
    {"image": "x.jpg", "reason_text": "...", "label": "相关"|"不相关",
     "teacher_score": 0.93, "confidence": 0.95}

输出（按图聚合为 anchor-candidates 结构）：
    {"image": "x.jpg", "candidates": [
        {"text": "...", "label": 1, "teacher_score": 0.93, "conf": 0.95}, ...]}

难负例优先策略（对应三通道挖掘）：
  - 每张图保留全部正例（1~2 个）；
  - 负例按 teacher_score 降序取 top-K（分数越高的负例越"像"，越难）；
  - 教师分数缺失时用 label 默认值（正 0.95 / 负 0.05）兜底并降 conf。

用法：
    python scripts/build_triplets.py --input judged.jsonl --out data/triplets.jsonl \
        --pos-per-image 2 --neg-per-image 4
"""

import argparse
import json
import random
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="裁判打标 jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--pos-per-image", type=int, default=2)
    p.add_argument("--neg-per-image", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    by_image = defaultdict(lambda: {"pos": [], "neg": []})
    for line in open(args.input, "r", encoding="utf-8"):
        r = json.loads(line.strip())
        label = 1 if r["label"] == "相关" else 0
        score = float(r.get("teacher_score", 0.95 if label else 0.05))
        conf = float(r.get("confidence", 0.5))      # 无置信度的兜底样本降权
        item = {"text": r["reason_text"], "label": label,
                "teacher_score": score, "conf": conf}
        by_image[r["image"]]["pos" if label else "neg"].append(item)

    out, skipped = [], 0
    for image, groups in by_image.items():
        pos = groups["pos"][: args.pos_per_image]
        # 难负例优先：教师分越高越难（模型最容易误判的）
        neg = sorted(groups["neg"], key=lambda x: -x["teacher_score"])[: args.neg_per_image]
        if not pos or not neg:                    # 缺正或缺负的图无法构成三元组
            skipped += 1
            continue
        cands = pos + neg
        rng.shuffle(cands)
        out.append({"image": image, "candidates": cands})

    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {args.out}: {len(out)} anchors（跳过 {skipped} 张缺正/负例的图）")


if __name__ == "__main__":
    main()
