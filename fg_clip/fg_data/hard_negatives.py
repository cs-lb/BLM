# -*- coding: utf-8 -*-
"""
Step 4：构造难负例
====================
方案要点：保留核心对象，只改一个细粒度属性或动作，
使负样本与正样本"几乎一样但又确实不同"：
    "一只正在跳跃的白色猫" -> "一只正在跳跃的黑色猫"（改颜色，框不变）

改写优先级（方案 3.4）：颜色 > 动作 > 数量 > 材质/状态。

用法：
    python -m fg_data.hard_negatives --config configs/fg_config.yaml

输出：data/fg/fg_train.jsonl（最终训练数据）
    {"image": ..., "caption": ...,
     "regions": [{"expr": "正例文本", "bbox": [...], "conf": ...,
                  "hard": ["难负例文本1", "难负例文本2"]}]}
"""

import argparse
import os
import random

from .common import load_config, read_jsonl, write_jsonl

# ---------------- 属性词表（同类异值，保证改写后语义确实冲突）----------------
COLOR_WORDS = ["红色", "黄色", "蓝色", "绿色", "白色", "黑色", "灰色", "紫色", "粉色", "棕色", "橙色", "金色", "银色"]
ACTION_PAIRS = [  # 互斥动作对（双向可替换）
    ("跳跃", "趴卧"), ("奔跑", "站立"), ("坐着", "站着"), ("躺", "站"),
    ("吃", "叼"), ("打开", "关闭"), ("举起", "放下"), ("穿", "脱"),
]
NUMBER_WORDS = ["一个", "两个", "三个", "四个", "一只", "两只", "三只", "一条", "两条", "一张", "两张"]
MATERIAL_PAIRS = [
    ("干净", "脏污"), ("完整", "破损"), ("新", "旧"),
    ("木质", "金属"), ("玻璃", "塑料"), ("皮质", "布艺"),
]


def _replace_once(text: str, candidates: list[tuple[str, str]], rng: random.Random) -> str | None:
    """在文本中找到第一个出现的候选词，替换为同组异值词。"""
    hits = [(w, i) for group in candidates for w in group if (i := text.find(w)) >= 0]
    if not hits:
        return None
    # 取出现位置最靠前的词做替换
    word, pos = min(hits, key=lambda x: x[1])
    group = next(g for g in candidates if word in g)
    others = [w for w in group if w != word]
    return text[:pos] + rng.choice(others) + text[pos + len(word):]


def make_hard_negatives(expr: str, k: int, rng: random.Random) -> list[str]:
    """按优先级依次尝试 颜色 > 动作 > 数量 > 材质，产出 k 个互不相同的难负例。"""
    strategies = [
        [tuple(COLOR_WORDS)],                 # 颜色：全部颜色同组，替换为组内异色
        ACTION_PAIRS,                          # 动作：互斥对
        [tuple(NUMBER_WORDS)],                 # 数量：全部数量词同组，替换为异值
        MATERIAL_PAIRS,                        # 材质/状态：互斥对
    ]
    negatives, used = [], {expr}
    for candidates in strategies:
        for _ in range(2):                    # 每种策略最多试 2 次
            if len(negatives) >= k:
                break
            neg = _replace_once(expr, candidates, rng)
            if neg and neg not in used:
                used.add(neg)
                negatives.append(neg)
        if len(negatives) >= k:
            break
    return negatives


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    rng = random.Random(cfg["seed"])

    samples = read_jsonl(os.path.join(pc["out_dir"], "regions.jsonl"))
    results, n_pos, n_hard = [], 0, 0
    for s in samples:
        regions = []
        for r in s["regions"]:
            hard = make_hard_negatives(r["expr"], pc["hard_neg_per_expr"], rng)
            if not hard:      # 改不出难负例的区域丢弃（无可学信号）
                continue
            n_pos += 1
            n_hard += len(hard)
            regions.append({**r, "hard": hard})
        if regions:
            results.append({"image": s["image"], "caption": s["caption"], "regions": regions})

    out_path = os.path.join(pc["out_dir"], "fg_train.jsonl")
    write_jsonl(out_path, results)
    print(f"[done] {out_path}: {len(results)} 张图，{n_pos} 个正例区域，{n_hard} 条难负例")


if __name__ == "__main__":
    main()
