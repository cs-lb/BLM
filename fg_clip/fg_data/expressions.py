# -*- coding: utf-8 -*-
"""
Step 2：提取引用表达式（Referring Expressions）
=================================================
方案要点：引用表达式 = 描述中指代图中特定物体/区域的短语，
必须带细粒度修饰（颜色/材质/动作/数量），纯"猫""狗"不算。

实现：SpaCy 名词短语块（noun_chunks）+ 修饰语过滤。
例："一个穿着蓝色T恤的男人在一片绿色的草地上玩一个红色的球"
  → ["穿着蓝色T恤的男人", "一片绿色的草地", "一个红色的球"]

用法：
    python -m fg_data.expressions --config configs/fg_config.yaml

输出：data/fg/expressions.jsonl   {"image": ..., "caption": ..., "expressions": [...]}
"""

import argparse
import os
import re

from .common import load_config, read_jsonl, write_jsonl

# 代词/无信息量开头，直接丢弃（方案质控规则）
_PRONOUN_HEAD = ("它", "这个", "那个", "这些", "那些", "其", "该")


def extract_with_spacy(caption: str, nlp) -> list[str]:
    """SpaCy 路径：名词短语块 + 修饰语过滤。"""
    doc = nlp(caption)
    expressions = []
    for chunk in doc.noun_chunks:
        # 只保留带修饰语的短语：长度≥2 且含形容词/动词/数词/量词（方案规则）
        if len(chunk) < 2:
            continue
        if not any(t.pos_ in ("ADJ", "VERB", "NUM") for t in chunk):
            continue
        text = chunk.text.strip()
        if len(text) < 2 or text.startswith(_PRONOUN_HEAD):
            continue
        expressions.append(text)
    return expressions


# 属性/数量词的轻量兜底（spaCy 模型未安装时降级使用，精度较低）
_ATTR_PATTERN = re.compile(
    r"[一二两三四五六七八九十百千万\d]+[个只张片条双件只匹头颗株]?[^，。；,.;]{1,12}?"
    r"(?:色|红|黄|蓝|绿|白|黑|灰|紫|粉|棕|橙|金|银|大|小|新|旧|长|短|高|矮)"
    r"[^，。；,.;]{0,10}"
)


def extract_fallback(caption: str) -> list[str]:
    """正则兜底路径：抓取「数量词/属性词 + 名词」片段。"""
    return [m.group(0).strip() for m in _ATTR_PATTERN.finditer(caption) if len(m.group(0)) >= 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    in_path = os.path.join(pc["out_dir"], "captions.jsonl")
    samples = read_jsonl(in_path)

    nlp = None
    try:
        import spacy
        nlp = spacy.load(pc["spacy_model"])
        print(f"[info] 使用 spaCy 模型 {pc['spacy_model']}")
    except Exception:
        print(f"[warn] spaCy 模型 {pc['spacy_model']} 不可用，降级为正则兜底抽取；"
              f"建议安装：python -m spacy download {pc['spacy_model']}")

    results = []
    n_expr_total = 0
    for s in samples:
        if nlp is not None:
            exprs = extract_with_spacy(s["caption"], nlp)
        else:
            exprs = extract_fallback(s["caption"])
        # 去重 + 数量约束（方案：每图保留 3~8 个，学习版默认 1~6）
        exprs = list(dict.fromkeys(exprs))[: pc["max_expr_per_image"]]
        if len(exprs) < pc["min_expr_per_image"]:
            continue
        n_expr_total += len(exprs)
        results.append({**s, "expressions": exprs})

    out_path = os.path.join(pc["out_dir"], "expressions.jsonl")
    write_jsonl(out_path, results)
    print(f"[done] {out_path}: {len(results)} 张图，共 {n_expr_total} 个引用表达式")


if __name__ == "__main__":
    main()
