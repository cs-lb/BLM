# -*- coding: utf-8 -*-
"""
FAISS 向量检索最小 demo
========================
两种模式：

1) toy —— 随机向量，只依赖 faiss + numpy，秒级跑完，看清 API 三件套：
       build（建库）→ add（灌向量）→ search（查 top-k）
       python scripts/demo_faiss.py --mode toy

2) bge —— 用本机缓存的 BGE-M3 把「拒绝理由语料」编码成向量，做真实语义检索。
       即审核 Copilot「搜索相似案例的拒绝理由」的迷你复现：
       python scripts/demo_faiss.py --mode bge --query "图片里有大面积裸露"

核心概念（面试口径）：
- 归一化 + IndexFlatIP（内积）≡ 余弦相似度检索，与双塔训练时的 F.normalize 对齐；
- IndexFlat* 是暴力精确检索（O(N·d) 扫全库），语料 <10 万时延迟可忽略；
- 语料上百万再换 IndexIVFFlat（倒排聚类，近似检索换 10-100 倍加速），
  代码里给出了切换示例（注释块）。
"""

import argparse

import numpy as np


# ------------------------------------------------------------------
# 语料：审核场景拒绝理由（bge 模式用）
# ------------------------------------------------------------------
CORPUS = [
    "图片包含大面积皮肤裸露，涉嫌色情低俗内容",
    "图片出现国家领导人形象，涉及政治敏感",
    "图片展示血腥暴力场景，可能造成不适",
    "图片包含枪支弹药等违禁品展示",
    "图片为药品销售广告，涉嫌违规医疗宣传",
    "图片包含二维码导流信息，涉嫌站外引流",
    "图片展示未成年人不当内容，严重违规",
    "图片包含赌博网站宣传内容",
    "正常商品展示图，未发现违规内容",
    "正常风景摄影图片，内容健康",
]

QUERIES = [
    "这张图有人物没穿衣服",
    "画面里都是血，很吓人",
    "一张猫咪的照片",
]


# ------------------------------------------------------------------
# 建库 / 检索的通用逻辑（两种模式共用）
# ------------------------------------------------------------------
def build_flat_index(vecs: np.ndarray):
    """归一化后用内积索引 = 余弦相似度。返回 (index, 已归一化向量)。"""
    import faiss

    vecs = vecs.astype(np.float32).copy()
    # 与双塔训练一致：先 L2 归一化，内积即余弦
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, vecs


def search(index, query_vecs: np.ndarray, topk: int = 5):
    """返回 (scores, ids)，ids 为语料行号，score 越大约相似（余弦 ∈ [-1,1]）。"""
    import faiss

    query_vecs = query_vecs.astype(np.float32).copy()
    faiss.normalize_L2(query_vecs)
    return index.search(query_vecs, topk)


# 大规模语料的近似索引示例（>100 万向量时替换 build_flat_index）：
#
#   nlist = 4096                              # 聚类中心数（≈ 4·√N 经验值）
#   quantizer = faiss.IndexFlatIP(dim)
#   index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#   index.train(vecs)                         # IVF 必须先训练聚类
#   index.add(vecs)
#   index.nprobe = 64                         # 检索时探查的簇数：越大越准越慢


# ------------------------------------------------------------------
# 模式 1：随机向量玩具版
# ------------------------------------------------------------------
def run_toy():
    rng = np.random.default_rng(42)
    n, dim, topk = 10_000, 128, 5

    corpus = rng.standard_normal((n, dim))
    query = corpus[[7, 123]] + 0.05 * rng.standard_normal((2, dim))  # 制造两个"近邻已知"的查询

    index, _ = build_flat_index(corpus)
    scores, ids = search(index, query, topk)

    print(f"[toy] 语料 {n} 条 × {dim} 维，top-{topk} 检索：")
    for i, (s_row, i_row) in enumerate(zip(scores, ids)):
        print(f"  query{i}（真身是 corpus[{[7, 123][i]}] 加噪）→ "
              f"top1=corpus[{i_row[0]}] score={s_row[0]:.4f} | "
              f"全部命中: {i_row.tolist()}")
    print("[toy] 预期：top1 就是真身（score≈1），验证索引工作正常。")


# ------------------------------------------------------------------
# 模式 2：BGE-M3 真实语义检索
# ------------------------------------------------------------------
def encode_bge(texts, model, tokenizer, device):
    import torch

    batch = tokenizer(texts, padding=True, truncation=True,
                      max_length=96, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch)
    cls = out.last_hidden_state[:, 0]          # BGE-M3 dense 检索标准取法：CLS 位
    return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()


def run_bge(query: str = None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    # 默认 CPU：demo 只有十几条短文本，CPU 秒级完成；
    # 本机实测 BGE-M3 在 MPS 上前向会段错误（torch 2.13 + transformers 5.x），勿踩
    device = "cpu"
    print(f"[bge] 加载 BGE-M3（{device}）...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3").to(device).eval()

    queries = [query] if query else QUERIES
    index, _ = build_flat_index(encode_bge(CORPUS, model, tokenizer, device))
    scores, ids = search(index, encode_bge(queries, model, tokenizer, device), topk=3)

    print(f"[bge] 语料 {len(CORPUS)} 条拒绝理由，top-3 检索：")
    for q, s_row, i_row in zip(queries, scores, ids):
        print(f"\n  Q: {q}")
        for rank, (s, i) in enumerate(zip(s_row, i_row), 1):
            print(f"    {rank}. [{s:.4f}] {CORPUS[i]}")
    print("\n[bge] 预期：语义相近的拒绝理由排在前面（注意没出现关键词也能命中）。")


def main():
    p = argparse.ArgumentParser(description="FAISS 检索 demo")
    p.add_argument("--mode", choices=["toy", "bge"], default="toy")
    p.add_argument("--query", type=str, default=None, help="bge 模式自定义查询")
    args = p.parse_args()
    if args.mode == "toy":
        run_toy()
    else:
        run_bge(args.query)


if __name__ == "__main__":
    main()
