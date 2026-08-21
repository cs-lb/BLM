# -*- coding: utf-8 -*-
"""
数据侧 binpack：平抑 step 级 token 波动
==========================================
与 packing 的关系（两个互补层级）：
    - packing（模型侧/collator）：batch 内所有图的 patch 直接拼接，
      消除 batch 内 padding 浪费，但**每个 step 的总 token 数仍然不定**；
    - binpack（数据侧/本模块）：按样本 token 数把样本"装箱"，
      让每桶（= 一个 step）的总 token 数稳定在 bin_token 附近，
      保证 GPU 利用率稳定、显存可预测、DDP 各卡步调一致。

两种策略（对齐 BLM 文档的数据侧改造）：
    1. 在线组 bin（业务数据集）：图片尺寸有上限（如 392x224），
       流式顺序累加，塞满一个 bin 就 yield —— 无需预处理、无需全局信息；
    2. 离线组 bin（开源数据集）：图片尺寸无上下限、长尾分布，
       先离线统计所有样本的 token 数，再用全局装箱算法
       （binpacking.to_constant_volume）近似最优分桶 —— 填充率更高。
"""

import json
import logging
import math
import os
import random
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# token 数估算（不加载图片，仅按尺寸元数据计算）
# ---------------------------------------------------------------
def estimate_image_tokens(
    height: int,
    width: int,
    patch_size: int = 14,
    factor: int = 28,
    min_pixels: int = 100352,
    max_pixels: int = 351232,
) -> int:
    """按 smart_resize 规则估算一张图的 patch token 数（gh * gw）。

    与 data.py::smart_resize 使用同一套参数，保证 binpack 的统计
    与训练时真实 token 数一致。
    """
    from .data import smart_resize

    rh, rw = smart_resize(height, width, factor, min_pixels, max_pixels)
    return (rh // patch_size) * (rw // patch_size)


# ---------------------------------------------------------------
# 策略一：在线组 bin（流式，适合尺寸有上限的业务数据集）
# ---------------------------------------------------------------
def online_bin_packer(
    src: Iterable[Dict],
    token_fn: Callable[[Dict], int],
    bin_token: int,
    log_ratio_prob: float = 0.01,
) -> Iterator[List[Dict]]:
    """流式装箱：顺序读取样本，累加 token 直到再加就会超过 bin_token，
    此时封箱 yield。样本尺寸越均匀，填充率越高。

    参数：
        src:             样本流，每个元素为 dict（至少包含图片与文本信息）
        token_fn:        从样本计算 token 数的函数
        bin_token:       每桶 token 预算，例如 448 * 64（64 张最大业务图）
        log_ratio_prob:  按概率打印填充率日志（避免刷屏）
    """
    current_bin: List[Dict] = []
    current_tokens = 0

    for sample in src:
        sample_tokens = token_fn(sample)

        # 再加会超预算且当前 bin 非空 -> 封箱
        if current_tokens + sample_tokens > bin_token and current_bin:
            if random.random() < log_ratio_prob:
                logger.info(
                    "yield bin: %d samples, %d tokens (fill %.2f%%)",
                    len(current_bin), current_tokens, current_tokens / bin_token * 100,
                )
            yield current_bin
            current_bin, current_tokens = [], 0

        current_bin.append(sample)
        current_tokens += sample_tokens

    if current_bin:  # 最后一个不满的 bin 也要产出
        logger.info(
            "yield last bin: %d samples, %d tokens (fill %.2f%%)",
            len(current_bin), current_tokens, current_tokens / bin_token * 100,
        )
        yield current_bin


# ---------------------------------------------------------------
# 策略二：离线组 bin（全局装箱，适合尺寸长尾的开源数据集）
# ---------------------------------------------------------------
def offline_pack_data(
    sample_list: Sequence[Dict],
    sample_token_nums: Sequence[int],
    pack_length: int,
) -> Tuple[List[List[Dict]], List[int]]:
    """离线装箱：基于全部样本的 token 数做全局近似最优分桶。

    优先使用 binpacking 库的 to_constant_volume（贪心均衡算法），
    未安装时退化为 first-fit decreasing（先排序再贪心，效果接近）。

    返回：
        packed_data:    List[List[Dict]]，外层是 bin，内层是该 bin 的样本
        sum_tokens_list: 每个 bin 的总 token 数（用于监控填充率）
    """
    indexed = list(enumerate(sample_token_nums))  # [(样本下标, token数), ...]

    try:
        import binpacking  # pip install binpacking
        grouped_indices = binpacking.to_constant_volume(indexed, pack_length, weight_pos=1)
        groups = [[idx for idx, _ in group] for group in grouped_indices]
    except ImportError:
        groups = _first_fit_decreasing(indexed, pack_length)

    packed_data: List[List[Dict]] = []
    sum_tokens_list: List[int] = []
    for group in groups:
        packed_data.append([sample_list[i] for i in group])
        sum_tokens_list.append(sum(sample_token_nums[i] for i in group))

    if sum_tokens_list:
        fill = sum(sum_tokens_list) / (len(sum_tokens_list) * pack_length) * 100
        logger.info("offline binpack: %d bins, avg fill %.2f%%", len(groups), fill)
    return packed_data, sum_tokens_list


def _first_fit_decreasing(
    indexed: List[Tuple[int, int]], pack_length: int
) -> List[List[int]]:
    """备用装箱：按 token 数降序，依次放入第一个装得下的 bin（装不下开新 bin）。"""
    bins: List[List[int]] = []
    remaining: List[int] = []
    for idx, tokens in sorted(indexed, key=lambda x: -x[1]):
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= tokens:
                bins[b].append(idx)
                remaining[b] -= tokens
                placed = True
                break
        if not placed:
            bins.append([idx])
            remaining.append(pack_length - tokens)
    return bins


# ---------------------------------------------------------------
# shuffle & batch：离线装箱前的标准动作
# ---------------------------------------------------------------
def shuffle_then_pack(
    sample_list: Sequence[Dict],
    token_fn: Callable[[Dict], int],
    pack_length: int,
    seed: int = 42,
) -> Tuple[List[List[Dict]], List[int]]:
    """开源数据集的标准流程：先 shuffle 打散，再统计 token 数离线装箱。

    shuffle 的目的：避免数据中相邻样本同源（同一视频/同一类目）被打进
    同一个 bin，保证每个 bin 内的样本多样性（对比学习的负例质量）。
    """
    indices = list(range(len(sample_list)))
    random.Random(seed).shuffle(indices)
    shuffled = [sample_list[i] for i in indices]
    token_nums = [token_fn(s) for s in shuffled]
    return offline_pack_data(shuffled, token_nums, pack_length)


# ---------------------------------------------------------------
# 面向训练流程的一站式建仓接口
# ---------------------------------------------------------------
def _read_image_size(path: str) -> Tuple[int, int]:
    """读取图片尺寸。PIL 惰性解码，只读文件头，不解码像素，速度很快。"""
    with Image.open(path) as im:
        return im.size  # (width, height)


def build_token_nums(
    samples: Sequence[Dict],
    image_root: str,
    data_cfg: Dict,
    cache_path: str = None,
) -> List[int]:
    """为全部样本计算 patch token 数（离线统计，结果带磁盘缓存）。

    缓存键为 jsonl 中的 image 字段；下次启动直接读缓存，跳过逐个读文件头。
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            token_map = json.load(f)
        if all(s["image"] in token_map for s in samples):
            logger.info("token 统计命中缓存：%s（%d 条）", cache_path, len(samples))
            return [token_map[s["image"]] for s in samples]
        logger.info("缓存不完整，重新统计 token 数...")

    factor = data_cfg["patch_size"] * data_cfg["merge_size"]
    token_map: Dict[str, int] = {}
    for s in tqdm(samples, desc="统计图片 token 数"):
        path = s["image"] if os.path.isabs(s["image"]) else os.path.join(image_root, s["image"])
        w, h = _read_image_size(path)
        token_map[s["image"]] = estimate_image_tokens(
            h, w,
            patch_size=data_cfg["patch_size"], factor=factor,
            min_pixels=data_cfg["min_pixels"], max_pixels=data_cfg["max_pixels"],
        )

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(token_map, f)
        logger.info("token 统计已缓存到 %s", cache_path)
    return [token_map[s["image"]] for s in samples]


def build_bins(
    samples: Sequence[Dict],
    token_nums: Sequence[int],
    strategy: str = "online",
    bin_token: int = 28672,
    seed: int = 42,
) -> List[List[Dict]]:
    """把样本序列重排为 bin 列表（每个 bin = 一个训练 step 的 batch）。

    参数：
        strategy:  "online"  流式贪心（业务数据，尺寸有上限、分布集中）
                   "offline" shuffle + 全局装箱（开源数据，尺寸长尾）
        bin_token: 每桶 token 预算，如 448*64=28672
    返回：
        List[List[Dict]]：外层是 bin，内层是该 bin 的样本 dict
    """
    if strategy == "online":
        keyed = [dict(s, _tokens=t) for s, t in zip(samples, token_nums)]
        bins = list(online_bin_packer(keyed, lambda s: s["_tokens"], bin_token))
        sum_tokens = [sum(s["_tokens"] for s in b) for b in bins]
    else:
        idx = list(range(len(samples)))
        random.Random(seed).shuffle(idx)          # 先打散，防同源样本同 bin
        shuffled = [samples[i] for i in idx]
        nums = [token_nums[i] for i in idx]
        keyed = [dict(s, _tokens=t) for s, t in zip(shuffled, nums)]
        bins, sum_tokens = offline_pack_data(keyed, [s["_tokens"] for s in keyed], bin_token)

    # 清理临时字段，还原为原始样本 dict
    for b in bins:
        for s in b:
            s.pop("_tokens", None)

    total = sum(sum_tokens)
    fill = total / (len(bins) * bin_token) * 100 if bins else 0.0
    logger.info(
        "build_bins(%s): %d 样本 -> %d bins，平均 %d 样本/bin，总填充率 %.2f%%",
        strategy, len(samples), len(bins),
        len(samples) // max(1, len(bins)), fill,
    )
    return bins
