# -*- coding: utf-8 -*-
"""
Step 4：LLM 批量构造难负例（Hard Negatives）
==================================================
难负例是整个细粒度方案的灵魂：保留核心对象，只改一个细粒度属性，
让负样本与正样本"几乎一样但又确实不同"，且指向同一个 bbox——
模型没法靠"找错物体"蒙混，必须学会区分属性：
    "一只正在跳跃的白色猫" -> "一只正在跳跃的黑色猫"（改颜色，框不变）

构造方式：LLM 改写（覆盖开放属性，不受固定词表限制）+ 三道质量校验：
  1. 结构校验：非空、不等于正例、去重、长度比例约束；
  2. 编辑相似度 >= MIN_SIM_RATIO：SequenceMatcher 衡量整体重合度，
     "只改一处"的难负例通常 > 0.7；LLM 重写整句时 ratio 显著降低，
     以此拦截"不难"或跑偏的输出（轻量语病校验）；
  3. LLM 产出不足 k 条时用规则词表补齐——流水线永不断。

加速方式：vLLM 把全部表达式的改写请求一次批量送入（continuous batching），
比逐条 generate 快一个数量级。

输入：data/out/regions.jsonl（Step3 产物）
输出：data/out/fg_train.jsonl（最终训练数据）
      {"image", "caption",
       "regions": [{"expr", "bbox", "conf", "hard": ["难负例1", ...]}]}

运行：PyCharm 直接点运行。
依赖：vllm transformers tqdm（vLLM 需 Linux + NVIDIA GPU）
"""

import difflib
import random

from tqdm import tqdm

import config as C
from common import out_path, read_jsonl, write_jsonl

# Prompt 设计要点：
# - "只改一个属性，其余完全不变"是"难"的前提：改多处模型很容易判负，学不到东西；
# - "互斥而非近义"保证语义确实冲突；
# - 纯行输出格式便于程序化解析。
LLM_PROMPT = (
    "你是视觉-语言对比学习的数据构造专家。给定一个描述图像区域的中文短语，"
    "请构造 {k} 个“难负例”（hard negatives），用于训练模型区分细粒度属性。\n"
    "要求：\n"
    "1. 每个难负例只改动原短语中的一个细粒度属性（颜色/动作/数量/材质/状态/大小等），"
    "其余文字保持完全不变；\n"
    "2. 改动后的语义必须与原短语确实冲突（互斥），不能是近义或无关替换；\n"
    "3. 语句通顺，符合中文表达习惯；\n"
    "4. 难负例之间互不相同；\n"
    "5. 每行输出一个难负例短语，不要编号、不要引号、不要任何解释。\n"
    "原短语：{expr}"
)

# ---------------- 规则词表（兜底；同类异值保证语义互斥） ----------------
COLOR_WORDS = ["红色", "黄色", "蓝色", "绿色", "白色", "黑色", "灰色",
               "紫色", "粉色", "棕色", "橙色", "金色", "银色"]
ACTION_PAIRS = [("跳跃", "趴卧"), ("奔跑", "站立"), ("坐着", "站着"),
                ("躺", "站"), ("吃", "叼"), ("打开", "关闭"),
                ("举起", "放下"), ("穿", "脱")]
NUMBER_WORDS = ["一个", "两个", "三个", "四个", "一只", "两只", "三只",
                "一条", "两条", "一张", "两张"]
MATERIAL_PAIRS = [("干净", "脏污"), ("完整", "破损"), ("新", "旧"),
                  ("木质", "金属"), ("玻璃", "塑料"), ("皮质", "布艺")]


def _replace_once(text: str, candidates: list, rng: random.Random) -> str | None:
    """找到文中第一个命中的候选词，替换为同组异值词。"""
    hits = [(w, i) for group in candidates for w in group
            if (i := text.find(w)) >= 0]
    if not hits:
        return None
    word, pos = min(hits, key=lambda x: x[1])   # 取位置最靠前的命中词
    group = next(g for g in candidates if word in g)
    others = [w for w in group if w != word]
    return text[:pos] + rng.choice(others) + text[pos + len(word):]


def make_rule_negatives(expr: str, k: int, rng: random.Random,
                        used: set[str]) -> list[str]:
    """规则兜底：按优先级 颜色 > 动作 > 数量 > 材质 依次尝试。"""
    strategies = [[tuple(COLOR_WORDS)], ACTION_PAIRS,
                  [tuple(NUMBER_WORDS)], MATERIAL_PAIRS]
    negatives = []
    for candidates in strategies:
        for _ in range(2):                    # 每种策略最多试 2 次（随机换词可能撞重复）
            if len(negatives) >= k:
                return negatives
            neg = _replace_once(expr, candidates, rng)
            if neg and neg not in used:
                used.add(neg)
                negatives.append(neg)
    return negatives


def _valid_negative(expr: str, neg: str, used: set[str]) -> bool:
    """单条难负例质量校验（拦截 LLM 跑偏的三道闸门）。"""
    if not neg or neg in used:
        return False
    # 长度比例约束：只动一个属性，长度不应剧烈变化
    if not (0.4 * len(expr) <= len(neg) <= 2.5 * len(expr)):
        return False
    # 编辑相似度：拦截整句重写（合格难负例 ratio 通常 > 0.7）
    return difflib.SequenceMatcher(None, expr, neg).ratio() >= C.MIN_SIM_RATIO


def parse_llm_output(raw: str) -> list[str]:
    """解析 LLM 输出：按行切分，剥掉可能的序号/项目符号/引号。"""
    cands = []
    for line in raw.splitlines():
        line = line.strip().strip("0123456789.、-·* \t\"'“”‘’")
        if line:
            cands.append(line)
    return cands


def main():
    samples = read_jsonl(out_path("regions.jsonl"))
    rng = random.Random(42)

    # ---- 拍平所有正例表达式，记录归属（样本下标, 区域内下标） ----
    flat_exprs, owners = [], []
    for si, s in enumerate(samples):
        for ri, r in enumerate(s["regions"]):
            flat_exprs.append(r["expr"])
            owners.append((si, ri))
    print(f"[info] 共 {len(flat_exprs)} 个正例表达式待构造难负例")

    # ---- vLLM 批量改写：所有表达式一次送入 ----
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=C.VLM_MODEL,
              max_model_len=C.VLM_MAX_MODEL_LEN,
              gpu_memory_utilization=C.GPU_MEM_UTIL)
    tokenizer = AutoTokenizer.from_pretrained(C.VLM_MODEL)
    # 多要 1 条容错（部分会被校验拦掉）
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user",
              "content": LLM_PROMPT.format(k=C.HARD_NEG_PER_EXPR + 1, expr=e)}],
            tokenize=False, add_generation_prompt=True)
        for e in flat_exprs
    ]
    # 改写需要多样性（温度 0.7），贪心会总产出同一种改法
    sampling = SamplingParams(temperature=C.HARD_NEG_TEMPERATURE,
                              top_p=C.HARD_NEG_TOP_P, max_tokens=128)
    print("[info] 进入 vLLM 批量推理...")
    outputs = llm.generate(prompts, sampling)

    # ---- 逐条校验 + 规则补齐 ----
    llm_ok, rule_fill = 0, 0
    hard_map: dict[tuple[int, int], list[str]] = {}
    for (si, ri), expr, out in tqdm(zip(owners, flat_exprs, outputs),
                                    total=len(owners), desc="校验难负例"):
        used = {expr}
        negatives = []
        for neg in parse_llm_output(out.outputs[0].text):
            if len(negatives) >= C.HARD_NEG_PER_EXPR:
                break
            if _valid_negative(expr, neg, used):
                used.add(neg)
                negatives.append(neg)
        llm_ok += len(negatives)
        if len(negatives) < C.HARD_NEG_PER_EXPR:
            # LLM 产出不足 -> 规则词表补齐，保证每条正例都有可学信号
            fill = make_rule_negatives(
                expr, C.HARD_NEG_PER_EXPR - len(negatives), rng, used)
            rule_fill += len(fill)
            negatives.extend(fill)
        if negatives:
            hard_map[(si, ri)] = negatives

    # ---- 回写区域并产出最终训练数据 ----
    results, n_pos, n_hard = [], 0, 0
    for si, s in enumerate(samples):
        regions = []
        for ri, r in enumerate(s["regions"]):
            hard = hard_map.get((si, ri))
            if not hard:      # 改不出难负例的区域丢弃（没有对比就没有可学信号）
                continue
            n_pos += 1
            n_hard += len(hard)
            regions.append({**r, "hard": hard})
        if regions:
            results.append({"image": s["image"], "caption": s["caption"],
                            "regions": regions})

    dst = out_path("fg_train.jsonl")
    write_jsonl(dst, results)
    print(f"[done] {dst}: {len(results)} 张图，{n_pos} 个正例区域，{n_hard} 条难负例"
          f"（LLM 通过校验 {llm_ok} 条，规则补齐 {rule_fill} 条）")
    print(f"[next] 可运行 step5_visualize.py 抽查标注质量")


if __name__ == "__main__":
    main()
