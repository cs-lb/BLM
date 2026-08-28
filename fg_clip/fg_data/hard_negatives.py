# -*- coding: utf-8 -*-
"""
Step 4：构造难负例
====================
方案要点：保留核心对象，只改一个细粒度属性或动作，
使负样本与正样本"几乎一样但又确实不同"：
    "一只正在跳跃的白色猫" -> "一只正在跳跃的黑色猫"（改颜色，框不变）

双后端（config: pipeline.hard_neg_backend）：
    llm  —— 大语言模型改写（默认，覆盖开放属性，不受词表限制）+ 三层质量校验
    rule —— 颜色/动作/数量/材质词表优先级改写（无模型时的兜底）

LLM 后端的质量校验（工业版"改写 + 语病校验"的学习版实现）：
    1. 结构校验：非空、不等于正例、去重、长度比例约束；
    2. 编辑相似度校验：与正例的 SequenceMatcher ratio >= 阈值——
       保证"只改一处"，防止 LLM 重写整句导致负例不"难"；
    3. 单条失败回退规则词表：LLM 产出不足 k 条时用规则补齐，保证流水线不断。

用法：
    python -m fg_data.hard_negatives --config configs/fg_config.yaml

输出：data/fg/fg_train.jsonl（最终训练数据）
    {"image": ..., "caption": ...,
     "regions": [{"expr": "正例文本", "bbox": [...], "conf": ...,
                  "hard": ["难负例文本1", "难负例文本2"]}]}
"""

import argparse
import difflib
import os
import random

from .common import load_config, read_jsonl, write_jsonl

# ---------------- 规则词表（fallback 后端；同类异值，保证改写后语义确实冲突）----------------
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


def make_hard_negatives_rule(expr: str, k: int, rng: random.Random) -> list[str]:
    """规则后端：按优先级 颜色 > 动作 > 数量 > 材质，产出至多 k 个互不相同的难负例。"""
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


# ---------------------------------------------------------------
# LLM 后端：大模型改写 + 质量校验
# ---------------------------------------------------------------
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


class LLMHardNegativeMaker:
    """LLM 难负例构造器：复用本地 Qwen2.5-VL（text-only 生成，无需图像输入）。

    与 dense_caption 同一模型权重——生产环境本来就已加载它，零额外模型成本。
    do_sample=True + 低温：改写需要一点多样性，但不能跑偏。
    """

    def __init__(self, model_name: str, max_new_tokens: int = 128):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _chat(self, prompt: str) -> str:
        """text-only 对话生成（不送图像，Qwen2.5-VL 兼容纯文本输入）。"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()

    def generate(self, expr: str, k: int) -> list[str]:
        """调用 LLM 生成并解析难负例列表（未校验，校验在 _valid_negative）。"""
        raw = self._chat(LLM_PROMPT.format(k=k + 1, expr=expr))  # 多要 1 条容错
        cands = []
        for line in raw.splitlines():
            # 剥掉可能的序号/项目符号/引号（模型偶尔不守"不要编号"的规矩）
            line = line.strip().strip("0123456789.、-·* \t\"'“”‘’")
            if line:
                cands.append(line)
        return cands


def _valid_negative(expr: str, neg: str, used: set[str],
                    min_ratio: float = 0.5) -> bool:
    """单条难负例质量校验（语病/跑偏的三道闸门）：

    1. 非空且不等于正例、未使用过；
    2. 长度比例约束：改写只动一个属性，长度不应剧烈变化；
    3. 编辑相似度 >= min_ratio：SequenceMatcher 衡量整体重合度——
       "只改一处"的难负例 ratio 通常 > 0.7；LLM 重写整句时 ratio 会显著降低，
       以此拦截"不难"或跑偏的输出（学习版的轻量语病校验）。
    """
    if not neg or neg in used:
        return False
    if not (0.4 * len(expr) <= len(neg) <= 2.5 * len(expr)):
        return False
    return difflib.SequenceMatcher(None, expr, neg).ratio() >= min_ratio


def make_hard_negatives(expr: str, k: int, rng: random.Random,
                        llm_maker: LLMHardNegativeMaker | None = None) -> list[str]:
    """统一入口：优先 LLM 构造（逐条校验），不足 k 条时用规则词表补齐。

    返回空列表时由调用方丢弃该区域（无可学信号）。
    """
    negatives, used = [], {expr}

    if llm_maker is not None:
        try:
            for neg in llm_maker.generate(expr, k):
                if len(negatives) >= k:
                    break
                if _valid_negative(expr, neg, used):
                    used.add(neg)
                    negatives.append(neg)
        except Exception as e:
            print(f"[warn] LLM 难负例生成失败（{type(e).__name__}），本条回退规则词表: {expr}")

    if len(negatives) < k:   # LLM 不可用或产出不足 -> 规则补齐（流水线不断）
        for neg in make_hard_negatives_rule(expr, k - len(negatives), rng):
            if neg not in used:
                used.add(neg)
                negatives.append(neg)
    return negatives


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    rng = random.Random(cfg["seed"])

    backend = pc.get("hard_neg_backend", "llm")
    llm_maker = None
    if backend == "llm":
        try:
            llm_maker = LLMHardNegativeMaker(
                pc.get("hard_neg_model", pc["caption_model"]))
            print(f"[info] 难负例后端: LLM（{pc.get('hard_neg_model', pc['caption_model'])}），"
                  f"单条失败自动回退规则词表")
        except Exception as e:
            print(f"[warn] LLM 加载失败（{type(e).__name__}: {e}），整体降级为规则词表后端")

    samples = read_jsonl(os.path.join(pc["out_dir"], "regions.jsonl"))
    results, n_pos, n_hard, n_llm = [], 0, 0, 0
    for s in samples:
        regions = []
        for r in s["regions"]:
            hard = make_hard_negatives(r["expr"], pc["hard_neg_per_expr"], rng, llm_maker)
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
