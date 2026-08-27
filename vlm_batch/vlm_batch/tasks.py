# -*- coding: utf-8 -*-
"""
任务注册表：prompt 构建 + 输出解析校验
==========================================
每个任务定义：
    build_prompt(record) -> str        从输入记录构建文本 prompt（图片由框架注入）
    parse(record, raw_text) -> dict|None  解析并校验模型输出，不合格返回 None（进失败重试）
    max_tokens / temperature / max_pixels / prompt_version

prompt_version 很关键：改 prompt 后必须 +1，缓存 key 含版本号，
旧版本结果自动作废重跑（否则换 prompt 会错误命中旧缓存）。
"""

import json
import re

# ---------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """从模型输出中提取第一个 JSON 对象（容忍 markdown 代码块包裹）。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------
# 任务 1：dense caption（飞轮 / FG-CLIP 详细描述）
# ---------------------------------------------------------------
CAPTION_PROMPT = (
    "请详细描述这张图片，要求：\n"
    "1. 列出图中所有主要物体，并说明其颜色、材质、状态等属性；\n"
    "2. 描述物体的动作和相互关系；\n"
    "3. 描述场景环境；\n"
    "4. 输出一段连贯文字，不要列表，不要分点。"
)


def caption_build(record: dict) -> str:
    return CAPTION_PROMPT


def caption_parse(record: dict, raw: str) -> dict | None:
    text = " ".join(raw.split())
    if len(text) < 20:                      # 过短视为生成失败
        return None
    return {"image": record["image"], "caption": text}


# ---------------------------------------------------------------
# 任务 2：候选拒绝理由生成（72B PE 口径，5~7 个候选混入无关项）
# ---------------------------------------------------------------
REASONS_LIBRARY = """R01 内容包含联系方式、二维码，涉嫌站外导流
R02 内容含有"最有效""包治"等夸大功效的绝对化用语，涉嫌虚假宣传
R03 人物着装暴露、动作挑逗，涉嫌低俗擦边
R04 画面出现赌具、筹码或赌博平台信息，涉嫌赌博
R05 画面出现枪支、管制刀具等违禁物品
R06 画面血腥暴力，令人不适
R07 未经授权使用知名品牌 logo，涉嫌侵权"""

CANDIDATE_PROMPT = f"""【角色】你是内容安全审核领域的资深专家。

【任务】观察这张图片，完成两步：
1. 从下方候选拒绝理由库中，选出 2~3 条与图片【确实相关】的；
2. 再从剩余理由中随机搭配 3~4 条【不相关】的，混合输出。

【候选拒绝理由库】
{REASONS_LIBRARY}

【判定标准】
- "相关"必须基于图片中清晰可见的内容，禁止仅凭类目、风格、氛围推测；
- 拿不准的一律放入"不相关"。

【输出格式】严格输出 JSON，不要输出其他内容：
{{"relevant": [{{"id": "R03", "evidence": "图中……"}}],
  "irrelevant": [{{"id": "R01"}}, {{"id": "R02"}}]}}"""


def candidate_build(record: dict) -> str:
    return CANDIDATE_PROMPT


def candidate_parse(record: dict, raw: str) -> dict | None:
    obj = extract_json(raw)
    if not obj or "relevant" not in obj or "irrelevant" not in obj:
        return None
    if not (1 <= len(obj["relevant"]) <= 3):
        return None
    # 展开为 pair 级记录（每个候选理由一条），供裁判任务消费
    pairs = []
    for r in obj["relevant"]:
        pairs.append({"reason_id": r["id"], "reason_text": r.get("evidence", ""),
                      "prior": "relevant"})
    for r in obj["irrelevant"]:
        pairs.append({"reason_id": r["id"], "reason_text": "", "prior": "irrelevant"})
    return {"image": record["image"], "candidates": pairs}


# ---------------------------------------------------------------
# 任务 3：裁判判定（相关/不相关）
# ---------------------------------------------------------------
JUDGE_PROMPT_TMPL = """【角色】你是严格、保守的内容审核判定专家。

【任务】判定图片与给定拒绝理由是否相关。

【拒绝理由】{reason_id} {reason_text}

【判定标准】
- 仅当图片内容【明确构成】该理由描述的违规时，判"相关"；
- 以下情况一律判"不相关"：
  · 仅凭风格、类目、拍摄场景相似（如"穿吊带"不等于色情低俗）；
  · 理由描述的元素在图中不可见、看不清、无法确认；
  · 需要脑补、联想才能建立的关联（防"假相关"）。

【输出格式】严格输出 JSON：
{{"analysis": "先描述图中看到了什么，再对照理由要件给结论",
  "label": "相关" | "不相关", "confidence": 0~1}}"""


def judge_build(record: dict) -> str:
    return JUDGE_PROMPT_TMPL.format(
        reason_id=record.get("reason_id", ""),
        reason_text=record.get("reason_text", record.get("text", "")))


def judge_parse(record: dict, raw: str) -> dict | None:
    obj = extract_json(raw)
    if not obj or obj.get("label") not in ("相关", "不相关"):
        return None
    if not obj.get("analysis") or len(obj["analysis"]) < 10:
        return None                          # 无分析过程的判定不可信，重试
    return {
        "image": record["image"],
        "reason_id": record.get("reason_id", ""),
        "reason_text": record.get("reason_text", record.get("text", "")),
        "label": obj["label"],
        "teacher_score": round(float(obj.get("confidence", 0.5))
                               if obj["label"] == "相关"
                               else 1 - float(obj.get("confidence", 0.5)), 4),
        "confidence": round(float(obj.get("confidence", 0.5)), 4),
        "analysis": obj["analysis"],
    }


# ---------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------
TASKS = {
    "caption": {
        "build": caption_build, "parse": caption_parse,
        "max_tokens": 256, "temperature": 0.3,
        "max_pixels": 224 * 28 * 28,       # caption 任务用小像素预算提速
        "prompt_version": "v1",
    },
    "candidate": {
        "build": candidate_build, "parse": candidate_parse,
        "max_tokens": 512, "temperature": 0.2,
        "max_pixels": 448 * 28 * 28,       # 风险图需要看清细节
        "prompt_version": "v1",
    },
    "judge": {
        "build": judge_build, "parse": judge_parse,
        "max_tokens": 256, "temperature": 0.1,
        "max_pixels": 448 * 28 * 28,
        "prompt_version": "v1",
    },
}
