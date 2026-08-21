#!/bin/bash
# ============================================================
# wukong100m 数据下载一键脚本（在你自己的终端运行）
#
# 用法：
#   bash scripts/download_data.sh          # 默认下载 10 万条
#   NUM=500000 bash scripts/download_data.sh   # 下载 50 万条
#
# 特性：
#   - caffeinate 包裹：下载期间 Mac 不会睡眠
#   - 断点续跑：已下载的图片自动复用，中断后重跑不浪费
#   - 日志同时输出到屏幕和 data/download.log
# 随时中断：Ctrl+C；重新运行同一命令即可续跑
# ============================================================
set -e
cd "$(dirname "$0")/.."     # 切到 blm_clip 项目根目录

NUM=${NUM:-100000}          # 目标样本数，可用环境变量覆盖
PY=/Users/zerobliu/.workbuddy/binaries/python/envs/default/bin/python

# 防重复运行
if pgrep -f "prepare_data.py" > /dev/null; then
    echo "[!] 已有 prepare_data.py 在运行，请勿重复启动"
    exit 1
fi

echo "[start] 目标 ${NUM} 条，日志见 data/download.log"
caffeinate -i "$PY" scripts/prepare_data.py \
    --mode url \
    --input data/raw/train-00000-of-00032.parquet \
    --url-col url --text-col caption \
    --num-samples "$NUM" --workers 64 --out-dir data 2>&1 | tee -a data/download.log

echo "[done] 当前图片数：$(ls data/images | wc -l)"
echo "[done] train/eval 条数："
wc -l data/train.jsonl data/eval.jsonl
