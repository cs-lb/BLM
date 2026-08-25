#!/bin/bash
# ============================================================
# AutoDL 双卡 4090 一键训练脚本
#
# 用法（在 AutoDL 实例的终端里）：
#   bash scripts/train_dual4090.sh            # 先冒烟 20 步
#   SMOKE=0 bash scripts/train_dual4090.sh    # 正式训练（yaml 里的 5000 步）
#
# 日志：outputs/dual4090_v0/train.log（tee 双写，tail -f 可实时看）
# 中断恢复：直接重跑本脚本 + --resume 最新 ckpt（见脚本尾部提示）
# ============================================================
set -e
cd "$(dirname "$0")/.."

SMOKE=${SMOKE:-1}
LOG_DIR=outputs/dual4090_v0
mkdir -p "$LOG_DIR"

# ---------- 0. 环境检查 ----------
echo "===== [0/4] 环境检查 ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"

# ---------- 1. 视觉权重检查 ----------
echo "===== [1/4] 视觉权重检查 ====="
if [ ! -f assets/qwenvit_7b_visual.pt ]; then
    echo "未发现 assets/qwenvit_7b_visual.pt，请先上传或运行："
    echo "  python scripts/extract_qwenvit.py --out assets/qwenvit_7b_visual.pt"
    exit 1
fi
ls -lh assets/qwenvit_7b_visual.pt

# ---------- 2. 数据检查 ----------
echo "===== [2/4] 数据检查 ====="
wc -l data/train.jsonl data/eval.jsonl
ls data/images | wc -l

# ---------- 3. 冒烟（可选）----------
if [ "$SMOKE" = "1" ]; then
    echo "===== [3/4] 冒烟 20 步（验证 NCCL + gather + loss 下降）====="
    torchrun --nproc_per_node=2 train.py --config configs/dual_4090.yaml --max-steps 20 \
        2>&1 | tee "$LOG_DIR/smoke.log"
    echo "冒烟通过后可正式训练：SMOKE=0 bash scripts/train_dual4090.sh"
    exit 0
fi

# ---------- 4. 正式训练 ----------
echo "===== [4/4] 正式训练（5000 步）====="
torchrun --nproc_per_node=2 train.py --config configs/dual_4090.yaml \
    2>&1 | tee -a "$LOG_DIR/train.log"

echo "训练完成。产物："
ls -lh "$LOG_DIR"
echo "恢复训练：torchrun --nproc_per_node=2 train.py --config configs/dual_4090.yaml --resume $LOG_DIR/ckpt_step<最新步数>.pt"
