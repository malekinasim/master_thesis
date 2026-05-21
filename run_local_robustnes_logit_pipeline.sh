#!/bin/bash
set -u
set -o pipefail

echo "========================================="
echo "LOCAL LOGIT LENS ROBUSTNESS PIPELINE STARTED"
echo "========================================="

cd /workspace
mkdir -p logs out
mkdir -p /workspace/.cache/huggingface

# env
export PYTHONPATH=/workspace
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface

# اگر لازم است:
# export HF_TOKEN="your_token_here"

# -----------------------------
# 1) venv
# -----------------------------
REBUILD_VENV=0

if [ ! -d ".venv" ]; then
  REBUILD_VENV=1
else
  source .venv/bin/activate
  python -c "import torch; import transformers; import pandas; import matplotlib" >/dev/null 2>&1 || REBUILD_VENV=1
  deactivate >/dev/null 2>&1 || true
fi

if [ "$REBUILD_VENV" -eq 1 ]; then
  echo "[setup] Rebuilding virtual environment..."
  rm -rf .venv
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip setuptools wheel

  echo "[setup] Installing dependencies..."
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  pip install --no-cache-dir transformers accelerate tqdm numpy pandas matplotlib scikit-learn
else
  echo "[setup] Using existing virtual environment"
  source .venv/bin/activate
fi

# -----------------------------
# 2) script path
# -----------------------------
LOCAL_LL_SCRIPT="/workspace/scripts/run_local_logitlens_robustness.py"

# اگر فایل جای دیگری است، این مسیر را عوض کن
# LOCAL_LL_SCRIPT="/workspace/run_local_logitlens_robustness.py"

# -----------------------------
# 3) configs
# -----------------------------
TASKS=("mcq" "single")

SIGMAS="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2"
NOISE_MODE="rel"
REPEATS=3
SEED=42
BATCH_SIZE=64

# اگر خواستی فقط بخشی از داده را برای تست سریع اجرا کنی:
MAX_ITEMS=0

DATASET_ROOT="/workspace/data"
OUT_ROOT="/workspace/out"

MODELS=(
  "Qwen/Qwen2.5-0.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# -----------------------------
# 4) run
# -----------------------------
for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL//\//__}"

  for TASK in "${TASKS[@]}"; do
    LOG_FILE="logs/local_logitlens_${TASK}_${MODEL_PATH}.log"

    echo "------------------------------------------------------"
    echo "MODEL: $MODEL"
    echo "TASK : $TASK"
    echo "LOG  : $LOG_FILE"
    echo "------------------------------------------------------"

    EXTRA_ARGS=()
    if [ "$MAX_ITEMS" -gt 0 ]; then
      EXTRA_ARGS+=(--max_items "$MAX_ITEMS")
    fi

    python -u "$LOCAL_LL_SCRIPT" \
      --model "$MODEL" \
      --task "$TASK" \
      --dataset_root "$DATASET_ROOT" \
      --out_root "$OUT_ROOT" \
      --sigmas "$SIGMAS" \
      --noise_mode "$NOISE_MODE" \
      --repeats "$REPEATS" \
      --seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      "${EXTRA_ARGS[@]}" \
      > "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
      echo "!!! Local Logit Lens robustness failed for $MODEL ($TASK). See $LOG_FILE"
      continue
    fi
  done
done

echo "========================================="
echo "PIPELINE FINISHED"
echo "Logs         : /workspace/logs/local_logitlens_*"
echo "Outputs (CSV): /workspace/out/reports/<MODEL__/robustness/local/logitlens_local/<task>/tables"
echo "Outputs (PNG): /workspace/out/reports/<MODEL__/robustness/local/logitlens_local/<task>/figs"
echo "========================================="