#!/bin/bash
set -u
set -o pipefail

echo "========================================="
echo "PROBE-SPACE ROBUSTNESS PIPELINE STARTED"
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
  python -c "import numpy; import pandas; import matplotlib; import sklearn" >/dev/null 2>&1 || REBUILD_VENV=1
  deactivate >/dev/null 2>&1 || true
fi

if [ "$REBUILD_VENV" -eq 1 ]; then
  echo "[setup] Rebuilding virtual environment..."
  rm -rf .venv
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip setuptools wheel

  echo "[setup] Installing dependencies..."
  pip install --no-cache-dir numpy pandas matplotlib scikit-learn
else
  echo "[setup] Using existing virtual environment"
  source .venv/bin/activate
fi

# -----------------------------
# 2) script path
# -----------------------------
PROBE_SCRIPT="/workspace/scripts/run_probe_space_robustness.py"

# اگر فایل فعلاً این اسم را دارد، یا rename کن یا این را بگذار:
# PROBE_SCRIPT="/workspace/run_probe_space_robustness new.py"

# -----------------------------
# 3) configs
# -----------------------------
TASKS=("mcq" "single")
METHODS=("logreg" "linsvm")

SIGMAS="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2"
NOISE_MODE="rel"
SEED=42

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
    for METHOD in "${METHODS[@]}"; do
      LOG_FILE="logs/probe_space_${TASK}_${METHOD}_${MODEL_PATH}.log"

      echo "------------------------------------------------------"
      echo "MODEL : $MODEL"
      echo "TASK  : $TASK"
      echo "METHOD: $METHOD"
      echo "LOG   : $LOG_FILE"
      echo "------------------------------------------------------"

      python -u "$PROBE_SCRIPT" \
        --model "$MODEL" \
        --task "$TASK" \
        --out_root "$OUT_ROOT" \
        --sigmas "$SIGMAS" \
        --noise_mode "$NOISE_MODE" \
        --method "$METHOD" \
        --seed "$SEED" \
        > "$LOG_FILE" 2>&1

      if [ $? -ne 0 ]; then
        echo "!!! Probe-space robustness failed for $MODEL / $TASK / $METHOD. See $LOG_FILE"
        continue
      fi
    done
  done
done

echo "========================================="
echo "PIPELINE FINISHED"
echo "Logs         : /workspace/logs/probe_space_*"
echo "Outputs (CSV): /workspace/out/reports/<MODEL>/robustness/local/probe_local/<task>/<method>/tables"
echo "Outputs (PNG): /workspace/out/reports/<MODEL>/robustness/local/probe_local/<task>/<method>/figs"
echo "========================================="