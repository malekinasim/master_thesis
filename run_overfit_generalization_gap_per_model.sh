#!/bin/bash
set -euo pipefail

echo "========================================="
echo "OverFit Generalization + PLOTS PIPELINE STARTED"
echo "========================================="

cd /workspace
mkdir -p logs out

# 1) venv
if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip -q

# 2) deps (this script only needs these)
echo "[setup] Installing dependencies..."
pip install -q numpy pandas matplotlib

# 3) env
export PYTHONPATH=/workspace

# 4) config (edit as needed)
OUT_ROOT="/workspace/out"
OUT_DIR="/workspace/out"


# models must be "__" format (as in your report folders)
MODELS=(
  "TinyLlama__TinyLlama-1.1B-Chat-v1.0"
  "EleutherAI__gpt-neo-125M"
  "Qwen__Qwen2.5-0.5B"
  "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B"
  "facebook__opt-125m"
  "meta-llama__Llama-3.2-1B"
)

TASKS="mcq,single"
METHODS="logreg,lda,linsvm,massmean"

# IMPORTANT: path to your script
SCRIPT="/workspace/scripts/overfit_generalization_gap_per_model.py"

# build comma list for --models
MODEL_LIST=""
for M in "${MODELS[@]}"; do
  if [ -z "$MODEL_LIST" ]; then
    MODEL_LIST="$M"
  else
    MODEL_LIST="${MODEL_LIST},$M"
  fi
done

LOG_FILE="logs/overfit_generalization_gap_per_model.log"

echo "------------------------------------------------------"
echo "SCRIPT : $SCRIPT"
echo "OUT : $OUT_ROOT"
echo "OUT : $OUT_DIR"
echo "MODELS : $MODEL_LIST"
echo "TASKS  : $TASKS"
echo "METHODS: $METHODS"
echo "LOG    : $LOG_FILE"
echo "------------------------------------------------------"

python "$SCRIPT" \
  --out_root "$OUT_ROOT" \
  --models "$MODEL_LIST" \
  --methods "$METHODS" \
  --tasks "$TASKS" \
  > "$LOG_FILE" 2>&1

echo "========================================="
echo "PIPELINE FINISHED"
echo "Log         : /workspace/$LOG_FILE"
echo "Figures     : /workspace/out/reports/figures"
echo "Tables      : /workspace/out/reports/tables"
echo "========================================="