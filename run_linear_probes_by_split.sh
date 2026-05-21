#!/bin/bash
set -e

echo "========================================="
echo "Starting RunPod Pipeline"
echo "========================================="

cd /workspace

# ------------------------------
# 1) Create venv if not exists
# ------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -U pip

# ------------------------------
# 2) Install required packages
# ------------------------------
echo "Installing required packages..."
pip install -q torch torchvision torchaudio transformers scikit-learn matplotlib pandas tqdm accelerate

# ------------------------------
# 3) Ensure imports from src work
# ------------------------------
export PYTHONPATH=/workspace

# ------------------------------
# 4) Create log directory
# ------------------------------
mkdir -p logs

# ------------------------------
# 5) Model list
# ------------------------------
MODELS=(
  "EleutherAI/gpt-neo-125M"
  "Qwen/Qwen2.5-0.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "facebook/opt-125m"
  "meta-llama/Llama-3.2-1B"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# ------------------------------
# 6) Run pipeline per model
# ------------------------------
for MODEL in "${MODELS[@]}"; do

  SAFE_NAME=$(echo "$MODEL" | tr '/' '__')
  LOG_FILE="logs/run_${SAFE_NAME}.log"

  echo "======================================================"
  echo ">>> Running pipeline for model: $MODEL"
  echo "Logging to: $LOG_FILE"
  echo "======================================================"

  {
    echo "[MCQ metrics export]"
    python scripts/export_metrics_by_split.py \
      --out_root out \
      --model "$MODEL" \
      --task mcq \
      --methods massmean,lda,logreg,linsvm

    echo "[SINGLE metrics export]"
    python scripts/export_metrics_by_split.py \
      --out_root out \
      --model "$MODEL" \
      --task single \
      --methods massmean,lda,logreg,linsvm

    echo ">>> Done model: $MODEL"

  } > "$LOG_FILE" 2>&1

done

echo "========================================="
echo "ALL MODELS DONE."
echo "========================================="