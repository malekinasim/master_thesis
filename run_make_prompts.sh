#!/bin/bash
set -e

echo "========================================="
echo "DATASET GENERATION PIPELINE STARTED"
echo "========================================="

cd /workspace

# ------------------------------
# 1) Setup virtual environment
# ------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -U pip

# ------------------------------
# 2) Install minimal dependencies
# ------------------------------
echo "Installing dependencies..."
pip uninstall -y torch torchvision torchaudio || true
pip cache purge || true
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -q transformers tqdm accelerate
# ------------------------------
# 3) Ensure src imports work
# ------------------------------
export PYTHONPATH=/workspace
export HF_TOKEN="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"


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
# 6) Generate datasets per model
# ------------------------------
for MODEL in "${MODELS[@]}"; do

  SAFE_NAME=$(echo "$MODEL" | tr '/' '__')
  LOG_FILE="logs/dataset_${SAFE_NAME}.log"

  echo "======================================================"
  echo ">>> Generating dataset for: $MODEL"
  echo "Logging to: $LOG_FILE"
  echo "======================================================"

  python scripts/make_prompt_pool.py \
    --model "$MODEL" \
    --n_math 1000 \
    --cap_reps 20 \
    > "$LOG_FILE" 2>&1

  echo ">>> Done dataset for: $MODEL"
  echo

done

echo "========================================="
echo "ALL DATASETS GENERATED."
echo "========================================="