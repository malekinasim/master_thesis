#!/bin/bash
set -u

echo "========================================="
echo "FEATURE EXTRACTION (NO DATASET BUILD)"
echo "========================================="

cd /workspace
mkdir -p logs out

# ------------------------------
# 1) venv
# ------------------------------
if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip -q

# ------------------------------
# 2) deps (RTX 5090 -> torch nightly cu128)
# ------------------------------
echo "[setup] Installing dependencies..."
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q
pip install -q transformers accelerate tqdm numpy

# ------------------------------
# 3) env
# ------------------------------
export PYTHONPATH=/workspace

# set token 
export HF_TOKEN="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"

# ------------------------------
# 4) models
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
# 5) params
# ------------------------------
TEST_RATIO=0.3
SEED=42
COMPUTE_DTYPE="auto"
STORE_DTYPE="float32"
TOKEN_POS="-1"   # e.g. -1 or comp:0

# ------------------------------
# 6) run extraction
# ------------------------------
for MODEL in "${MODELS[@]}"; do
  SAFE_NAME="${MODEL//\//__}"
  DATASET_PATH="data/${SAFE_NAME}/prompt_pool.json"

  MCQ_LOG="logs/extract_mcq_${SAFE_NAME}.log"
  SINGLE_LOG="logs/extract_single_${SAFE_NAME}.log"

  echo "======================================================"
  echo ">>> MODEL: $MODEL"
  echo "Dataset: $DATASET_PATH"
  echo "======================================================"

  if [ ! -f "$DATASET_PATH" ]; then
    echo "!!! Dataset not found for $MODEL -> skipping."
    continue
  fi

  echo "[step] MCQ feature extraction..."
  python scripts/extract_features.py \
    --task mcq \
    --model "$MODEL" \
    --dataset "$DATASET_PATH" \
    --test_ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --out_root "/workspace/out" \
    --remote false \
    --compute_dtype "$COMPUTE_DTYPE" \
    --store_dtype "$STORE_DTYPE" \
    --token_pos "$TOKEN_POS" \
    > "$MCQ_LOG" 2>&1

  if [ $? -ne 0 ]; then
    echo "!!! MCQ failed for $MODEL (see $MCQ_LOG). Continuing..."
  fi

  echo "[step] SINGLE feature extraction..."
  python scripts/extract_features.py \
    --task single \
    --model "$MODEL" \
    --dataset "$DATASET_PATH" \
    --test_ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --out_root "/workspace/out" \
    --remote false \
    --compute_dtype "$COMPUTE_DTYPE" \
    --store_dtype "$STORE_DTYPE" \
    --token_pos "$TOKEN_POS" \
    > "$SINGLE_LOG" 2>&1

  if [ $? -ne 0 ]; then
    echo "!!! SINGLE failed for $MODEL (see $SINGLE_LOG)."
  fi

  echo ">>> DONE: $MODEL"
  echo
done

echo "========================================="
echo "FINISHED"
echo "Logs: /workspace/logs"
echo "Out:  /workspace/out/features"
echo "========================================="