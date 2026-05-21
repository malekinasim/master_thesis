#!/bin/bash
set -u

echo "========================================="
echo "LINEAR PROBES PIPELINE STARTED"
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

# 2) deps (torch nightly for RTX 5090) + sklearn/pandas/matplotlib
echo "[setup] Installing dependencies..."
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q
pip install -q transformers accelerate tqdm numpy pandas matplotlib scikit-learn

# 3) env
export PYTHONPATH=/workspace
export HF_TOKEN="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"

# 4) configs
METHODS="massmean,lda,logreg,linsvm"
BEST_BY="auroc,acc,mean_margin,fisher"
VIZ_LAYERS="best,first,mid,last"
PCA_LAYERS=""              # e.g. "best,mid,last" or empty to skip
PCA_COMPONENTS=2
PCA_SAMPLE=4000
PCA_SCATTER=""             # set to "--pca_scatter" if you want
SEED=42

# IMPORTANT: set these to your actual script filenames
PHASE2_SCRIPT="scripts/run_linear_probes.py"     # <-- change to your file name
SUMMARY_SCRIPT="scripts/probes_summary.py"       # <-- change to your file name

# 5) models list (must match your folder naming with "__")
MODELS=(
  "EleutherAI/gpt-neo-125M"
  "Qwen/Qwen2.5-0.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "facebook/opt-125m"
  "meta-llama/Llama-3.2-1B"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# 6) run Phase-2 per model per task
for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL//\//__}"

  for TASK in mcq single; do
    CACHE_DIR="/workspace/out/features/${MODEL_PATH}/${TASK}"
    TR_NPZ="${CACHE_DIR}/train.npz"
    TE_NPZ="${CACHE_DIR}/test.npz"
    LOG_FILE="logs/linprobes_${TASK}_${MODEL_PATH}.log"

    echo "------------------------------------------------------"
    echo "MODEL: $MODEL"
    echo "TASK : $TASK"
    echo "CACHE: $CACHE_DIR"
    echo "LOG  : $LOG_FILE"
    echo "------------------------------------------------------"

    # if features are missing, skip (no crash)
    if [ ! -f "$TR_NPZ" ] || [ ! -f "$TE_NPZ" ]; then
      echo "!!! Missing feature cache for $MODEL_PATH/$TASK -> skipping." | tee -a "$LOG_FILE"
      continue
    fi

    # run phase-2 (linear probes)
    python "$PHASE2_SCRIPT" \
      --task "$TASK" \
      --model "$MODEL" \
      --out_root "/workspace/out" \
      --methods "$METHODS" \
      --viz_layers "$VIZ_LAYERS" \
      --pca_layers "$PCA_LAYERS" \
      --pca_components "$PCA_COMPONENTS" \
      --pca_sample "$PCA_SAMPLE" \
      $PCA_SCATTER \
      --seed "$SEED" \
      --best_by "$BEST_BY" \
      > "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
      echo "!!! Phase-2 failed for $MODEL ($TASK). See $LOG_FILE"
      # continue to next task/model
      continue
    fi
  done
done

echo "========================================="
echo "PHASE-2 DONE. Now running SUMMARY script"
echo "========================================="

# 7) run summary/plots across models
# build model list in "__" format for the summary script input
MODEL_LIST=""
for MODEL in "${MODELS[@]}"; do
  MP="${MODEL//\//__}"
  if [ -z "$MODEL_LIST" ]; then
    MODEL_LIST="$MP"
  else
    MODEL_LIST="${MODEL_LIST},$MP"
  fi
done

SUMMARY_LOG="logs/linprobes_summary.log"

python "$SUMMARY_SCRIPT" \
  --task "mcq,single" \
  --models "$MODEL_LIST" \
  --out_root "/workspace/out" \
  --methods "$METHODS" \
  --plots \
  --plot_metrics "auroc,acc,mean_margin" \
  --normalize_depth \
  --family_compare \
  --family_metrics "auroc,acc,mean_margin" \
  --family_grid_n 101 \
  --edl_k 2 \
  > "$SUMMARY_LOG" 2>&1

echo "========================================="
echo "PIPELINE FINISHED"
echo "Phase-2 logs: /workspace/logs/linprobes_*"
echo "Summary log : $SUMMARY_LOG"
echo "Outputs     : /workspace/out/reports"
echo "========================================="