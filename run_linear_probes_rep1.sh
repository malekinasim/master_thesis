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
METHODS="logreg,linsvm"
BEST_BY="auroc,acc,mean_margin"
VIZ_LAYERS="best,first,mid,last"
PCA_LAYERS=""              # e.g. "best,mid,last" or empty to skip
PCA_COMPONENTS=2
PCA_SAMPLE=4000
PCA_SCATTER=""             # set to "--pca_scatter" if you want
SEED=42


SUMMARY_SCRIPT="scripts/probes_summary.py"       # <-- change to your file name

# 5) models list (must match your folder naming with "__")
MODELS=(
  "Qwen/Qwen2.5-0.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)


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

python scripts/probes_summary.py \
  --task "mcq,single" \
  --models "Qwen/Qwen2.5-0.5B,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  --out_root "/out" \
  --methods "logreg,linsvm" \
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