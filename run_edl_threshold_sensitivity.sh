#!/bin/bash
set -euo pipefail

echo "========================================="
echo "edl_threshold_sensitivity PIPELINE STARTED"
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
INPUTS="table_edl_all_tasks_all_methods_thr0p9_k2.csv,table_edl_all_tasks_all_methods_thr0p85_k2.csv,table_edl_all_tasks_all_methods_thr0p8_k2.csv"
LOG_FILE="logs/edl_threshold_sensitivity.log"

# IMPORTANT: path to your script
SCRIPT="/workspace/scripts/edl_threshold_sensitivity.py"

echo "------------------------------------------------------"
echo "SCRIPT : $SCRIPT"
echo "OUT    : $OUT_ROOT"
echo "INPUTS : $INPUTS"
echo "LOG    : $LOG_FILE"
echo "------------------------------------------------------"
python "$SCRIPT" \
    --csvs "$INPUTS" \
    --out_root "$OUT_ROOT" \
    > "$LOG_FILE" 2>&1
echo "========================================="
echo "PIPELINE FINISHED"
echo "Log         : /workspace/$LOG_FILE"
echo "Figures     : /workspace/out/figures"
echo "Tables      : /workspace/out/tables"
echo "========================================="

