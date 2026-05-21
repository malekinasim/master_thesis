#!/bin/bash
set -u

echo "========================================="
echo "E2E ROBUSTNESS PIPELINE STARTED"
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

# 2) deps (torch nightly for RTX 5090) + common deps
echo "[setup] Installing dependencies..."
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q
pip install -q transformers accelerate tqdm numpy pandas matplotlib scikit-learn

# 3) env
export PYTHONPATH=/workspace

# اگر لازم دارید (مثلاً برای مدل‌های HF که نیاز به توکن دارند)
# export HF_TOKEN="PUT_YOUR_TOKEN_HERE"

# 4) IMPORTANT: set these to your actual script filenames/paths
# اگر این فایل‌ها را داخل repo گذاشته‌اید، مسیر را مطابق آن تغییر دهید
E2E_SCRIPT="/workspace/scripts/e2e_robustnes.py"
PLOT_SCRIPT="/workspace/scripts/plot_e2e_from_csv.py"

# 5) configs
TASKS=("mcq" "single")

# sigmas را همان‌طور که در اسکریپت پایتون هست می‌توانید نگه دارید یا تغییر دهید
SIGMAS="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2"
NOISE_MODE="rel"          # rel یا abs
POSITION="all"            # last | all | window
REPEATS=3
SEED=42
MAX_ITEMS=0               # اگر >0 بگذارید، فقط همان تعداد آیتم برای سرعت (CPU) استفاده می‌شود
MCQ_SCORE="sumlogprob"    # sumlogprob یا firsttoken

# مسیرهای دیتا/خروجی (مطابق default های اسکریپت هم هست)
DATASET_ROOT="/workspace/data"
OUT_ROOT="/workspace/out"

# 6) models list (must match your folder naming with "__" for data layout)
MODELS=(
  "Qwen/Qwen2.5-0.5B"
  "EleutherAI/gpt-neo-125M"
  "facebook/opt-125m"
  "meta-llama/Llama-3.2-1B"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# اگر فقط می‌خواهید CSV تولید شود و رسم نمودارها را بعداً انجام دهید:
# NO_PLOTS="--no_plots"
NO_PLOTS=""

# 7) run per model per task
for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL//\//__}"

  for TASK in "${TASKS[@]}"; do
    LOG_FILE="logs/e2e_${TASK}_${MODEL_PATH}.log"

    echo "------------------------------------------------------"
    echo "MODEL: $MODEL"
    echo "TASK : $TASK"
    echo "LOG  : $LOG_FILE"
    echo "------------------------------------------------------"

    # اجرای محاسبات robustness + ساخت CSV ها (و اگر NO_PLOTS خالی باشد، شکل‌ها را هم می‌سازد)
    python "$E2E_SCRIPT" \
      --model "$MODEL" \
      --task "$TASK" \
      --dataset_root "$DATASET_ROOT" \
      --out_root "$OUT_ROOT" \
      --sigmas "$SIGMAS" \
      --noise_mode "$NOISE_MODE" \
      --position "$POSITION" \
      --repeats "$REPEATS" \
      --seed "$SEED" \
      --mcq_score "$MCQ_SCORE" \
      --max_items "$MAX_ITEMS" \
      $NO_PLOTS \
      > "$LOG_FILE" 2>&1

    if [ $? -ne 0 ]; then
      echo "!!! E2E robustness failed for $MODEL ($TASK). See $LOG_FILE"
      continue
    fi

    # (اختیاری) رسم دوباره از CSV ها (بدون forward) — حتی اگر در مرحله قبل --no_plots زده باشید مفید است
    PLOT_LOG="logs/plot_e2e_${TASK}_${MODEL_PATH}.log"
    python "$PLOT_SCRIPT" \
      --task "$TASK" \
      --model "$MODEL" \
      --out_root "$OUT_ROOT" \
      > "$PLOT_LOG" 2>&1

  done
done

echo "========================================="
echo "PIPELINE FINISHED"
echo "Logs         : /workspace/logs/e2e_*"
echo "Plots logs   : /workspace/logs/plot_e2e_*"
echo "Outputs (CSV): /workspace/out/reports/<MODEL__/...>/robustness/e2e/tables"
echo "Outputs (PNG): /workspace/out/reports/<MODEL__/...>/robustness/e2e/figures"
echo "========================================="