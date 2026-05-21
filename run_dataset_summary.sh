cd /workspace
source .venv/bin/activate
export PYTHONPATH=/workspace
pip install openpyxl
python scripts/summarize_prompt_pools.py --data_root data --out_dir out/tables