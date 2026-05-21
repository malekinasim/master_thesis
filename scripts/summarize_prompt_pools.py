#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute dataset summary tables from model-specific prompt pools.

Reads:
  data/<MODEL_WITH__>/prompt_pool.json

Writes (CSV):
  - out/prompt_pool_summary.csv                                (Table 8 style)
  - out/tables/table4_prompt_pool_composition_by_model.csv     (Table 4)
  - out/tables/table5_mcq_split_summary_by_model.csv           (Table 5)
  - out/tables/table6_single_split_summary_by_model.csv        (Table 6)
  - out/tables/table7_boolean_vocab_by_model.csv               (Table 7)

Optional (if openpyxl installed):
  - out/tables/dataset_tables.xlsx

Split protocol (group-wise by question id):
  1) test split with test_ratio (default 0.30), seed=42
  2) validation split from remaining train pool with val_ratio (default 0.20), seed=42

Probe-instance accounting:
  - MCQ: instances are (question, option) pairs => sum_k len(options)
         pos = #questions, neg = sum_k (len(options)-1)
  - Single-token: derived instances are (gold, decoy) => 2 * #questions
         pos = #questions, neg = #questions
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# ----------------------------
# Helpers
# ----------------------------

def detect_domain(item: Dict[str, Any]) -> str:
    """Return 'arithmetic' or 'capital' based on question text."""
    q = (item.get("question") or "").lower()
    if "is the capital of" in q:
        return "capital"
    return "arithmetic"


def safe_model_dir_name(model: str) -> str:
    """HF name -> folder name."""
    return model.replace("/", "__")


def hf_name_from_dir(model_dir: str) -> str:
    """Folder name -> HF name (best-effort)."""
    return model_dir.replace("__", "/")


def load_items(prompt_pool_path: Path) -> List[Dict[str, Any]]:
    with prompt_pool_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_rows(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")


def print_pretty_table_table8(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Model",
        "MCQ Arithmetic", "MCQ Capital",
        "Single Arithmetic", "Single Capital",
        "Total",
    ]
    col_widths = [len(h) for h in headers]
    for r in rows:
        values = [
            r["model_label"],
            str(r["mcq_arithmetic"]),
            str(r["mcq_capital"]),
            str(r["single_arithmetic"]),
            str(r["single_capital"]),
            str(r["total"]),
        ]
        col_widths = [max(w, len(v)) for w, v in zip(col_widths, values)]

    def fmt_row(values):
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        values = [
            r["model_label"],
            str(r["mcq_arithmetic"]),
            str(r["mcq_capital"]),
            str(r["single_arithmetic"]),
            str(r["single_capital"]),
            str(r["total"]),
        ]
        print(fmt_row(values))


# ----------------------------
# Table 8: per-model prompt counts
# ----------------------------

def summarize_prompt_pool(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "mcq_arithmetic": 0,
        "mcq_capital": 0,
        "single_arithmetic": 0,
        "single_capital": 0,
        "total": 0,
    }
    for it in items:
        task = (it.get("task") or "").strip().lower()
        if task not in ("mcq", "single"):
            continue
        domain = detect_domain(it)
        key = f"{task}_{domain}"
        counts[key] += 1
        counts["total"] += 1
    return counts


# ----------------------------
# Table 4: composition + derived probe instances
# ----------------------------

def table4_rows_for_model(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # group items by (task, domain)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for it in items:
        task = (it.get("task") or "").strip().lower()
        if task not in ("mcq", "single"):
            continue
        domain = detect_domain(it)
        grouped.setdefault((task, domain), []).append(it)

    order = [("mcq", "arithmetic"), ("mcq", "capital"), ("single", "arithmetic"), ("single", "capital")]
    rows: List[Dict[str, Any]] = []

    for task, domain in order:
        g = grouped.get((task, domain), [])
        if not g:
            continue

        questions = len(g)

        if task == "mcq":
            ks = [len(it.get("options") or []) for it in g]
            uniq = sorted(set(ks))
            if len(uniq) == 1:
                k_desc = f"K={uniq[0]} options"
            else:
                k_desc = "K∈{" + ",".join(map(str, uniq)) + "} options"
            probe_instances = sum(ks)
            task_format = "MCQ"
        else:
            k_desc = "gold + 1 decoy"
            probe_instances = 2 * questions
            task_format = "Single-token"

        domain_label = "Arithmetic (addition)" if domain == "arithmetic" else "Capital statements (boolean)"

        rows.append({
            "Task format": task_format,
            "Domain": domain_label,
            "Questions": questions,
            "K / pairing": k_desc,
            "Probe instances": probe_instances,
        })

    total_questions = sum(r["Questions"] for r in rows)
    total_instances = sum(r["Probe instances"] for r in rows)
    rows.append({
        "Task format": "Total",
        "Domain": "-",
        "Questions": total_questions,
        "K / pairing": "-",
        "Probe instances": total_instances,
    })
    return rows


# ----------------------------
# Group-wise splitting by question id
# ----------------------------

def group_split(items: List[Dict[str, Any]], test_ratio: float, seed: int, group_key: str = "id"
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_items, test_items) by splitting groups."""
    if not items:
        return [], []

    # group ids
    groups = [str(it.get(group_key, "")) for it in items]
    uniq_groups = list(dict.fromkeys(groups))  # preserve first-seen order (deterministic given items)
    rng = random.Random(seed)
    rng.shuffle(uniq_groups)

    n_test = int(round(len(uniq_groups) * test_ratio))
    test_groups = set(uniq_groups[:n_test])

    train_items = [it for it in items if str(it.get(group_key, "")) not in test_groups]
    test_items = [it for it in items if str(it.get(group_key, "")) in test_groups]
    return train_items, test_items


def make_splits(items: List[Dict[str, Any]], seed: int, test_ratio: float, val_ratio: float,
                group_key: str = "id") -> Dict[str, List[Dict[str, Any]]]:
    """Return dict with keys: train_small, validation, test."""
    train_pool, test_items = group_split(items, test_ratio=test_ratio, seed=seed, group_key=group_key)
    train_small, validation = group_split(train_pool, test_ratio=val_ratio, seed=seed, group_key=group_key)
    return {"train_small": train_small, "validation": validation, "test": test_items}


# ----------------------------
# Table 5/6: split summaries
# ----------------------------

def split_summary(items: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    questions = len(items)
    math_q = sum(1 for it in items if detect_domain(it) == "arithmetic")
    cap_q = questions - math_q

    if task == "mcq":
        ks = [len(it.get("options") or []) for it in items]
        probe_instances = sum(ks)
        pos = questions
        neg = sum((k - 1) for k in ks)
    else:
        probe_instances = 2 * questions
        pos = questions
        neg = questions

    pos_pct = (pos / probe_instances * 100.0) if probe_instances > 0 else 0.0

    return {
        "Questions": questions,
        "Math": math_q,
        "Capital": cap_q,
        "Probe instances": probe_instances,
        "Pos (y=1)": pos,
        "Neg (y=0)": neg,
        "Pos %": round(pos_pct, 1),
    }


def table5_or_6_rows_for_model(items: List[Dict[str, Any]], task: str,
                               seed: int, test_ratio: float, val_ratio: float,
                               group_key: str = "id") -> List[Dict[str, Any]]:
    task_items = [it for it in items if (it.get("task") or "").strip().lower() == task]
    splits = make_splits(task_items, seed=seed, test_ratio=test_ratio, val_ratio=val_ratio, group_key=group_key)

    rows: List[Dict[str, Any]] = []
    rows.append({"Split": "Train_small (80% of train)", **split_summary(splits["train_small"], task)})
    rows.append({"Split": "Validation (20% of train)", **split_summary(splits["validation"], task)})
    rows.append({"Split": "Test", **split_summary(splits["test"], task)})
    rows.append({"Split": "Total", **split_summary(task_items, task)})
    return rows


# ----------------------------
# Table 7: boolean vocab (capital domain)
# ----------------------------

def vocab_str(tokens: List[str]) -> str:
    # present without leading/trailing whitespace (for readability in thesis)
    cleaned = sorted({(t or "").strip() for t in tokens if (t or "").strip()})
    return "/".join(cleaned) if cleaned else "-"


def table7_row_for_model(items: List[Dict[str, Any]]) -> Dict[str, str]:
    mcq_opts: List[str] = []
    single_ans: List[str] = []

    for it in items:
        if detect_domain(it) != "capital":
            continue
        task = (it.get("task") or "").strip().lower()
        if task == "mcq":
            mcq_opts.extend(list(it.get("options") or []))
        elif task == "single":
            single_ans.append(it.get("answer") or "")

    return {
        "Capital MCQ options": vocab_str(mcq_opts),
        "Capital Single answers": vocab_str(single_ans),
    }


# ----------------------------
# Optional Excel export
# ----------------------------

def try_write_xlsx(xlsx_path: Path, sheets: Dict[str, Tuple[List[str], List[Dict[str, Any]]]]) -> None:
    try:
        from openpyxl import Workbook
    except Exception:
        print("openpyxl not installed -> skipping XLSX export.")
        return

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    # remove default sheet
    default = wb.active
    wb.remove(default)

    for sheet_name, (headers, rows) in sheets.items():
        ws = wb.create_sheet(title=sheet_name[:31])  # Excel limit
        ws.append(headers)
        for r in rows:
            ws.append([r.get(h, "") for h in headers])

    wb.save(xlsx_path)
    print(f"Saved XLSX: {xlsx_path.resolve()}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="Root folder containing model subfolders")
    parser.add_argument("--out_csv", default="out/prompt_pool_summary.csv",
                        help="Per-model prompt summary CSV (Table 8 style)")
    parser.add_argument("--out_dir", default="out/tables",
                        help="Directory to write Table 4/5/6/7 CSVs")
    parser.add_argument("--xlsx", default="out/tables/dataset_tables.xlsx",
                        help="Optional Excel workbook path (requires openpyxl). Set to '' to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.30)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--group_key", default="id",
                        help="Field used as group id for splitting (default: id)")
    parser.add_argument("--models", nargs="*", default=None,
                        help='Optional list of HF model names (e.g., "EleutherAI/gpt-neo-125M"). '
                             "If omitted, all model folders under data_root are scanned.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to load
    models_to_load: List[Tuple[str, str, Path]] = []
    # tuple: (model_label, model_dir_name, prompt_pool_path)

    if args.models:
        for hf_name in args.models:
            model_dir_name = safe_model_dir_name(hf_name)
            prompt_path = data_root / model_dir_name / "prompt_pool.json"
            models_to_load.append((hf_name, model_dir_name, prompt_path))
    else:
        for model_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            prompt_path = model_dir / "prompt_pool.json"
            if not prompt_path.exists():
                continue
            model_dir_name = model_dir.name
            hf_name = hf_name_from_dir(model_dir_name)
            # model_label: keep directory name (matches your printed table)
            models_to_load.append((model_dir_name, model_dir_name, prompt_path))

    # Load all prompt pools
    model_items: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for model_label, model_dir_name, prompt_path in models_to_load:
        if not prompt_path.exists():
            continue
        items = load_items(prompt_path)
        model_items.append((model_label, model_dir_name, items))

    if not model_items:
        raise SystemExit(f"No prompt pools found under {data_root.resolve()}")

    # ---------- Table 8 (prompt summary) ----------
    table8_rows: List[Dict[str, Any]] = []
    for model_label, _, items in model_items:
        counts = summarize_prompt_pool(items)
        table8_rows.append({"model_label": model_label, **counts})

    table8_rows.sort(key=lambda r: r["model_label"])

    print_pretty_table_table8(table8_rows)

    # write CSV (Table 8)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers8 = ["model_label", "mcq_arithmetic", "mcq_capital", "single_arithmetic", "single_capital", "total"]
    write_csv_rows(out_csv, headers8, table8_rows)
    print(f"\nSaved CSV: {out_csv.resolve()}")

    # ---------- Table 4 ----------
    table4_rows: List[Dict[str, Any]] = []
    for model_label, _, items in model_items:
        rows = table4_rows_for_model(items)
        for r in rows:
            table4_rows.append({"Model": model_label, **r})

    headers4 = ["Model", "Task format", "Domain", "Questions", "K / pairing", "Probe instances"]
    write_csv_rows(out_dir / "table4_prompt_pool_composition_by_model.csv", headers4, table4_rows)

    # ---------- Table 5 (MCQ splits) ----------
    table5_rows: List[Dict[str, Any]] = []
    for model_label, _, items in model_items:
        rows = table5_or_6_rows_for_model(items, task="mcq",
                                          seed=args.seed, test_ratio=args.test_ratio, val_ratio=args.val_ratio,
                                          group_key=args.group_key)
        for r in rows:
            table5_rows.append({"Model": model_label, **r})

    headers5 = ["Model", "Split", "Questions", "Math", "Capital", "Probe instances", "Pos (y=1)", "Neg (y=0)", "Pos %"]
    write_csv_rows(out_dir / "table5_mcq_split_summary_by_model.csv", headers5, table5_rows)

    # ---------- Table 6 (Single splits) ----------
    table6_rows: List[Dict[str, Any]] = []
    for model_label, _, items in model_items:
        rows = table5_or_6_rows_for_model(items, task="single",
                                          seed=args.seed, test_ratio=args.test_ratio, val_ratio=args.val_ratio,
                                          group_key=args.group_key)
        for r in rows:
            table6_rows.append({"Model": model_label, **r})

    headers6 = headers5
    write_csv_rows(out_dir / "table6_single_split_summary_by_model.csv", headers6, table6_rows)

    # ---------- Table 7 (Boolean vocab) ----------
    table7_rows: List[Dict[str, Any]] = []
    for model_label, _, items in model_items:
        vocab_row = table7_row_for_model(items)
        table7_rows.append({"Model": model_label, **vocab_row})

    headers7 = ["Model", "Capital MCQ options", "Capital Single answers"]
    write_csv_rows(out_dir / "table7_boolean_vocab_by_model.csv", headers7, table7_rows)

    print(f"Saved Table 4/5/6/7 CSVs to: {out_dir.resolve()}")

    # ---------- Optional XLSX ----------
    if args.xlsx and args.xlsx.strip():
        sheets = {
            "Table8_PerModelCounts": (headers8, table8_rows),
            "Table4_Composition": (headers4, table4_rows),
            "Table5_MCQ_Splits": (headers5, table5_rows),
            "Table6_Single_Splits": (headers6, table6_rows),
            "Table7_BoolVocab": (headers7, table7_rows),
        }
        try_write_xlsx(Path(args.xlsx), sheets)


if __name__ == "__main__":
    main()