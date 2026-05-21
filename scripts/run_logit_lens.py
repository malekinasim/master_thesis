# scripts/run_logit_lens.py
# Layer-wise logit lens for MCQ and Single-token prompts (with optional Tuned Lens)
import sys
from pathlib import Path
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- make 'src' importable when running as a script ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.util import load_model_and_tokenizer
from src.io import (
    load_prompts_with_options,
    ensure_dir,
    save_CSV_layers_MCQ_Margins,
    save_csv_margins,
)
from src.tuned import TunedDiag
# logit-lens core + EDL + margins
from src.logit_lens import mcq_alllayer_scores, early_decision_layer, compute_margins_per_layer_logits


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def pick_item(items, idx: int | None):
    """Robust pick: if idx is None -> first; if out of range -> random; else -> that index."""
    if not items:
        return None
    n = len(items)
    if idx is None:
        return items[0]
    if 0 <= idx < n:
        return items[idx]
    return items[random.randint(0, n - 1)]

def overlay_edl_line(ax: plt.Axes, edl_idx: int | None, color="red", label="EDL"):
    """Draw a vertical line at early-decision layer index (if provided)."""
    if edl_idx is None or edl_idx < 0:
        return
    ax.axvline(x=edl_idx, color=color, linestyle="--", alpha=0.8, label=label)

def save_mcq_plot_with_edl(res, out_png: str, title: str, edl_idx: int | None):
    """
    Plot MCQ margins per layer (raw/tuned) and overlay EDL vertical line.
    """
    raw_scores, _, raw_m12, raw_gold = res["raw"]
    tuned_part = res["tuned"]
    tuned_m12, tuned_gold = None, None
    if tuned_part is not None:
        _, _, tuned_m12, tuned_gold = tuned_part

    L = len(raw_m12)
    layers = list(range(L))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(layers, raw_m12, label="top1-top2 (raw)")
    if raw_gold and any(v is not None for v in raw_gold):
        ax.plot(layers, [np.nan if v is None else v for v in raw_gold], label="gold-margin (raw)")
    if tuned_m12 is not None:
        ax.plot(layers, tuned_m12, label="top1-top2 (tuned)")
        if tuned_gold and any(v is not None for v in tuned_gold):
            ax.plot(layers, [np.nan if v is None else v for v in tuned_gold], label="gold-margin (tuned)")

    overlay_edl_line(ax, edl_idx, color="red", label="EDL")
    ax.set_xlabel("Layer index (0 = first after embedding)")
    ax.set_ylabel("Margin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_single_plot_with_edl(Z_full, out_png: str, title: str, edl_idx: int | None):
    """
    Plot Single-task margins per layer (raw/tuned) and overlay EDL vertical line.
    """
    raw = Z_full["full"]["raw"]
    tuned = Z_full["full"].get("tuned", None)
    top1_full = raw.get("top1_top2_full", [])
    gold_full = raw.get("gold_full", None)
    L = len(top1_full)
    layers = list(range(L))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(layers, top1_full, label="top1-top2 full (raw)")
    if gold_full:
        ax.plot(layers, gold_full, label="gold full (raw)")
    if tuned:
        t_top1 = tuned.get("top1_top2_full", None)
        if t_top1:
            ax.plot(layers, t_top1, label="top1-top2 full (tuned)")
        t_gold = tuned.get("gold_full", None)
        if t_gold:
            ax.plot(layers, t_gold, label="gold full (tuned)")

    overlay_edl_line(ax, edl_idx, color="red", label="EDL")
    ax.set_xlabel("Layer index (0 = first after embedding)")
    ax.set_ylabel("Margin")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Logit Lens (MCQ/Single) with optional Tuned Lens + EDL overlay")

    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M", help="HF model name or local path")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" ))
    parser.add_argument("--task", choices=["mcq", "single"], required=True,default="single")
    parser.add_argument("--mcq_idx", type=int, default=None, help="Index of MCQ item (if None: first)")
    parser.add_argument("--single_idx", type=int, default=None, help="Index of Single item (if None: first)")
    parser.add_argument("--margin_thresh", type=float, default=0.0, help="EDL margin threshold")
    parser.add_argument("--persist_k", type=int, default=2, help="EDL persistence window length")
    parser.add_argument("--ln_f_mode", choices=["none", "last_only", "all"], default="all")
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        default=True,
        help="If set, skip embedding state; first layer is the first transformer block",
    )
    parser.add_argument("--remote", default=False)

    # NEW: dataset-level aggregation without losing per-item outputs
    parser.add_argument("--aggregate", action="store_true",default=True,
                        help="Also compute dataset-level aggregates (keeps per-item plots).")
    parser.add_argument("--save_each", default=False,
                        help="When --aggregate is set, also save per-item plots/CSVs.")

    args = parser.parse_args()

    # Load model/tokenizer
    model, tok = load_model_and_tokenizer(args.model, remote=args.remote)

    # Output folders
    model_path = args.model.replace("/", "__")
    tables_dir = os.path.join(REPO_ROOT,"out", "reports", model_path,"logit", "tables")
    figs_dir = os.path.join(REPO_ROOT,"out", "reports", model_path,"logit", "figures")
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    # Load items
    dataset=os.path.join(args.dataset, model_path, "prompt_pool.json")
    mcq_items, single_items = load_prompts_with_options(dataset, tok, require_single_token=False)

    # Tuned lens (optional)
    tuned_path= os.path.join(REPO_ROOT ,"data",model_path,"tuned_diag.json")
    
    tuned = TunedDiag.from_json(tuned_path, device=next(model.parameters()).device) if os.path.exists(tuned_path) else None

    # ---------------- MCQ ----------------
    if args.task == "mcq":
        valid_mcq = [
            it
            for it in mcq_items
            if isinstance(it.get("question"), str)
            and isinstance(it.get("options"), list)
            and len(it["options"]) >= 2
            and it.get("answer") in it["options"]
        ]
        if not valid_mcq:
            print("[LogitLens-MCQ] No valid MCQ items.")
            return

        # Single-item mode 
        if not args.aggregate:
            item = pick_item(valid_mcq, args.mcq_idx)
            q, options, gold = item["question"], item["options"], item.get("answer")
            gold=gold.strip()

            res = mcq_alllayer_scores(
                model=model,
                tokenizer=tok,
                prompt_text=q,
                options=options,
                gold_opt=gold.strip(),
                pos=-1,
                ln_f_mode=args.ln_f_mode,
                skip_embedding=args.skip_embedding,
                tuned=tuned,
            )
            ed = early_decision_layer(
                res,
                margin_thresh=args.margin_thresh,
                use_tuned=bool(res.get("tuned")),
                use_gold=False,
                persist_k=args.persist_k,
            )

            base = f"mcq_{item.get('id', 'item')}"
            save_CSV_layers_MCQ_Margins(res, options=options, out_dir=tables_dir, fname=f"{base}__layer_margins.csv")
            out_png = os.path.join(figs_dir, f"{base}__layer_margins.png")
            save_mcq_plot_with_edl(
                res,
                out_png=out_png,
                title=f"MCQ margins per layer ({'raw + tuned' if res.get('tuned') else 'raw'})",
                edl_idx=(ed["idx"] if ed else None),
            )
            print(f"[SAVE] {out_png}")
            return

        # Aggregate mode over full dataset (MCQ)
        L_ref = None
        raw_top12_sum = raw_win_cnt = raw_gold_sum = raw_gold_cnt = None
        t_top12_sum = t_win_cnt = t_gold_sum = t_gold_cnt = None
        edl_list = []

        for it in valid_mcq:
            q, options, gold = it["question"], it["options"], it["answer"]
            res = mcq_alllayer_scores(
                model=model,
                tokenizer=tok,
                prompt_text=q,
                options=options,
                gold_opt=gold,
                pos=-1,
                ln_f_mode=args.ln_f_mode,
                skip_embedding=args.skip_embedding,
                tuned=tuned,
            )

            raw_scores, raw_winners, raw_m12, raw_gold = res["raw"]
            L = len(raw_m12)
            if L_ref is None:
                L_ref = L
                raw_top12_sum = np.zeros(L)
                raw_win_cnt = np.zeros(L)
                raw_gold_sum = np.zeros(L)
                raw_gold_cnt = np.zeros(L)
                if res.get("tuned"):
                    t_top12_sum = np.zeros(L)
                    t_win_cnt = np.zeros(L)
                    t_gold_sum = np.zeros(L)
                    t_gold_cnt = np.zeros(L)

            # Aggregate RAW
            for i in range(L):
                m12 = raw_m12[i]
                if m12 is not None and not np.isnan(m12):
                    raw_top12_sum[i] += m12
                if raw_winners[i] == gold:
                    raw_win_cnt[i] += 1
                if raw_gold and i < len(raw_gold):
                    gm = raw_gold[i]
                    if gm is not None and not np.isnan(gm):
                        raw_gold_sum[i] += gm
                        raw_gold_cnt[i] += 1

            # Aggregate TUNED
            if res.get("tuned"):
                t_scores, t_winners, t_m12, t_gold = res["tuned"]
                for i in range(L):
                    tm = t_m12[i]
                    if tm is not None and not np.isnan(tm):
                        t_top12_sum[i] += tm
                    if t_winners[i] == gold:
                        t_win_cnt[i] += 1
                    if t_gold and i < len(t_gold):
                        gm = t_gold[i]
                        if gm is not None and not np.isnan(gm):
                            t_gold_sum[i] += gm
                            t_gold_cnt[i] += 1

            # EDL (per item)
            ed = early_decision_layer(
                res,
                margin_thresh=args.margin_thresh,
                use_tuned=bool(res.get("tuned")),
                use_gold=False,
                persist_k=args.persist_k,
            )
            if ed and ("idx" in ed):
                edl_list.append(ed["idx"])

            # Keep per-item outputs if requested
            if args.save_each:
                base = f"mcq_{it.get('id', 'item')}"
                save_CSV_layers_MCQ_Margins(res, options=options, out_dir=tables_dir, fname=f"{base}__layer_margins.csv")
                out_png = os.path.join(figs_dir, f"{base}__layer_margins.png")
                save_mcq_plot_with_edl(
                    res,
                    out_png=out_png,
                    title=f"MCQ margins per layer ({'raw + tuned' if res.get('tuned') else 'raw'})",
                    edl_idx=(ed["idx"] if ed else None),
                )

        # Final aggregates
        N = len(valid_mcq)
        raw_top12_mean = raw_top12_sum / max(N, 1)
        raw_win_rate = raw_win_cnt / max(N, 1)
        raw_gold_mean = np.divide(raw_gold_sum, np.maximum(raw_gold_cnt, 1))

        if t_top12_sum is not None:
            t_top12_mean = t_top12_sum / max(N, 1)
            t_win_rate = t_win_cnt / max(N, 1)
            t_gold_mean = np.divide(t_gold_sum, np.maximum(t_gold_cnt, 1))
        else:
            t_top12_mean = t_win_rate = t_gold_mean = None

        # Save aggregate CSV
        rows = []
        for i in range(L_ref):
            rows.append(
                {
                    "layer": i,
                    "raw_top1_top2_mean": float(raw_top12_mean[i]),
                    "raw_win_rate": float(raw_win_rate[i]),
                    "raw_gold_margin_mean": (float(raw_gold_mean[i]) if raw_gold_cnt[i] > 0 else np.nan),
                    "tuned_top1_top2_mean": (float(t_top12_mean[i]) if t_top12_mean is not None else np.nan),
                    "tuned_win_rate": (float(t_win_rate[i]) if t_win_rate is not None else np.nan),
                    "tuned_gold_margin_mean": (float(t_gold_mean[i]) if t_gold_mean is not None else np.nan),
                }
            )
        agg_csv = os.path.join(tables_dir, "mcq__dataset_aggregate_layers.csv")
        pd.DataFrame(rows).to_csv(agg_csv, index=False)
        print("[SAVE]", agg_csv)

        # Aggregate plots
        plt.figure(figsize=(10, 4))
        plt.plot(range(L_ref), raw_top12_mean, label="RAW mean Top1-Top2")
        if t_top12_mean is not None:
            plt.plot(range(L_ref), t_top12_mean, label="TUNED mean Top1-Top2")
        plt.xlabel("layer")
        plt.ylabel("mean margin")
        plt.title("Dataset-level mean margins per layer (MCQ)")
        plt.legend()
        plt.grid(alpha=0.3)
        out_png = os.path.join(figs_dir, "mcq__dataset_margins.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        print("[SAVE]", out_png)

        if edl_list:
            plt.figure(figsize=(6, 4))
            plt.hist(edl_list, bins=min(30, L_ref))
            plt.xlabel("EDL index")
            plt.ylabel("count")
            plt.title("EDL distribution across dataset (MCQ)")
            out_png2 = os.path.join(figs_dir, "mcq__dataset_edl_hist.png")
            plt.tight_layout()
            plt.savefig(out_png2, dpi=180)
            plt.close()
            print("[SAVE]", out_png2)
        return

    # ---------------- Single ----------------
    if args.task == "single":
        valid_single = [it for it in single_items if isinstance(it.get("question"), str) and isinstance(it.get("answer"), str)]
        if not valid_single:
            print("[LogitLens-Single] No single items.")
            return

        # Single-item mode (keep old behavior)
        if not args.aggregate:
            item = pick_item(valid_single, args.single_idx)
            q, gold_text = item["question"], item["answer"]

            Z_full = compute_margins_per_layer_logits(
                model,
                tok,
                text=q,
                outputs=None,
                pos=-1,
                ln_f_mode=args.ln_f_mode,
                skip_embedding=args.skip_embedding,
                gold_text=gold_text,
                options=None,
                gold_option=None,
                tuned=tuned,
            )

            # EDL based on gold_full (if available) else top1_top2_full
            raw_full = Z_full["full"]["raw"]
            gold_full = raw_full.get("gold_full", None)
            ed_idx = None
            if gold_full:
                L = len(gold_full)
                for i in range(L):
                    end = min(i + args.persist_k, L)
                    window = gold_full[i:end]
                    if all((m is not None) and (m >= args.margin_thresh) for m in window):
                        ed_idx = i
                        break

            base = f"single_{item.get('id', 'item')}"
            save_csv_margins(Z_full, out_dir=tables_dir, fname=f"{base}__layer_margins.csv")
            out_png = os.path.join(figs_dir, f"{base}__layer_margins.png")
            save_single_plot_with_edl(
                Z_full,
                out_png=out_png,
                title=f"Single-token margins per layer ({'raw + tuned' if Z_full['full'].get('tuned') else 'raw'})",
                edl_idx=ed_idx,
            )
            print(f"[SAVE] {out_png}")
            return

        # Aggregate mode over full dataset (Single)
        L_ref = None
        raw_top12_sum = raw_gold_sum = raw_gold_cnt = None
        t_top12_sum = t_gold_sum = t_gold_cnt = None
        edl_list = []

        for it in valid_single:
            q, gold_text = it["question"], it["answer"]
            Z_full = compute_margins_per_layer_logits(
                model,
                tok,
                text=q,
                outputs=None,
                pos=-1,
                ln_f_mode=args.ln_f_mode,
                skip_embedding=args.skip_embedding,
                gold_text=gold_text,
                options=None,
                gold_option=None,
                tuned=tuned,
            )

            raw = Z_full["full"]["raw"]
            top1_full = raw.get("top1_top2_full", [])
            gold_full = raw.get("gold_full", None)
            L = len(top1_full)
            if L_ref is None:
                L_ref = L
                raw_top12_sum = np.zeros(L)
                raw_gold_sum = np.zeros(L)
                raw_gold_cnt = np.zeros(L)
                if Z_full["full"].get("tuned"):
                    t_full = Z_full["full"]["tuned"]
                    t_top12_sum = np.zeros(L)
                    t_gold_sum = np.zeros(L)
                    t_gold_cnt = np.zeros(L)

            # Aggregate RAW
            for i in range(L):
                m12 = top1_full[i] if i < len(top1_full) else None
                if m12 is not None and not np.isnan(m12):
                    raw_top12_sum[i] += m12
                if gold_full and i < len(gold_full):
                    gm = gold_full[i]
                    if gm is not None and not np.isnan(gm):
                        raw_gold_sum[i] += gm
                        raw_gold_cnt[i] += 1

            # Aggregate TUNED
            t_rec = Z_full["full"].get("tuned", None)
            if t_rec:
                t_top = t_rec.get("top1_top2_full", [])
                t_gold = t_rec.get("gold_full", None)
                for i in range(L):
                    tm = t_top[i] if i < len(t_top) else None
                    if tm is not None and not np.isnan(tm):
                        t_top12_sum[i] += tm
                    if t_gold and i < len(t_gold):
                        gm = t_gold[i]
                        if gm is not None and not np.isnan(gm):
                            t_gold_sum[i] += gm
                            t_gold_cnt[i] += 1

            # EDL per item (gold_full-based if available)
            ed_idx = None
            if gold_full:
                for i in range(L):
                    end = min(i + args.persist_k, L)
                    window = gold_full[i:end]
                    if all((m is not None) and (m >= args.margin_thresh) for m in window):
                        ed_idx = i
                        break
            if ed_idx is not None:
                edl_list.append(ed_idx)

            # Keep per-item outputs if requested
            if args.save_each:
                base = f"single_{it.get('id', 'item')}"
                save_csv_margins(Z_full, out_dir=tables_dir, fname=f"{base}__layer_margins.csv")
                out_png = os.path.join(figs_dir, f"{base}__layer_margins.png")
                save_single_plot_with_edl(
                    Z_full,
                    out_png=out_png,
                    title=f"Single-token margins per layer ({'raw + tuned' if Z_full['full'].get('tuned') else 'raw'})",
                    edl_idx=ed_idx,
                )

        # Final aggregates
        N = len(valid_single)
        raw_top12_mean = raw_top12_sum / max(N, 1)
        raw_gold_mean = np.divide(raw_gold_sum, np.maximum(raw_gold_cnt, 1))

        if t_top12_sum is not None:
            t_top12_mean = t_top12_sum / max(N, 1)
            t_gold_mean = np.divide(t_gold_sum, np.maximum(t_gold_cnt, 1))
        else:
            t_top12_mean = t_gold_mean = None

        # Save aggregate CSV
        rows = []
        for i in range(L_ref):
            rows.append(
                {
                    "layer": i,
                    "raw_top1_top2_mean": float(raw_top12_mean[i]),
                    "raw_gold_margin_mean": (float(raw_gold_mean[i]) if raw_gold_cnt[i] > 0 else np.nan),
                    "tuned_top1_top2_mean": (float(t_top12_mean[i]) if t_top12_mean is not None else np.nan),
                    "tuned_gold_margin_mean": (float(t_gold_mean[i]) if t_gold_mean is not None else np.nan),
                }
            )
        agg_csv = os.path.join(tables_dir, "single__dataset_aggregate_layers.csv")
        pd.DataFrame(rows).to_csv(agg_csv, index=False)
        print("[SAVE]", agg_csv)

        # Aggregate plots
        plt.figure(figsize=(10, 4))
        plt.plot(range(L_ref), raw_top12_mean, label="RAW mean Top1-Top2")
        if t_top12_mean is not None:
            plt.plot(range(L_ref), t_top12_mean, label="TUNED mean Top1-Top2")
        plt.xlabel("layer")
        plt.ylabel("mean margin")
        plt.title("Dataset-level mean margins per layer (Single)")
        plt.legend()
        plt.grid(alpha=0.3)
        out_png = os.path.join(figs_dir, "single__dataset_margins.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        print("[SAVE]", out_png)

        if edl_list:
            plt.figure(figsize=(6, 4))
            plt.hist(edl_list, bins=min(30, L_ref))
            plt.xlabel("EDL index")
            plt.ylabel("count")
            plt.title("EDL distribution across dataset (Single)")
            out_png2 = os.path.join(figs_dir, "single__dataset_edl_hist.png")
            plt.tight_layout()
            plt.savefig(out_png2, dpi=180)
            plt.close()
            print("[SAVE]", out_png2)
        return

    raise ValueError("Invalid task")


if __name__ == "__main__":
    main()
