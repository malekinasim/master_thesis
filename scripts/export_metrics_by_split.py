import os, sys, glob, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.feature_cache import load_feature_cache
from src.linear_probes import fit_eval_probes_per_layer, tune_probes_on_layer
from src.metrics import roc_auc
from src.io import ensure_dir
from src.mass_mean import mass_mean_eval_per_layer_feature, acc_question, mean_margin


def _normalize_df_rows(rows):
    df = pd.DataFrame(rows)
    df["layer"] = df["layer"].astype(int)
    df["split"] = df["split"].astype(str).str.lower()
    return df.sort_values(["split", "layer"])


def _rows_from_scores(split_name, scores_by_layer, y, q):
    rows = []
    for li, s in sorted(scores_by_layer.items()):
        try:
            au = roc_auc(y, s)
        except Exception:
            au = np.nan
        accQ = acc_question(s, y, q)
        mm   = mean_margin(s, y, q)
        rows.append({
            "layer": int(li),
            "split": split_name,
            "acc": float(accQ),
            "auroc": float(au) if np.isfinite(au) else np.nan,
            "mean_margin": float(mm),
        })
    return rows


def _extract_scores_from_res(res_dict):
    scores = {}
    for li, r in res_dict.items():
        s = r.get("scores", None)
        if s is not None:
            scores[int(li)] = np.asarray(s)
    return scores


def main():
    ap = argparse.ArgumentParser("Export per-layer metrics by split (train/validation/test) for probes")
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="HF name, e.g. meta-llama/Llama-3.2-1B OR path form with __")
    ap.add_argument("--task", default="mcq", choices=["mcq", "single"])
    ap.add_argument("--methods", default="logreg,linsvm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.20, help="Validation ratio inside train cache (group-wise)")
    args = ap.parse_args()

    model_path = args.model.replace("/", "__")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    cache_dir  = os.path.join(args.out_root, "features", model_path, args.task)
    tr_npz = Path(os.path.join(cache_dir, "train.npz"))
    te_npz = Path(os.path.join(cache_dir, "test.npz"))
    if not tr_npz.exists() or not te_npz.exists():
        raise FileNotFoundError(f"Missing caches: {tr_npz} / {te_npz}")

    Xtr_layers, ytr, qtr = load_feature_cache(tr_npz)
    Xte_layers, yte, qte = load_feature_cache(te_npz)

    # ---- Split train cache into train_small + validation (grouped by qid) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(gss.split(Xtr_layers[0], ytr, groups=qtr))

    X_train_layers = {li: Xtr_layers[li][train_idx] for li in Xtr_layers}
    y_train = ytr[train_idx]
    q_train = qtr[train_idx]

    X_val_layers = {li: Xtr_layers[li][val_idx] for li in Xtr_layers}
    y_val = ytr[val_idx]
    q_val = qtr[val_idx]

    # ---- Tune hyperparameters on a mid-layer using train_small ----
    mid_li = sorted(X_train_layers.keys())[len(X_train_layers)//2]
    tuned = {}
    for m in methods:
        if m in {"lda", "logreg", "linsvm"}:
            tuned[m] = tune_probes_on_layer(X_train_layers[mid_li], y_train, method=m, random_state=args.seed)
        else:
            tuned[m] = None

    # output dir
    out_dir = os.path.join(args.out_root, "reports", model_path, "lin_probs", "tables", args.task)
    ensure_dir(out_dir)

    # ---- Export per method: train_small / validation / test ----
    for m in methods:
        rows = []

        if m == "massmean":
            # Fit on train_small only
            W = mass_mean_eval_per_layer_feature(X_train_layers, y_train)

            scores_train = {li: (X_train_layers[li] @ w + b) for li, (w, b) in W.items()}
            scores_val   = {li: (X_val_layers[li]   @ w + b) for li, (w, b) in W.items()}
            scores_test  = {li: (Xte_layers[li]     @ w + b) for li, (w, b) in W.items()}

            rows += _rows_from_scores("train",      scores_train, y_train, q_train)
            rows += _rows_from_scores("validation", scores_val,   y_val,   q_val)
            rows += _rows_from_scores("test",       scores_test,  yte,     qte)

        else:
            # Fit on train_small, evaluate on train_small/validation/test
            res_train = fit_eval_probes_per_layer(
                X_train_layers, y_train, X_train_layers, y_train,
                best_params=tuned[m], method=m
            )
            res_val = fit_eval_probes_per_layer(
                X_train_layers, y_train, X_val_layers, y_val,
                best_params=tuned[m], method=m
            )
            res_test = fit_eval_probes_per_layer(
                X_train_layers, y_train, Xte_layers, yte,
                best_params=tuned[m], method=m
            )

            scores_train = _extract_scores_from_res(res_train)
            scores_val   = _extract_scores_from_res(res_val)
            scores_test  = _extract_scores_from_res(res_test)

            rows += _rows_from_scores("train",      scores_train, y_train, q_train)
            rows += _rows_from_scores("validation", scores_val,   y_val,   q_val)
            rows += _rows_from_scores("test",       scores_test,  yte,     qte)

        df = _normalize_df_rows(rows)
        out_csv = os.path.join(out_dir, f"{args.task}_{m}_perlayer_metrics_by_split.csv")
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")


if __name__ == "__main__":
    main()