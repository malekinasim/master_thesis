import sys, os, argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit  # اضافه شد: برای Split کردن گروهی
from sklearn.decomposition import PCA
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.metrics import roc_auc
from src.feature_cache import load_feature_cache
from src.io import ensure_dir
from src.linear_probes import fit_eval_probes_per_layer, tune_probes_on_layer
from src.viz import (
    plot_combined_diagnostics,
    plot_layer_acc_curve,
    plot_pca_variance_curve,
    plot_pca_scatter_2d,
)
from src.mass_mean import (
    mass_mean_eval_per_layer_feature, acc_question, mean_margin
)

def _pick_best_layer(res_dict: dict, metric: str) -> tuple[int, dict[int, float]]:
    """
    res_dict: {layer_idx -> {'acc':..., 'auroc':..., 'mean_margin':..., 'fisher':...}}
    metric:   one of {'auroc','acc','mean_margin','fisher'}
    returns: (best_li, values_per_layer)
    """
    vals = {}
    for li, r in res_dict.items():
        if metric == "acc":
            vals[li] = r.get("acc", float("-inf"))
        elif metric == "mean_margin":
            vals[li] = r.get("mean_margin", float("-inf"))
        elif metric == "fisher":
            vals[li] = r.get("fisher", float("-inf"))
        else:
            vals[li] = r.get("auroc", float("-inf"))  # default to auroc
    # در صورت مساوی بودن مقدار، لایه عمیق‌تر ترجیح داده شود
    best_li = max(vals.keys(), key=lambda k: (vals[k], k))
    return int(best_li), vals

def _spec_layers(spec, L_total, best_map=None, available_layers=None):
    """
    spec: 'best,first,mid,last,10,...'
    best_map: dict[method] -> int | Iterable[int]   (شامل BESTهای هر روش)
    """
    tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
    valid = set(range(int(L_total))) if available_layers is None else {int(li) for li in available_layers}
    tag2li = {"first": 0, "mid": int(L_total)//2, "last": int(L_total)-1}

    out = set()
    for t in tokens:
        if t in tag2li:
            li = tag2li[t]
            if li in valid: 
                out.add(li)
        elif t == "best" and best_map:
            for m, v in best_map.items():
                if isinstance(v, (list, tuple, set)):
                    for li in v:
                        li = int(li)
                        if li in valid: 
                            out.add(li)
                else:
                    li = int(v)
                    if li in valid: 
                        out.add(li)
        else:
            try:
                li = int(t)
                if li in valid: 
                    out.add(li)
            except ValueError:
                continue
    return sorted(int(li) for li in out)

def main():
    ap = argparse.ArgumentParser("Phase-2: run probes & plots from cached features")
    ap.add_argument("--task", choices=["mcq","single"], required=True)
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--methods", default="lda,logreg,linsvm",
                    help="comma list among: massmean,lda,logreg,linsvm")
    ap.add_argument("--viz_layers", default="best,first,mid,last,",
                    help="e.g. 'best,first,mid,last,10,22'")
    ap.add_argument("--pca_layers", default="",
                    help="layers to run PCA variance ratios on (same syntax as viz_layers); empty to skip")
    ap.add_argument("--pca_components", type=int, default=2)
    ap.add_argument("--pca_sample", type=int, default=4000,
                    help="max samples per layer for PCA (downsample for speed)")
    ap.add_argument("--pca_scatter", action="store_true",
                    help="if set, save 2D PCA scatter (labels-colored) for selected pca_layers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--best_by", default="auroc",
                    help="Comma list of metrics to pick BEST layers by (e.g., 'auroc,acc,mean_margin,fisher')")
    args = ap.parse_args()

    model_path = args.model.replace("/", "__")
    cache_dir  = os.path.join(args.out_root, "features", model_path, args.task)
    tables_dir = os.path.join(args.out_root, "reports", model_path,"lin_probs", "tables", args.task)
    figs_dir   = os.path.join(args.out_root, "reports", model_path, "lin_probs", "figures", args.task)
    ensure_dir(tables_dir); ensure_dir(figs_dir)

    # Load cached features for full Train and Test
    tr_npz = Path(os.path.join(cache_dir, "train.npz"))
    te_npz = Path(os.path.join(cache_dir, "test.npz"))
    Xtr_layers, ytr, qtr = load_feature_cache(tr_npz)   # Full Train features
    Xte_layers, yte, qte = load_feature_cache(te_npz)   # Test features
    L = len(Xtr_layers)  # total number of layers

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # --- Split Train into Train_small (80%) and Validation (20%) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(gss.split(Xtr_layers[0], ytr, groups=qtr))
    X_train_layers = {li: Xtr_layers[li][train_idx] for li in Xtr_layers}
    X_val_layers   = {li: Xtr_layers[li][val_idx]   for li in Xtr_layers}
    y_train, y_val = ytr[train_idx], ytr[val_idx]
    q_train, q_val = qtr[train_idx], qtr[val_idx]

    # --- Hyperparameter tuning on a mid layer using Train_small ---
    mid_li = sorted(X_train_layers.keys())[L//2]
    X_tune, y_tune = X_train_layers[mid_li], y_train
    tuned = {}
    for m in methods:
        if m in {"lda", "logreg", "linsvm"}:
            tuned[m] = tune_probes_on_layer(X_tune, y_tune, random_state=args.seed, method=m)
        else:
            tuned[m] = None

    # --- Train & evaluate probes on Validation for each layer ---
    res_val_all = {}
    for m in methods:
        if m == "massmean":
            # Train on Train_small and eval on Validation (per layer)
            W = mass_mean_eval_per_layer_feature(X_train_layers, y_train)
            res = {}
            for li, (w, b) in W.items():
                Xv = X_val_layers[li]
                s_val = Xv @ w + b
                acc_val = acc_question(s_val, y_val, q_val)
                try:
                    au_val = roc_auc(y_val, s_val)
                except Exception:
                    au_val = np.nan
                mm_val = mean_margin(s_val, y_val, q_val)
                res[li] = {"acc": acc_val, "auroc": au_val, "mean_margin": mm_val, "scores": s_val}
            res_val_all[m] = res
        else:
            # Train on Train_small, eval on Validation
            res = fit_eval_probes_per_layer(
                X_train_layers, y_train, X_val_layers, y_val,
                best_params=tuned[m], method=m
            )
            # Override accuracy with question-level accuracy and compute margin on Validation
            for li, r in res.items():
                s_val = r.get("scores")
                if s_val is None:
                    continue
                accQ_val = acc_question(s_val, y_val, q_val)   # دقت هر سوال در Validation
                mm_val   = mean_margin(s_val, y_val, q_val)    # حاشیهٔ میانگین در Validation
                r["acc_question"] = accQ_val
                r["mean_margin"]  = mm_val
                r["acc"] = accQ_val
            res_val_all[m] = res

    # --- Determine best layer(s) per metric using Validation results ---
    best_by_list = [mb.strip().lower() for mb in args.best_by.split(",") if mb.strip()]
    best_by_map = {mb: {} for mb in best_by_list}      # metric -> {method -> best_layer}
    best_union  = {m: set() for m in methods}          # method -> set of best layers across metrics
    for m, res in res_val_all.items():
        if not res:
            continue
        for mb in best_by_list:
            best_li, _ = _pick_best_layer(res, metric=mb)
            best_by_map[mb][m] = best_li
            best_union[m].add(best_li)
    for mb, mp in best_by_map.items():
        print(f"[best_by={mb}] " + ", ".join(f"{m}:{li}" for m, li in mp.items()))
    available_layers = sorted({int(li) for res in res_val_all.values() for li in res.keys()})
    want_layers = _spec_layers(args.viz_layers, L, best_map=best_union, available_layers=available_layers)
    print("want_layers:", want_layers)
    want_pca_layers = _spec_layers(args.pca_layers, L, best_map=best_union, available_layers=available_layers) if args.pca_layers else []

    # --- Retrain on full Train and evaluate on Test (final results) ---
    res_all = {}
    for m in methods:
        if m == "massmean":
            # Fit on full Train, eval on Test for each layer
            W_full = mass_mean_eval_per_layer_feature(Xtr_layers, ytr)
            res = {}
            for li, (w, b) in W_full.items():
                Xt = Xte_layers[li]
                s_test = Xt @ w + b
                acc_test = acc_question(s_test, yte, qte)
                try:
                    au_test = roc_auc(yte, s_test)
                except Exception:
                    au_test = np.nan
                mm_test = mean_margin(s_test, yte, qte)
                res[li] = {"acc": acc_test, "auroc": au_test, "mean_margin": mm_test, "scores": s_test}
            res_all[m] = res
        else:
            res = fit_eval_probes_per_layer(
                Xtr_layers, ytr, Xte_layers, yte,
                best_params=tuned[m], method=m
            )
            # Compute question-level accuracy and margin on Test, override acc
            for li, r in res.items():
                s_test = r.get("scores")
                if s_test is None:
                    continue
                accQ_test = acc_question(s_test, yte, qte)   # دقت هر سوال در Test
                mm_test   = mean_margin(s_test, yte, qte)    # حاشیهٔ میانگین در Test
                r["acc_question"] = accQ_test
                r["mean_margin"]  = mm_test
                r["acc"] = accQ_test
            res_all[m] = res

    # --- Save per-layer metrics to CSV (Test results) ---
    for m, res in res_all.items():
        rows = []
        for li, r in sorted(res.items()):
            rows.append({
                "layer": li,
                "acc":   r.get("acc",   np.nan),
                "auroc": r.get("auroc", np.nan),
                "mean_margin": r.get("mean_margin", np.nan),
                "fisher": r.get("fisher", np.nan),
                "thr0":  r.get("thr0",  0.0),
                "thr*":  r.get("thr_star", 0.0),
                "tp0":   r.get("cm_thr0", {}).get("tp", np.nan),
                "fp0":   r.get("cm_thr0", {}).get("fp", np.nan),
                "tn0":   r.get("cm_thr0", {}).get("tn", np.nan),
                "fn0":   r.get("cm_thr0", {}).get("fn", np.nan)
            })
        out_csv = os.path.join(tables_dir, f"{args.task}_{m}_perlayer_metrics.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("[SAVE]", out_csv)

    # --- Plot combined diagnostics for specified layers (on Test data) ---
    for m in methods:
        if m not in res_all:
            continue
        for li in want_layers:
            if li not in res_all[m]:
                continue
            layer_res = res_all[m][li]
            s = layer_res.get('scores', layer_res.get('score'))
            badges = [mb for mb in best_by_list if best_by_map.get(mb, {}).get(m) == li]
            title = f"{args.task.upper()} | {m.upper()} | Layer {li}"
            if badges:
                title += " (BEST by " + ",".join(b.upper() for b in badges) + ")"
            out_png = os.path.join(figs_dir, f"{args.task}_{m}_layer_{li}.png")
            plot_combined_diagnostics(
                s, yte, title, out_png,
                show_kde=True,
                pca_Xtr=Xtr_layers[li].astype(np.float32, copy=False),
                pca_Xte=Xte_layers[li].astype(np.float32, copy=False),
                pca_yte=yte
            )

    # --- Plot accuracy/AUROC curves across layers (Test data) ---
    for m, res in res_all.items():
        acc_map    = {li: r.get("acc", np.nan) for li, r in res.items()}
        margin_map = {li: r.get("mean_margin", np.nan) for li, r in res.items()}
        auroc_map  = {li: r.get("auroc", np.nan) for li, r in res.items()}
        out_acc = os.path.join(figs_dir, f"{args.task}_{m}_acc_curve.png")
        plot_layer_acc_curve(
            acc_map, margin_map, auroc_map,
            f"{args.task.upper()} {m.upper()} per-layer ACC ({args.model})",
            out_acc,
            highlight_best=True
        )

    # --- PCA variance ratios per selected layers (on Test) ---
    if want_pca_layers:
        rng = np.random.default_rng(args.seed)
        rows = []
        for li in want_pca_layers:
            if li not in Xte_layers:
                continue
            X = Xte_layers[li]
            y = yte
            if args.pca_sample and len(X) > args.pca_sample:
                idx = rng.choice(len(X), size=args.pca_sample, replace=False)
                X = X[idx]
                y = y[idx]
            pca = PCA(n_components=min(args.pca_components, X.shape[1]))
            X_pca = pca.fit_transform(X)
            var = pca.explained_variance_ratio_
            rows.append({
                "layer": li,
                **{f"pca_var_{i+1}": (float(var[i]) if i < len(var) else np.nan)
                   for i in range(args.pca_components)},
                "n_samples": len(X)
            })
            if args.pca_scatter and X_pca.shape[1] >= 2:
                out_scatter = os.path.join(figs_dir, f"{args.task}_layer{li}_pca_scatter.png")
                plot_pca_scatter_2d(X_pca[:, :2], y, f"PCA scatter (layer {li})", out_scatter)
        if rows:
            df_pca = pd.DataFrame(rows).sort_values("layer")
            out_pca_csv = os.path.join(tables_dir, f"{args.task}_pca_varratio.csv")
            df_pca.to_csv(out_pca_csv, index=False)
            print("[SAVE]", out_pca_csv)
            if args.pca_components >= 2:
                plt_path = os.path.join(figs_dir, f"{args.task}_pca_varratio.png")
                plot_pca_variance_curve(
                    df_pca,
                    f"{args.task.upper()} PCA variance ratios per layer ({args.model})",
                    plt_path,
                    n_components=min(args.pca_components, 2)
                )

    # --- Save best layers by metrics (based on Validation selection) ---
    rows = []
    for mb, mp in best_by_map.items():
        for m, li in mp.items():
            rows.append({"metric": mb, "method": m, "best_layer": int(li)})
    pd.DataFrame(rows).to_csv(
        os.path.join(tables_dir, f"{args.task}_best_layers_by_metrics.csv"),
        index=False
    )

if __name__ == "__main__":
    main()
