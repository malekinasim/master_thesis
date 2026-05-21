#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End robustness (T5): baseline + noise-vs-accuracy curves per layer
- MCQ: sum of log-probs over multi-token options (teacher-forced), ACC@Q, AUROC, flip-rate
- Single: multi-token exact-match (teacher-forced)
- Robustness per layer = normalized AUC under ACC(sigma) curve
"""

import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# --- your helpers (existing in repo) ---
from src.io import load_prompts_with_options              # dataset loading (MCQ/Single)  # noqa: E402
from src.util import load_model_and_tokenizer             # model/tokenizer loader        # noqa: E402
# feature cache to get per-layer, per-dim std for rel-noise
from src.feature_cache import load_feature_cache           # cache IO                      # noqa: E402

# --- AUROC helper (fallback if needed) ---
try:
    from src.metrics import roc_auc                        # your metrics                  # noqa: E402
except Exception:
    from sklearn.metrics import roc_auc_score
    def roc_auc(y, s):
        return float(roc_auc_score(y, s))


# ---------------------- utilities ----------------------
def try_get_layers(model):
    """
    Return list of transformer block modules to hook for different HF models.
    """
    paths = [
        ("transformer.h",       lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("model.layers",        lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("gpt_neox.layers",     lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
    ]
    for name, getter in paths:
        blocks = getter(model)
        if blocks is not None:
            blocks = list(blocks)
            if len(blocks) > 0:
                print(f"[info] using blocks from {name}: n_layers={len(blocks)}")
                return blocks
    # heuristic fallback
    cand = []
    for m in model.modules():
        if hasattr(m, "self_attn") or hasattr(m, "attention") or hasattr(m, "mlp"):
            if isinstance(m, nn.Module) and len(list(m.children())) > 0:
                cand.append(m)
    if cand:
        print(f"[warn] using heuristic block list, n={len(cand)}")
        return cand
    raise RuntimeError("Cannot locate transformer block list on this model")


def acc_question_from_scores(scores, labels, qids):
    """
    Accuracy@Question: for each question-id, does the top-scored option have label==1?
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    qids   = np.asarray(qids)
    acc = []
    for q in np.unique(qids):
        idx = (qids == q)
        s_q = scores[idx]; y_q = labels[idx]
        if s_q.size == 0: 
            continue
        acc.append(1 if y_q[int(np.argmax(s_q))] == 1 else 0)
    return float(np.mean(acc)) if len(acc) else float("nan")


# --------- robust std from cache (per-layer, per-dim) ----------
def _robust_std_vector(X, q_clip=(5,95), ddof=0):
    s = np.asarray(X, dtype=np.float32).std(axis=0, ddof=ddof)
    s[~np.isfinite(s)] = np.nan
    pos = s[(s>0) & np.isfinite(s)]
    if pos.size == 0:
        s_typ = float(np.nanstd(X))
        if not np.isfinite(s_typ) or s_typ == 0.0:
            s_typ = 1.0
        return torch.from_numpy(np.full_like(s, s_typ, dtype=np.float32))
    med = float(np.nanmedian(pos))
    s = np.where(np.isfinite(s) & (s>0), s, med)
    if q_clip is not None:
        lo, hi = np.nanpercentile(pos, q_clip)
        s = np.clip(s, lo, hi)
    return torch.from_numpy(s.astype(np.float32))


def get_layer_std_from_cache(out_root, model_path, task):
    """
    Load per-layer per-dim std from training cache (used in 'rel' noise).
    Falls back to None if cache absent.
    Uses your existing NPZ layout.  (src/feature_cache.py)
    """
    cache_path = Path(out_root) / "features" / model_path / task / "train.npz"
    if not cache_path.exists():
        print(f"[warn] training cache not found: {cache_path}")
        return None
    X_layers, _, _ = load_feature_cache(cache_path)  # -> dict[layer] -> [N,D]  :contentReference[oaicite:4]{index=4}
    stds = {}
    for li, X in X_layers.items():
        stds[int(li)] = _robust_std_vector(X)  # CPU [D]
    return stds


# --------- prompt/completion merging & tokenization-safe ids ----------
def _merge_prompt_completion(prompt: str, completion: str, add_space_between: bool = True):
    """
    Ensure a clean prompt/completion boundary for tokenizers:
    If completion doesn't start with a space and prompt doesn't end with space/newline,
    insert a single bridge space. Return (merged_text, start_char_of_completion).
    """
    p = str(prompt)
    c = str(completion)
    need_bridge = (add_space_between and c and not c[:1].isspace() and not p.endswith((" ", "\t", "\n")))
    if need_bridge:
        return p + " " + c, len(p) + 1
    else:
        return p + c, len(p)


def _completion_token_ids_in_context(tok, prompt: str, completion: str, add_space_between: bool = True):
    """
    Token-ids of completion *in the prompt+completion context*, via offset_mapping.
    """
    merged, start_char = _merge_prompt_completion(prompt, completion, add_space_between=add_space_between)
    enc = tok(
        merged, return_tensors="pt", padding=False, truncation=False,
        return_offsets_mapping=True, add_special_tokens=False
    )
    ids  = enc["input_ids"][0].tolist()
    offs = enc["offset_mapping"][0].tolist()
    out = []
    for i, (a, b) in enumerate(offs):
        if b > start_char and a >= start_char:
            out.append(int(ids[i]))
    return out


# --------- next-token log-probs (for MCQ sum-logprob) ----------
def _sum_logprob_for_ids(model, tok, device, prompt: str, token_ids: list[int]) -> float:
    """
    Teacher-forced sum of next-token log-probabilities over a *multi-token* completion.
    """
    s = 0.0
    prefix = ""
    for tid in token_ids:
        inp = tok(prompt + prefix, return_tensors="pt", padding=False).to(device)
        z = model(**inp).logits[:, -1, :]            # [1, V]
        logp = torch.log_softmax(z, dim=-1)[0, tid]  # scalar
        s += float(logp.item())
        prefix += tok.decode([tid], clean_up_tokenization_spaces=False)
    return float(s)


def mcq_score_sum_logprob(model, tok, device, q: str, options: list[str], gold: str):
    """
    Robust MCQ scoring: per option -> sum of log-probs over its (multi-token) ids
    (computed in the (prompt + generated_so_far) context).
    Returns scores (list[float]), labels (list[int]).
    """
    scores, labels = [], []
    for o in options:
        ids = _completion_token_ids_in_context(tok, q, o, add_space_between=True)
        if len(ids) == 0: 
            continue
        s = _sum_logprob_for_ids(model, tok, device, q, ids)
        scores.append(s)
        labels.append(1 if (str(o).strip() == str(gold).strip()) else 0)
    return scores, labels

 

# --------- Single: multi-token exact-match via teacher forcing ----------
@torch.no_grad()
def single_exact_match(model, tok, device, prompt: str, completion: str, max_steps: int = 64) -> bool:
    """
    At each step, top-1(next-token) must equal the gold token for that step.
    """
    gold_ids = _completion_token_ids_in_context(tok, prompt, completion, add_space_between=True)
    if not gold_ids:
        return False
    prefix = ""
    for tid in gold_ids[:max_steps]:
        enc = tok(prompt + prefix, return_tensors="pt", padding=False).to(device)
        z = model(**enc).logits[:, -1, :]
        pred = int(z.argmax(dim=-1).item())
        if pred != tid:
            return False
        prefix += tok.decode([tid], clean_up_tokenization_spaces=False)
    return True

def case_variants(ans: str):
    a = ans.strip()
    cands = [a, a.lower(), a.title(), a.upper()]
    # با و بدون فاصلهٔ پیشرو (برای اولین توکن):
    cands += [" " + x for x in list(set(cands))]
    return list(dict.fromkeys(cands))  # یکتا

def single_exact_match_any_variant(model, tok, device, q, gold, max_steps=64):
    for cand in case_variants(gold):
        ok= single_exact_match(model, tok, device, q, cand, max_steps=max_steps)
        if ok:
            return True
    return False 
# --------- Pre-tokenized helpers for speed ----------
def prepare_mcq_items(tok, raw_items, require_single_token=False):
    """
    Pre-tokenize MCQ items so scoring reuses token ids across noise levels.
    Each option is represented by:
      - full ids of (prompt+option)
      - prompt_len: how many tokens belong to the prompt
      - is_correct: whether this option is the gold answer
    """
    prepared = []
    for i, it in enumerate(raw_items):
        qid = str(it["id"])
        q, opts, gold = it["question"], it["options"], it["answer"]
        base_prompt = f"{q}\nOptions: {' | '.join(opts)}\nAnswer:"
        opts_data = []
        for o in opts:
            # merge prompt + completion with a clean boundary
            merged, start_char = _merge_prompt_completion(base_prompt, o, add_space_between=True)

            # IMPORTANT: return_tensors="pt" تا بعداً .tolist() درست کار کند
            enc = tok(
                merged,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            ids  = enc["input_ids"][0].tolist()          # list[int]
            offs = enc["offset_mapping"][0].tolist()     # list[(start,end)]

            # تعداد توکن‌های متعلق به prompt (بر حسب offset)
            prompt_len = sum(1 for (a, b) in offs if b <= start_char)
            opt_len    = len(ids) - prompt_len
            if prompt_len <= 0 or opt_len <= 0:
                continue
            if require_single_token and opt_len != 1:
                continue

            opts_data.append({
                "ids": ids,
                "prompt_len": prompt_len,
                "is_correct": str(o).strip() == str(gold).strip(),
            })

        # فقط سوال‌هایی که حداقل ۲ گزینهٔ معتبر دارند
        if len(opts_data) >= 2:
            prepared.append({"qid": qid, "options": opts_data})
    return prepared


def prepare_single_items(tok, raw_items):
    """
    Pre-tokenize Single items with case/space variants.
    """
    prepared = []
    for i, it in enumerate(raw_items):
        qid = str(it["id"])
        variants = []
        for cand in case_variants(it["answer"]):
            merged, start_char = _merge_prompt_completion(
                it["question"], cand, add_space_between=True
            )
            enc = tok(
                merged,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            ids  = enc["input_ids"][0].tolist()
            offs = enc["offset_mapping"][0].tolist()
            prompt_len = sum(1 for (a, b) in offs if b <= start_char)
            comp_ids   = ids[prompt_len:]
            if prompt_len <= 0 or len(comp_ids) == 0:
                continue
            variants.append({
                "ids": ids,
                "prompt_len": prompt_len,
                "comp_ids": comp_ids,
            })
        if variants:
            prepared.append({"qid": qid, "variants": variants})
    return prepared


@torch.no_grad()
def mcq_eval_fast(model, tok, device, prepared_items, baseline_winner=None, scoring_mode: str = "sumlogprob"):
    """
    Batched MCQ scoring per question using pre-tokenized ids.
    """
    pad_id = tok.pad_token_id
    scores_all, labels_all, qids_all = [], [], []
    winners = {}
    flips = 0
    n_questions = 0

    for item in prepared_items:
        opt_ids_list = [o["ids"] for o in item["options"]]
        max_len = max(len(ids) for ids in opt_ids_list)
        input_ids = torch.full((len(opt_ids_list), max_len), pad_id, device=device)
        attn = torch.zeros_like(input_ids)
        starts = []
        for b, ids in enumerate(opt_ids_list):
            L = len(ids)
            start = max_len - L  # left-pad so last position is real token
            starts.append(start)
            input_ids[b, start:start+L] = torch.tensor(ids, device=device)
            attn[b, start:start+L] = 1

        logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, T, V]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)

        opt_scores = []
        valid_idx = []
        for b, opt in enumerate(item["options"]):
            prompt_len = opt["prompt_len"]
            opt_len = 1 if scoring_mode == "firsttoken" else len(opt["ids"]) - prompt_len
            if prompt_len <= 0 or opt_len <= 0:
                continue
            start = starts[b]
            positions = torch.arange(start + prompt_len - 1, start + prompt_len - 1 + opt_len, device=device)
            targets = torch.tensor(opt["ids"][prompt_len:prompt_len + opt_len], device=device)
            s = logp[b, positions, targets].sum()
            opt_scores.append(float(s.item()))
            valid_idx.append(b)

        if len(opt_scores) < 2:
            continue

        n_questions += 1
        best = int(np.argmax(np.array(opt_scores)))
        winners[item["qid"]] = best
        if baseline_winner is not None and item["qid"] in baseline_winner and best != baseline_winner[item["qid"]]:
            flips += 1

        for j, b in enumerate(valid_idx):
            opt = item["options"][b]
            scores_all.append(opt_scores[j])
            labels_all.append(1 if opt["is_correct"] else 0)
            qids_all.append(item["qid"])

    acc = acc_question_from_scores(scores_all, labels_all, qids_all)
    try:
        au = roc_auc(np.asarray(labels_all, dtype=int), np.asarray(scores_all, dtype=float))
    except Exception:
        au = float("nan")
    flip_rate = flips / max(n_questions, 1) if baseline_winner is not None else 0.0
    return acc, au, flip_rate, winners, n_questions


@torch.no_grad()
def single_eval_fast(model, tok, device, prepared_items, baseline_winner=None):
    """
    Batched Single exact-match with case variants per item.
    """
    pad_id = tok.pad_token_id
    correct, total, flips = 0, 0, 0
    winners = {}

    for item in prepared_items:
        variants = item["variants"]
        max_len = max(len(v["ids"]) for v in variants)
        input_ids = torch.full((len(variants), max_len), pad_id, device=device)
        attn = torch.zeros_like(input_ids)
        prompt_lens, comp_lens = [], []
        starts = []
        for b, var in enumerate(variants):
            ids = var["ids"]
            L = len(ids)
            start = max_len - L  # left-pad so last position is real token
            starts.append(start)
            input_ids[b, start:start+L] = torch.tensor(ids, device=device)
            attn[b, start:start+L] = 1
            prompt_lens.append(var["prompt_len"])
            comp_lens.append(len(var["comp_ids"]))

        logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, T, V]
        ok_any = False
        for b, var in enumerate(variants):
            p_len = prompt_lens[b]
            c_len = comp_lens[b]
            if p_len <= 0 or c_len <= 0:
                continue
            start = starts[b]
            positions = torch.arange(start + p_len - 1, start + p_len - 1 + c_len, device=device)
            preds = logits[b, positions, :].argmax(dim=-1)
            target = torch.tensor(var["comp_ids"], device=device)
            if torch.equal(preds, target):
                ok_any = True
                break

        winners[item["qid"]] = ok_any
        if baseline_winner is not None and item["qid"] in baseline_winner and ok_any != baseline_winner[item["qid"]]:
            flips += 1
        correct += int(ok_any)
        total += 1

    acc = correct / max(total, 1)
    flip_rate = flips / max(total, 1) if baseline_winner is not None else 0.0
    return acc, float("nan"), flip_rate, winners, total
# --------- hook factories ----------
def _make_position_indices(T, mode="last", k=3):
    if mode == "last":
        return [T-1]
    if mode == "all":
        return list(range(T))
    if mode == "window":
        start = max(0, T-k)
        return list(range(start, T))
    return [T-1]
def make_noise_hook(sig_val: float, std_vec: torch.Tensor | None,
                    pos_mode: str, window_k: int,
                    generator: torch.Generator):
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out   # [B,T,D]
        dtype, dev = h.dtype, h.device
        B, T, D = h.shape

        per_dim = (std_vec.to(dev, dtype) if std_vec is not None
                   else torch.ones(D, device=dev, dtype=dtype))

        idxs = _make_position_indices(T, mode=pos_mode, k=window_k)

        h2 = h.clone()
        for t in idxs:
            eps_t = torch.randn((B, D), device=dev, dtype=dtype, generator=generator) * (sig_val * per_dim)
            h2[:, t, :] = h2[:, t, :] + eps_t

        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return _hook

def make_noise_hook_old(sig_val: float, std_vec: torch.Tensor | None,
                    pos_mode: str, window_k: int,
                    generator: torch.Generator):
    """
    Forward hook that adds N(0, (sigma * std_vec)^2) to hidden states at chosen positions.
    Works with outputs that are Tensor or tuple(..., hidden, ...).
    """
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out           # [B,T,D]
        dtype, dev = h.dtype, h.device
        B, T, D = h.shape
        per_dim = (std_vec.to(dev, dtype) if std_vec is not None
                   else torch.ones(D, device=dev, dtype=dtype))
        idxs = _make_position_indices(T, mode=pos_mode, k=window_k)
        eps = torch.randn((B, D), device=dev, dtype=dtype, generator=generator) * (sig_val * per_dim)
        h2 = h.clone()
        for t in idxs:
            h2[:, t, :] = h2[:, t, :] + eps
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return _hook


# --------- plotting / robustness AUC ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _normalized_auc(sigmas, accs):
    """
    Trapezoidal AUC normalized to [0,1] by dividing by (sigma_max - sigma_min).
    Requires sigma sorted ascending. Includes sigma=0 baseline.
    """
    x = np.asarray(sigmas, dtype=float)
    y = np.asarray(accs,  dtype=float)
    if len(x) < 2:
        return float("nan")
    area = np.trapz(y, x)
    denom = (x[-1] - x[0])
    return float(area / denom) if denom > 0 else float("nan")


# ---------------------- main ----------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("T5 End-to-End robustness vs noise per layer")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--task",  choices=["mcq","single"], default="single")
    ap.add_argument("--dataset_root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--out_root",     default=str(REPO_ROOT / "out"))
    ap.add_argument("--sigmas",       default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2")
    ap.add_argument("--noise_mode",   choices=["rel","abs"], default="rel")
    ap.add_argument("--repeats",      type=int, default=1)
    ap.add_argument("--position",     choices=["last","all","window"], default="all")
    ap.add_argument("--window_k",     type=int, default=3)
    ap.add_argument("--require_single_token", action="store_true",
                    help="for MCQ: if set, keep only single-token options; else multi-token options are allowed")
    ap.add_argument("--device",       default="auto")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--mcq_score",    choices=["sumlogprob","firsttoken"], default="sumlogprob",
                    help="MCQ scoring mode; sumlogprob is robust to multi-token options")
    ap.add_argument("--max_items",    type=int, default=5,
                    help="if >0, limit number of items used (for speed on CPU)")
    ap.add_argument("--no_plots", action="store_true",
                help="skip plotting (CSV only)")
    args = ap.parse_args()

    # device / seed
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # load model/tokenizer (your util)  :contentReference[oaicite:5]{index=5}
    model, tok = load_model_and_tokenizer(args.model, device=device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so last position is a real token (noise hook hits non-PAD)

    # load dataset (your util)  :contentReference[oaicite:6]{index=6}
    model_path = args.model.replace("/", "__")
    ds_path = Path(args.dataset_root) / model_path / "prompt_tar.json"
    if not ds_path.exists():
        ds_path = Path(args.dataset_root) / model_path / "prompt_pool.json"
    d_mcq, d_single = load_prompts_with_options(str(ds_path), tok,
                                                require_single_token=(args.require_single_token if args.task=="mcq" else False))

    # pick items
    if args.task == "mcq":
        items = [it for it in d_mcq
                 if isinstance(it.get("question"), str)
                 and isinstance(it.get("options"), list)
                 and it.get("answer") in it["options"]]
        if not items:
            raise RuntimeError("No MCQ items to evaluate.")
    else:
        items = [it for it in d_single
                 if isinstance(it.get("question"), str)
                 and isinstance(it.get("answer"),  str)]
        if not items:
            raise RuntimeError("No SINGLE items to evaluate.")

    # optional subsample for speed
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]

    # layers to hook
    blocks = try_get_layers(model)
    n_layers = len(blocks)

    # sigmas (ensure include 0.0)
    sigmas = sorted(set(float(s) for s in args.sigmas.split(",") if s.strip()))
    if 0.0 not in sigmas:
        sigmas = [0.0] + sigmas

    # per-layer std for rel-mode via cache (your NPZ layout)  :contentReference[oaicite:7]{index=7}
    stds = None
    if args.noise_mode == "rel":
        stds = get_layer_std_from_cache(args.out_root, model_path, args.task)
        if stds is None:
            print("[warn] cache not found -> fallback to abs noise")
            args.noise_mode = "abs"

    # output dirs
    root_reports = Path(args.out_root) / "reports" / model_path
    tables_dir = root_reports  /"robustness" /"e2e"/"tables"/args.task
    figs_dir   = root_reports /"robustness" /"e2e"/"figures"/args.task 
    _ensure_dir(tables_dir); _ensure_dir(figs_dir)

    # ---------------- Pre-tokenize ----------------
    if args.task == "mcq":
        prepared = prepare_mcq_items(tok, items, require_single_token=args.require_single_token)
        if not prepared:
            raise RuntimeError("No MCQ items to evaluate after tokenization.")
    else:
        prepared = prepare_single_items(tok, items)
        if not prepared:
            raise RuntimeError("No SINGLE items to evaluate after tokenization.")

    # ---------------- Baseline (sigma=0) ----------------
    print("[stage] baseline (sigma=0)")
    if args.task == "mcq":
        base_acc, base_au, _, baseline_winner_by_q, n_questions = mcq_eval_fast(
            model, tok, device, prepared, scoring_mode=args.mcq_score
        )
        print(f"[baseline] ACC@Q={base_acc:.4f} AUROC={base_au:.4f} Nq={n_questions}")
    else:
        base_acc, base_au, _, baseline_winner_by_q, n_questions = single_eval_fast(
            model, tok, device, prepared
        )
        print(f"[baseline] ACC={base_acc:.4f} N={n_questions}")

    if n_questions == 0:
        raise RuntimeError("No valid questions after filtering.")
    # ---------------- Evaluate: noise per layer - sigma ----------------
    rows = []  # for big CSV: (layer, sigma, acc, au, flip_rate, n)
    perlayer_curve = {li: {"sigma": [], "acc": []} for li in range(n_layers)}  # for AUC

    for li, block in enumerate(blocks):
        layer_std = None
        if args.noise_mode == "rel" and stds is not None and li in stds:
            layer_std = stds[li]  # CPU [D]

        for sig in sigmas:
            # baseline point reuse
            if sig == 0.0:
                rows.append({
                    "layer": li, "sigma": sig,
                    "acc": base_acc, "auroc": base_au,
                    "flip_rate": 0.0, "n": n_questions
                })
                perlayer_curve[li]["sigma"].append(sig)
                perlayer_curve[li]["acc"].append(base_acc)
                continue

            # repeat‑average
            acc_list, au_list, flip_list = [], [], []
            for rep in range(args.repeats):
                g = torch.Generator(device=device).manual_seed(args.seed + 1000*li + 10*rep + 3)

                # attach hook on this layer
                hook = block.register_forward_hook(
                    make_noise_hook(sig_val=sig, std_vec=layer_std,
                                    pos_mode=args.position, window_k=args.window_k,
                                    generator=g)
                )
                try:
                    if args.task == "mcq":
                        acc, au, flip_rate, _, _ = mcq_eval_fast(
                            model, tok, device, prepared,
                            baseline_winner=baseline_winner_by_q,
                            scoring_mode=args.mcq_score,
                        )
                        acc_list.append(acc); au_list.append(au); flip_list.append(flip_rate)

                    else:  # Single
                        acc, au, flip_rate, _, _ = single_eval_fast(
                            model, tok, device, prepared,
                            baseline_winner=baseline_winner_by_q,
                        )
                        acc_list.append(acc); au_list.append(au); flip_list.append(flip_rate)

                finally:
                    hook.remove()

            # aggregate repeats
            acc_m = float(np.nanmean(acc_list))
            au_m  = float(np.nanmean(au_list)) if args.task == "mcq" else float("nan")
            flip_m= float(np.nanmean(flip_list))

            rows.append({
                "layer": li, "sigma": sig,
                "acc": acc_m, "auroc": au_m,
                "flip_rate": flip_m, "n": n_questions
            })
            perlayer_curve[li]["sigma"].append(sig)
            perlayer_curve[li]["acc"].append(acc_m)

        # --- save per-layer plot: ACC vs Sigma ---
        if not args.no_plots:
            s = np.array(perlayer_curve[li]["sigma"])
            a = np.array(perlayer_curve[li]["acc"])
            idx = np.argsort(s); s = s[idx]; a = a[idx]
            plt.figure(figsize=(5,4))
            plt.plot(s, a, marker="o")
            plt.xlabel("sigma (noise std)")
            plt.ylabel("Accuracy")
            plt.title(f"Layer {li}: ACC vs sigma")
            plt.grid(alpha=0.3)
            out_png = figs_dir / f"acc_vs_sigma_layer{li}.png"
            plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

    # --- big CSV: per (layer, sigma) ---
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["layer","sigma"])
    out_csv = tables_dir / f"{args.task}_e2e_noise_curves.csv"
    df.to_csv(out_csv, index=False)
    print("[SAVE]", out_csv)

    # --- Robustness (AUC over sigma) per layer ---
    rob_rows = []
    for li in range(n_layers):
        s = np.array(perlayer_curve[li]["sigma"])
        a = np.array(perlayer_curve[li]["acc"])
        idx = np.argsort(s); s = s[idx]; a = a[idx]
        rob = _normalized_auc(s, a)
        rob_rows.append({"layer": li, "robustness_auc": rob, "baseline_acc": a[s==0.0][0] if (s==0.0).any() else np.nan})

    df_rob = pd.DataFrame(rob_rows).sort_values("layer")
    out_csv2 = tables_dir / f"{args.task}_e2e_robustness_per_layer.csv"
    df_rob.to_csv(out_csv2, index=False)
    print("[SAVE]", out_csv2)

    # --- plot robustness vs layer ---
    if not args.no_plots:
        plt.figure(figsize=(7,4))
        plt.plot(df_rob["layer"], df_rob["robustness_auc"], marker="s")
        plt.xlabel("Layer")
        plt.ylabel("Robustness (AUC of ACC vs sigma)")
        plt.title("End-to-End Robustness per Layer")
        plt.grid(alpha=0.3)
        out_png2 = figs_dir / f"{args.task}_robustness_vs_layer.png"
        plt.tight_layout(); plt.savefig(out_png2, dpi=180); plt.close()
        print("[SAVE]", out_png2)


if __name__ == "__main__":
    main()
