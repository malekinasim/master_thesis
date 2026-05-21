# margins.py
# Utilities to compute per-option and per-class margins from logits.
# All functions assume the last dimension corresponds to classes/options.

from typing import Optional
import torch
from src.hooks import *
from src.tuned import TunedDiag
from typing import List, Dict, Optional, Any
# ---------------------------------------------------------------------------
# Top-1 vs. Top-2 margin (single example)
# ---------------------------------------------------------------------------
def top1_top2_margin_1d(z: torch.Tensor) -> float:
    """
    Compute (top1 - top2) from a 1D logits vector.

    Args:
        z: 1D tensor of shape [V] (V = number of classes/options).

    Returns:
        float: top1 - top2 margin. If V < 2, returns NaN.
    """
    if z.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(z.shape)}")
    if z.numel() < 2:
        return float("nan")
    v = torch.topk(z, k=2).values
    return float((v[0] - v[1]).item())


# ---------------------------------------------------------------------------
# Top-1 vs. Top-2 margin (batched)
# ---------------------------------------------------------------------------
def top1_top2_margin_batched(z: torch.Tensor) -> torch.Tensor:
    """
    Compute (top1 - top2) along the last dimension for a batched logits tensor.

    Args:
        z: Tensor of shape [..., V] where V is the number of classes/options.

    Returns:
        Tensor of shape [...] with top1 - top2 margins.
        If V < 2, returns a tensor filled with NaNs.
    """
    V = z.shape[-1]
    if V < 2:
        return torch.full(z.shape[:-1], float("nan"), dtype=z.dtype, device=z.device)
    # topk(..., k=2) over the last dim: returns (..., 2)
    vals, _ = torch.topk(z, k=2, dim=-1)
    return (vals[..., 0] - vals[..., 1])


# ---------------------------------------------------------------------------
# Gold margin over full vocabulary/classes (single example)
# ---------------------------------------------------------------------------
def gold_margin_from_logits_1d(z: torch.Tensor, gold_id: int) -> Optional[float]:
    """
    Gold margin = logits[gold] - max(logits of all other classes).

    Args:
        z: 1D logits tensor of shape [V].
        gold_id: integer index of the gold/true class in [0, V).

    Returns:
        Optional[float]: margin value as Python float, or None if there is no competitor (V <= 1).
    """
    if z.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(z.shape)}")
    V = z.shape[0]
    if not (0 <= gold_id < V):
        raise IndexError(f"gold_id {gold_id} is out of range [0, {V})")
    if V <= 1:
        return None  # no competitor

    # Build a boolean mask: True for competitors, False for gold.
    mask = torch.ones(V, dtype=torch.bool, device=z.device)
    mask[gold_id] = False

    # Replace gold position with a very negative value so it never wins the max.
    very_neg = torch.finfo(z.dtype).min
    z_comp = z.masked_fill(~mask, very_neg)

    max_comp = torch.max(z_comp)
    margin = (z[gold_id] - max_comp).item()
    return float(margin)


# ---------------------------------------------------------------------------
# Gold margin over full vocabulary/classes (batched)
# ---------------------------------------------------------------------------
def gold_margin_from_logits_batched(z: torch.Tensor, gold_ids: torch.Tensor) -> torch.Tensor:
    """
    Batched version of gold margin over the last dimension.

    Args:
        z: Tensor of shape [..., V] (logits).
        gold_ids: Long tensor of shape [...] with gold indices aligned to z's batch dims.

    Returns:
        Tensor of shape [...] with margins. If V < 2, returns NaNs.
    """
    V = z.shape[-1]
    if V < 2:
        return torch.full(z.shape[:-1], float("nan"), dtype=z.dtype, device=z.device)

    # Build a mask: True for competitors, False where index == gold.
    ar = torch.arange(V, device=z.device).view(*([1] * (z.ndim - 1)), V)
    mask = (ar != gold_ids.unsqueeze(-1))  # shape [..., V]

    very_neg = torch.finfo(z.dtype).min
    z_comp = z.masked_fill(~mask, very_neg)
    max_comp, _ = z_comp.max(dim=-1)

    gold_vals = z.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    return gold_vals - max_comp


# ---------------------------------------------------------------------------
# Gold margin restricted to options (single example)
# ---------------------------------------------------------------------------
def gold_margin_opts_1d(z: torch.Tensor, gidx: int) -> float:
    """
    Gold margin over a restricted options vector (e.g., MCQ options only).

    Args:
        z: 1D tensor of shape [K] (K = number of options).
        gidx: integer index of the gold option in [0, K).

    Returns:
        float: z[gidx] - max(z[others]). If K < 2, returns NaN.
    """
    if z.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(z.shape)}")
    K = z.numel()
    if not (0 <= gidx < K):
        raise IndexError(f"gidx {gidx} is out of range [0, {K})")
    if K < 2:
        return float("nan")

    # Mask-based variant—avoid empty-slice issues when gidx is at edges.
    mask = torch.ones_like(z, dtype=torch.bool)
    mask[gidx] = False
    rival = torch.max(z[mask])
    return float((z[gidx] - rival).item())


# ---------------------------------------------------------------------------
# Gold margin restricted to options (batched)
# ---------------------------------------------------------------------------
def gold_margin_opts_batched(z: torch.Tensor, gidx: torch.Tensor) -> torch.Tensor:
    """
    Batched gold margin over a restricted options tensor.

    Args:
        z: Tensor of shape [..., K] (K = number of options).
        gidx: Long tensor of shape [...] with gold indices aligned to z's batch dims.

    Returns:
        Tensor of shape [...] with margins. If K < 2, returns NaNs.
    """
    K = z.shape[-1]
    if K < 2:
        return torch.full(z.shape[:-1], float("nan"), dtype=z.dtype, device=z.device)

    ar = torch.arange(K, device=z.device).view(*([1] * (z.ndim - 1)), K)
    mask = (ar != gidx.unsqueeze(-1))  # True for rivals, False for gold

    very_neg = torch.finfo(z.dtype).min
    z_comp = z.masked_fill(~mask, very_neg)
    rival, _ = z_comp.max(dim=-1)

    gold_vals = z.gather(-1, gidx.unsqueeze(-1)).squeeze(-1)
    return gold_vals - rival

@torch.no_grad()
@torch.no_grad()
def compute_margins_per_layer_logits(
    model, tokenizer, text=None, outputs=None, pos=-1,
    ln_f_mode="last_only", skip_embedding=False,
    gold_text=None,                   # may be single- OR multi-token
    options: "list[str]|None" = None, # ditto
    gold_option: "str|None" = None,
    tuned: "TunedDiag|None" = None
):
    # --- helpers ----------------------------------------------------------
    def _ids(s: str):
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    def _decode_one(tid: int) -> str:
        # minimal safe decode for one id; works with GPT-2 family
        return tokenizer.decode([tid], clean_up_tokenization_spaces=False)

    def _per_layer_logit_for_ids(prompt: str, token_ids: list[int]) -> list[float]:
        """
        Sum logits over a multi-token completion by stepping through positions.
        Returns a Python list of len = #layers, each the cumulative logit.
        """
        sums = None
        prefix = ""
        for tid in token_ids:
            # logits for the *next* token at current end-of-text position
            res = layerwise_logits_for_pos(
                model, tokenizer,
                text=(prompt + prefix), outputs=None, pos=-1,
                ln_f_mode=ln_f_mode, skip_embedding=skip_embedding,
                tuned=tuned, option_ids=[tid]
            )
            # res is either list[Tensor([1])] or {"raw":[...],"tuned":[...]}
            layer_list = res if (tuned is None) else res["raw"]
            # convert layer tensors ([1]) -> scalar floats
            step_vals = [float(z.view(-1)[0].item()) for z in layer_list]
            if sums is None:
                sums = step_vals
            else:
                sums = [a + b for a, b in zip(sums, step_vals)]
            prefix += _decode_one(tid)  # advance prefix
        return sums or []

    # --- gold ids (full-vocab) -------------------------------------------
    gold_ids = None
    if gold_text is not None:
        gtxt = str(gold_text).strip()
        if len(gtxt) > 0:
            gold_ids = _ids(gtxt)
            # don't raise if len>1; we'll use the multi-token path

    # --- option ids (restricted vocab) -----------------------------------
    option_ids, gold_opt_idx = None, None
    if options:
        option_ids = []
        clean_opts = []
        for o in options:
            o_clean = str(o).strip()
            ids = _ids(o_clean)
            option_ids.append(ids)      # NOTE: list[int] PER OPTION (multi-token allowed)
            clean_opts.append(o_clean)
        if gold_option is not None:
            try:
                gold_opt_idx = clean_opts.index(str(gold_option).strip())
            except ValueError:
                raise ValueError("gold_option must be one of options")

    # --- compute per-layer margins ---------------------------------------
    # RAW/TUNED packing identical to your original structure
    out = {"full": {}, "opts": {}}  # will fill with {raw:{...}, tuned:{...}} like before

    # --- FULL (gold over full vocab) -------------------------------------
    def _pack_full(series: str):
        res = {}

        # (A) top1–top2 over full vocab at one position (prompt-only, next-token logits)
        # RAW or TUNED depending on 'series'
        is_tuned = (series == "tuned") and (tuned is not None)
        # full-vocab logits at text, pos=-1
        full = layerwise_logits_for_pos(
            model, tokenizer,
            text=(text or ""),
            outputs=None,
            pos=-1,
            ln_f_mode=ln_f_mode,
            skip_embedding=skip_embedding,
            tuned=(tuned if is_tuned else None),
            option_ids=None                      # <-- FULL VOCAB
        )
        layer_list = full if not is_tuned else full["tuned"]
        # each z is shape [V]; compute top1-top2
        top1_top2 = []
        for z in layer_list:
            if z.numel() >= 2:
                vals, _ = torch.topk(z, k=2)
                top1_top2.append(float((vals[0] - vals[1]).item()))
            else:
                top1_top2.append(float("nan"))
        res["top1_top2_full"] = top1_top2

        # (B) gold_full for multi/single-token gold (we already implemented the stepping-sum path)
        if gold_ids is not None and len(gold_ids) >= 1:
            sums = _per_layer_logit_for_ids(text or "", gold_ids)  # stepping sum
            res["gold_full"] = sums

        return res


    # --- OPTS (MCQ: options-only projection) -----------------------------
    def _per_layer_option_scores(prompt: str, opt_ids_list: list[list[int]]):
        """
        Returns per-layer scores for each option by stepping per sequence.
        shape: [L_layers][K_options] as Python lists
        """
        if not opt_ids_list:
            return []
        # compute per-layer score for each option
        per_opt = [ _per_layer_logit_for_ids(prompt, ids) for ids in opt_ids_list ]  # K x L
        # transpose to [L][K]
        L = len(per_opt[0]) if per_opt and per_opt[0] else 0
        K = len(per_opt)
        per_layer = []
        for li in range(L):
            per_layer.append([ per_opt[k][li] for k in range(K) ])
        return per_layer

    def _pack_opts(series: str):
        res = {}
        if option_ids:
            per_layer_scores = _per_layer_option_scores(text or "", option_ids)  # [L][K]
            # top1-top2 margin per layer
            top1_top2 = []
            gold_marg = []
            for vec in per_layer_scores:
                if len(vec) < 2:
                    top1_top2.append(float("nan"))
                    gold_marg.append(float("nan"))
                    continue
                arr = np.asarray(vec, dtype=float)
                # top1-top2
                idx = np.argsort(-arr)
                top1_top2.append(float(arr[idx[0]] - arr[idx[1]]))
                # gold margin (if we know which option is gold)
                if gold_opt_idx is not None:
                    rivals = np.delete(arr, gold_opt_idx)
                    gold_marg.append(float(arr[gold_opt_idx] - float(rivals.max()) if rivals.size else np.nan))
            res["top1_top2_opts"] = top1_top2
            if gold_opt_idx is not None:
                res["gold_opts"] = gold_marg
        return res

    # assemble output in your familiar format
    out["full"]["raw"] = _pack_full("raw")
    if tuned is not None:
        out["full"]["tuned"] = _pack_full("tuned")
    else:
        out["full"]["tuned"] = None

    out["opts"]["raw"] = _pack_opts("raw") if options else {}
    if tuned is not None and options:
        out["opts"]["tuned"] = _pack_opts("tuned")
    elif options:
        out["opts"]["tuned"] = None

    return out





@torch.no_grad()
@torch.no_grad()
def mcq_alllayer_scores(
    model,
    tokenizer,
    prompt_text: str,
    options: List[str],
    *,
    gold_opt: Optional[str] = None,
    pos: int = -1,                     # -1 => last non-pad (next-token logits)
    ln_f_mode: str = "last_only",      # {"none","last_only","all"}
    skip_embedding: bool = True,
    tuned: Optional["TunedDiag"] = None,
    outputs: Optional[Any] = None,     # optional reuse (not used for multi-token steps)
) -> Dict[str, Any]:
    """
    Compute per-layer MCQ scores by summing per-token logits for each option (supports multi-token).
    Returns:
      {
        "raw":   (scores_per_layer, winners_per_layer, top1_top2_margins, gold_margins),
        "tuned": same or None,
        "outputs": outputs_of_prompt_forward  (for convenience; not reused for steps)
      }
    Notes:
      * scores_per_layer: List[Dict[option_str, float]], one dict per layer (RAW series)
      * winners_per_layer: List[str]
      * top1_top2_margins: List[float]  (NaN if only one option)
      * gold_margins:     List[float]   (empty if gold_opt is None)
    """
    # --------- device / eval ---------
    model.eval()
    device = next(model.parameters()).device

    # --------- tokenize prompt (one forward for convenience/logging) ----
    enc = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if outputs is None:
        outputs = model(**enc, output_hidden_states=True, use_cache=False)
    else:
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("`outputs` has no hidden_states. Re-run with output_hidden_states=True.")

    # --------- utilities -------------
    def _ids(s: str) -> List[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    def _decode_one(tid: int) -> str:
        return tokenizer.decode([tid], clean_up_tokenization_spaces=False)

    # per-sequence (option) per-layer score by stepping tokens
    def _seq_score_per_layer(option_text: str, use_tuned: bool) -> List[float]:
        # split into token ids (may be 1 or many)
        tids = _ids(option_text.strip())
        if not tids:
            return []
        sums = None
        prefix = ""                       # grow as we consume tokens
        for tid in tids:
            # logits for the next token at end-of (prompt + prefix)
            res = layerwise_logits_for_pos(
                model, tokenizer,
                text=(prompt_text + prefix), outputs=None, pos=-1,
                ln_f_mode=ln_f_mode, skip_embedding=skip_embedding,
                tuned=(tuned if use_tuned else None),
                option_ids=[tid]
            )
            vecs = res if (tuned is None or not use_tuned) else res["tuned"]
            # convert list[Tensor([1])] -> list[float] (per layer)
            step_vals = [float(z.view(-1)[0].item()) for z in vecs]
            sums = step_vals if sums is None else [a + b for a, b in zip(sums, step_vals)]
            prefix += _decode_one(tid)    # advance prefix by the just-scored token
        return sums or []

    # --------- prepare options --------
    if not options:
        raise ValueError("`options` must be non-empty.")
    clean_opts = [str(o).strip() for o in options]
    if gold_opt is not None:
        try:
            gold_idx = clean_opts.index(str(gold_opt).strip())
        except ValueError:
            gold_idx = None
    else:
        gold_idx = None

    # --------- compute RAW/TUNED per-layer option scores ---------------
    # Each is a matrix [L][K] (L layers, K options); then we map to dict per layer.
    # RAW
    raw_LK: List[List[float]] = []
    for opt in clean_opts:
        s = _seq_score_per_layer(opt, use_tuned=False)   # length L
        raw_LK.append(s)
    # transpose to [L][K]
    L = len(raw_LK[0]) if raw_LK and raw_LK[0] else 0
    K = len(clean_opts)
    raw_per_layer: List[List[float]] = []
    for li in range(L):
        raw_per_layer.append([raw_LK[k][li] for k in range(K)])

    # TUNED (if provided)
    tuned_per_layer: Optional[List[List[float]]] = None
    if tuned is not None:
        t_LK: List[List[float]] = []
        for opt in clean_opts:
            s = _seq_score_per_layer(opt, use_tuned=True)
            t_LK.append(s)
        tuned_per_layer = [[t_LK[k][li] for k in range(K)] for li in range(L)]

    # --------- pack outputs (RAW) --------------------------------------
    raw_scores:  List[Dict[str, float]] = []
    raw_winners: List[str] = []
    raw_m12:     List[float] = []
    raw_gold:    List[float] = []

    for li in range(L):
        arr = np.asarray(raw_per_layer[li], dtype=float)   # [K]
        # scores dict
        raw_scores.append({clean_opts[j]: float(arr[j]) for j in range(K)})
        # winner + top1-top2
        if K >= 2:
            idx = np.argsort(-arr)
            raw_winners.append(clean_opts[int(idx[0])])
            raw_m12.append(float(arr[idx[0]] - arr[idx[1]]))
        else:
            raw_winners.append(clean_opts[0])
            raw_m12.append(float("nan"))
        # gold margin
        if gold_idx is not None and K >= 2:
            rivals = np.delete(arr, gold_idx)
            raw_gold.append(float(arr[gold_idx] - rivals.max()) if rivals.size else float("nan"))

    # --------- pack outputs (TUNED) ------------------------------------
    tuned_scores:  List[Dict[str, float]] = []
    tuned_winners: List[str] = []
    tuned_m12:     List[float] = []
    tuned_gold:    List[float] = []

    if tuned_per_layer is not None:
        for li in range(L):
            arr = np.asarray(tuned_per_layer[li], dtype=float)   # [K]
            tuned_scores.append({clean_opts[j]: float(arr[j]) for j in range(K)})
            if K >= 2:
                idx = np.argsort(-arr)
                tuned_winners.append(clean_opts[int(idx[0])])
                tuned_m12.append(float(arr[idx[0]] - arr[idx[1]]))
            else:
                tuned_winners.append(clean_opts[0])
                tuned_m12.append(float("nan"))
            if gold_idx is not None and K >= 2:
                rivals = np.delete(arr, gold_idx)
                tuned_gold.append(float(arr[gold_idx] - rivals.max()) if rivals.size else float("nan"))

    return {
        "raw":   (raw_scores, raw_winners, raw_m12, raw_gold),
        "tuned": (tuned_scores, tuned_winners, tuned_m12, tuned_gold) if tuned_per_layer is not None else None,
        "outputs": outputs
    }

def early_decision_layer(
    res: Dict[str, List],
    margin_thresh: float = 0.0,
    use_tuned: bool = False,     # False -> RAW, True -> TUNED (if available)
    use_gold: bool = False,      # False -> top1–top2, True -> gold-margin (needs res[series][3])
    persist_k: int = 1,          # how many consecutive layers must satisfy the condition
    require_final_lock: bool = True,  # if True, winners in the window must match final (last-layer) winner
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Find the earliest layer i at which the model's decision is 'locked in' according to:
      - winner stability for a window of length persist_k
      - margin >= margin_thresh for all layers in that window
      - (optionally) winners in the window equal the final winner (last layer)

    Args:
        res: dict with keys "raw" (and optionally "tuned").
             res[series] is a list/tuple where:
               [1] winners_per_layer: List[Any] of length L
               [2] margins_top1:     List[Optional[float]] of length L  (top1 - top2)
               [3] gold_margins:     List[Optional[float]] of length L  (optional)
        margin_thresh: minimum required margin at each layer in the window
        use_tuned: pick "tuned" series if available, otherwise "raw"
        use_gold: use gold margins if available (res[series][3]); else use top1-top2 margins (res[series][2])
        persist_k: number of consecutive layers to check from i onward
        require_final_lock: if True, winners[j] must equal final winner for all j in the window
        debug: print reasons for rejections (useful for diagnosing)

    Returns:
        dict with:
          - idx: earliest index i meeting criteria
          - winner_final: winner at last layer
          - winners_window: winners[i : i+persist_k]
          - margins_window: margins[i : i+persist_k]
          - margin_at_i: margins[i]
          - series: "raw" or "tuned"
          - metric: "answer" or "top1_top2"
          - persist_k, threshold
        or None if no such layer exists.
    """
    # ---------- choose series ----------
    series = "tuned" if use_tuned and (res.get("tuned") is not None) else "raw"
    if series not in res:
        raise ValueError(f"[early_decision_layer_v2] series '{series}' not found in res keys {list(res.keys())}")

    if not isinstance(res[series], (list, tuple)) or len(res[series]) < 3:
        raise ValueError(f"[early_decision_layer_v2] res[{series}] must be list/tuple with at least 3 elements.")

    winners = res[series][1]
    margins_top1 = res[series][2]
    gold_margins = res[series][3] if len(res[series]) > 3 else None

    # ---------- basic validation ----------
    if not isinstance(winners, list) or not isinstance(margins_top1, list):
        raise ValueError("[early_decision_layer_v2] winners and margins must be lists.")

    L = len(winners)
    if L == 0 or len(margins_top1) != L:
        raise ValueError(f"[early_decision_layer_v2] invalid lengths: L={L}, len(margins_top1)={len(margins_top1)}")

    if use_gold:
        if gold_margins is None or len(gold_margins) != L:
            if debug:
                print("[early_decision_layer_v2] gold margins requested but unavailable or wrong length; returning None.")
            return None
        margins = gold_margins
        metric = "answer"
    else:
        margins = margins_top1
        metric = "top1_top2"

    if persist_k < 1:
        raise ValueError("[early_decision_layer_v2] persist_k must be >= 1")

    # ---------- final winner ----------
    final_w = winners[-1]
    if debug:
        print(f"[EDL] series={series}, metric={metric}, L={L}, final_w={final_w}, "
              f"thresh={margin_thresh}, persist_k={persist_k}, require_final_lock={require_final_lock}")

    # ---------- scan for earliest stable layer ----------
    for i in range(L):
        end = min(i + persist_k, L)
        ok = True
        reasons = []
        # window checks
        for j in range(i, end):
            # margin check
            mj = margins[j]
            if mj is None or (mj < margin_thresh):
                ok = False
                if debug:
                    reasons.append(f"layer {j}: margin {mj} < {margin_thresh}")
                break
            # winner lock check (optional)
            if require_final_lock and winners[j] != final_w:
                ok = False
                if debug:
                    reasons.append(f"layer {j}: winner '{winners[j]}' != final '{final_w}'")
                break

        if ok:
            return {
                "idx": i,
                "winner_final": final_w,
                "winner_at_i": winners[i],
                "winners_window": winners[i:end],
                "margins_window": margins[i:end],
                "margin_at_i": margins[i],
                "series": series,
                "metric": metric,
                "persist_k": persist_k,
                "threshold": margin_thresh,
            }
        elif debug:
            print(f"[EDL] reject i={i}: " + ("; ".join(reasons) if reasons else "no reason logged"))

    # ---------- no layer qualifies ----------
    if debug:
        print("[EDL] no early decision layer found under given settings.")
    return None


