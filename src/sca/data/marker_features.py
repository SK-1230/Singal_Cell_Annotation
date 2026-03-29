"""
Marker feature engineering utilities.

Converts raw marker extraction results (positive_markers list from marker_extraction.py)
into compact summary statistics suitable for:
- SFT prompt construction
- Evidence-aware confidence estimation
- Grounded inference context assembly

Key functions:
    summarize_positive_markers()   — derive aggregate stats from positive_markers list
    summarize_negative_markers()   — brief summary of negative markers
    build_marker_feature_vector()  — flat dict of numeric features for calibration
    format_markers_for_prompt()    — human-readable text block for LLM prompts
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def summarize_positive_markers(
    positive_markers: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute aggregate statistics from a positive_markers list.

    Returns a dict with keys:
        n_markers           : int — total number of positive markers
        top_genes           : List[str] — gene names, top_k
        mean_logfc_top5     : float | None
        mean_pct_in_top5    : float | None
        mean_pct_out_top5   : float | None
        mean_score_top5     : float | None
        max_logfc           : float | None
        min_padj            : float | None — smallest (best) adjusted p-value
        specificity_score   : float | None — mean(pct_in - pct_out) for top markers
    """
    if not positive_markers:
        return {
            "n_markers": 0,
            "top_genes": [],
            "mean_logfc_top5": None,
            "mean_pct_in_top5": None,
            "mean_pct_out_top5": None,
            "mean_score_top5": None,
            "max_logfc": None,
            "min_padj": None,
            "specificity_score": None,
        }

    top = positive_markers[:top_k]
    top_genes = [m["gene"] for m in top if m.get("gene")]

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 6) if clean else None

    def _max(vals: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in vals if v is not None]
        return round(max(clean), 6) if clean else None

    def _min(vals: List[Optional[float]]) -> Optional[float]:
        clean = [v for v in vals if v is not None]
        return round(min(clean), 6) if clean else None

    logfcs = [m.get("logfoldchange") for m in top]
    pct_ins = [m.get("pct_in") for m in top]
    pct_outs = [m.get("pct_out") for m in top]
    scores = [m.get("score") for m in top]
    padjs = [m.get("pvals_adj") for m in positive_markers]  # use all, not just top

    # Specificity: mean(pct_in - pct_out) for top markers
    spec_vals = [
        (pi - po)
        for pi, po in zip(pct_ins, pct_outs)
        if pi is not None and po is not None
    ]
    specificity = round(sum(spec_vals) / len(spec_vals), 4) if spec_vals else None

    return {
        "n_markers": len(positive_markers),
        "top_genes": top_genes,
        "mean_logfc_top5": _mean(logfcs),
        "mean_pct_in_top5": _mean(pct_ins),
        "mean_pct_out_top5": _mean(pct_outs),
        "mean_score_top5": _mean(scores),
        "max_logfc": _max([m.get("logfoldchange") for m in positive_markers]),
        "min_padj": _min([p for p in padjs if p is not None and p > 0]),
        "specificity_score": specificity,
    }


def summarize_negative_markers(
    negative_markers: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Compact summary of negative/absent markers.

    Returns:
        n_negative_markers  : int
        top_negative_genes  : List[str]
        reasons             : List[str]
    """
    if not negative_markers:
        return {"n_negative_markers": 0, "top_negative_genes": [], "reasons": []}

    top = negative_markers[:top_k]
    return {
        "n_negative_markers": len(negative_markers),
        "top_negative_genes": [m["gene"] for m in top if m.get("gene")],
        "reasons": list({m.get("reason", "") for m in top if m.get("reason")}),
    }


# ---------------------------------------------------------------------------
# Feature vector for calibration / downstream ML
# ---------------------------------------------------------------------------

def build_marker_feature_vector(
    positive_markers: List[Dict[str, Any]],
    negative_markers: List[Dict[str, Any]],
    n_cells: int,
    marker_quality_score: Optional[float],
    hardness_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Build a flat numeric feature dict for downstream calibration models.

    Features:
        n_positive_markers
        n_negative_markers
        n_cells
        log_n_cells              — log(n_cells + 1)
        mean_logfc_top5
        mean_pct_in_top5
        mean_pct_out_top5
        mean_score_top5
        specificity_score
        marker_quality_score
        flag_low_cells           — 0/1
        flag_low_quality         — 0/1
        flag_ontology_unmapped   — 0/1
        flag_rare_label          — 0/1
    """
    pos_summary = summarize_positive_markers(positive_markers)
    hardness_flags = hardness_flags or {}

    return {
        "n_positive_markers": pos_summary["n_markers"],
        "n_negative_markers": len(negative_markers),
        "n_cells": n_cells,
        "log_n_cells": round(math.log(n_cells + 1), 4),
        "mean_logfc_top5": pos_summary["mean_logfc_top5"],
        "mean_pct_in_top5": pos_summary["mean_pct_in_top5"],
        "mean_pct_out_top5": pos_summary["mean_pct_out_top5"],
        "mean_score_top5": pos_summary["mean_score_top5"],
        "specificity_score": pos_summary["specificity_score"],
        "marker_quality_score": marker_quality_score,
        "flag_low_cells": int(bool(hardness_flags.get("low_cells", False))),
        "flag_low_quality": int(bool(hardness_flags.get("low_marker_quality", False))),
        "flag_ontology_unmapped": int(bool(hardness_flags.get("ontology_unmapped", False))),
        "flag_rare_label": int(bool(hardness_flags.get("rare_label_in_dataset", False))),
    }


# ---------------------------------------------------------------------------
# Prompt text formatting
# ---------------------------------------------------------------------------

def format_markers_for_prompt(
    positive_markers: List[Dict[str, Any]],
    negative_markers: Optional[List[Dict[str, Any]]] = None,
    top_k_positive: int = 10,
    top_k_negative: int = 3,
    include_stats: bool = True,
) -> Tuple[str, str]:
    """
    Format marker lists into text blocks for LLM prompts.

    Returns:
        (positive_block, negative_block) — two strings ready to embed in a prompt.

    positive_block example (include_stats=True):
        1. EPCAM (logFC=2.10, pct_in=0.91, pct_out=0.12)
        2. KRT8  (logFC=1.80, pct_in=0.88, pct_out=0.15)
        ...

    positive_block example (include_stats=False):
        EPCAM, KRT8, KRT18, CDH1, MUC1

    negative_block example:
        PTPRC, LST1, CD14
    """
    pos_lines = []
    for m in positive_markers[:top_k_positive]:
        gene = m.get("gene", "?")
        if include_stats:
            lfc = m.get("logfoldchange")
            pct_in = m.get("pct_in")
            pct_out = m.get("pct_out")
            parts = []
            if lfc is not None:
                parts.append(f"logFC={lfc:.2f}")
            if pct_in is not None:
                parts.append(f"pct_in={pct_in:.2f}")
            if pct_out is not None:
                parts.append(f"pct_out={pct_out:.2f}")
            stat_str = f" ({', '.join(parts)})" if parts else ""
            rank = m.get("rank", len(pos_lines) + 1)
            pos_lines.append(f"{rank}. {gene}{stat_str}")
        else:
            pos_lines.append(gene)

    if include_stats:
        positive_block = "\n".join(pos_lines)
    else:
        positive_block = ", ".join(pos_lines)

    neg_block_genes = []
    for m in (negative_markers or [])[:top_k_negative]:
        gene = m.get("gene", "?")
        neg_block_genes.append(gene)
    negative_block = ", ".join(neg_block_genes)

    return positive_block, negative_block
