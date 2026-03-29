"""
Marker extraction utilities for Phase 1 v2 evidence-aware records.

Provides:
- extract_positive_markers()   : top up-regulated marker genes with stats
- extract_negative_markers()   : low-expression / down-regulated genes
- compute_detection_stats()    : pct_in / pct_out for each gene
- compute_marker_quality_score(): per-label numeric quality
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def is_bad_marker_gene(gene: str, uninformative: set, bad_prefixes: tuple) -> bool:
    g = str(gene).upper()
    if g in uninformative:
        return True
    return any(g.startswith(p) for p in bad_prefixes)


def compute_detection_stats(
    adata: ad.AnnData,
    label_mask: np.ndarray,
    gene_names: List[str],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute pct_in and pct_out for each gene given a boolean mask
    separating the target cluster from all other cells.

    Returns dict: gene → (pct_in, pct_out)
    """
    if adata.n_obs == 0 or len(gene_names) == 0:
        return {}

    gene_idx = [i for i, g in enumerate(adata.var_names) if g in set(gene_names)]
    if not gene_idx:
        return {}

    target_mask = label_mask.astype(bool)
    other_mask = ~target_mask

    X = adata.X
    if hasattr(X, "toarray"):
        # sparse — process gene by gene to avoid dense explosion
        stats: Dict[str, Tuple[float, float]] = {}
        for gi in gene_idx:
            gene = adata.var_names[gi]
            col = np.asarray(X[:, gi].todense()).ravel()
            n_in = int(target_mask.sum())
            n_out = int(other_mask.sum())
            pct_in = float((col[target_mask] > 0).sum() / n_in) if n_in > 0 else 0.0
            pct_out = float((col[other_mask] > 0).sum() / n_out) if n_out > 0 else 0.0
            stats[gene] = (round(pct_in, 4), round(pct_out, 4))
    else:
        stats = {}
        for gi in gene_idx:
            gene = adata.var_names[gi]
            col = X[:, gi]
            n_in = int(target_mask.sum())
            n_out = int(other_mask.sum())
            pct_in = float((col[target_mask] > 0).sum() / n_in) if n_in > 0 else 0.0
            pct_out = float((col[other_mask] > 0).sum() / n_out) if n_out > 0 else 0.0
            stats[gene] = (round(pct_in, 4), round(pct_out, 4))

    return stats


def extract_positive_markers(
    de_df: pd.DataFrame,
    adata: ad.AnnData,
    label_mask: np.ndarray,
    top_k: int = 10,
    uninformative_genes: Optional[set] = None,
    bad_prefixes: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """
    Extract top-k positive (up-regulated) marker genes with full stats.

    Each entry contains:
        gene, rank, logfoldchange, pvals_adj, pct_in, pct_out, score
    """
    uninformative_genes = uninformative_genes or set()
    bad_prefixes = bad_prefixes or ()

    df = de_df[de_df["names"].notna()].copy()
    df["names"] = df["names"].astype(str)
    df = df[~df["names"].apply(lambda g: is_bad_marker_gene(g, uninformative_genes, bad_prefixes))]

    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf) > 0]

    df = df.head(top_k).reset_index(drop=True)
    if df.empty:
        return []

    gene_list = df["names"].tolist()
    det_stats = compute_detection_stats(adata, label_mask, gene_list)

    markers = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        gene = str(row["names"])
        lfc = _safe_float(row.get("logfoldchanges"))
        padj = _safe_float(row.get("pvals_adj"))
        pct_in, pct_out = det_stats.get(gene, (None, None))

        # Composite score: penalize high pct_out, reward high lfc and pct_in
        score: Optional[float] = None
        if lfc is not None and pct_in is not None and pct_out is not None:
            score = round(float(lfc) * float(pct_in) * max(0.0, 1.0 - float(pct_out)), 4)

        markers.append({
            "gene": gene,
            "rank": rank,
            "logfoldchange": lfc,
            "pvals_adj": padj,
            "pct_in": pct_in,
            "pct_out": pct_out,
            "score": score,
        })

    return markers


def extract_negative_markers(
    de_df: pd.DataFrame,
    adata: ad.AnnData,
    label_mask: np.ndarray,
    top_k: int = 5,
    uninformative_genes: Optional[set] = None,
    bad_prefixes: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """
    Extract top-k negative markers — genes expressed in most other clusters
    but absent/low in this cluster.

    Strategy: from the full DE result, take genes ranked near the bottom
    (most negative logFC), as these are down-regulated in this cluster.
    """
    uninformative_genes = uninformative_genes or set()
    bad_prefixes = bad_prefixes or ()

    df = de_df[de_df["names"].notna()].copy()
    df["names"] = df["names"].astype(str)
    df = df[~df["names"].apply(lambda g: is_bad_marker_gene(g, uninformative_genes, bad_prefixes))]

    neg_markers = []

    if "logfoldchanges" in df.columns:
        # Take genes with logFC <= 0, sort ascending (most negative first)
        neg_df = df[df["logfoldchanges"].replace([np.inf, -np.inf], np.nan).fillna(0) <= 0]
        neg_df = neg_df.sort_values("logfoldchanges").head(top_k).reset_index(drop=True)
    else:
        # Fallback: tail of the ranked list
        neg_df = df.tail(top_k).reset_index(drop=True)

    if neg_df.empty:
        return []

    gene_list = neg_df["names"].tolist()
    det_stats = compute_detection_stats(adata, label_mask, gene_list)

    for rank, (_, row) in enumerate(neg_df.iterrows(), start=1):
        gene = str(row["names"])
        pct_in, pct_out = det_stats.get(gene, (None, None))
        lfc = _safe_float(row.get("logfoldchanges"))

        reason = "low_expression"
        if pct_out is not None and pct_in is not None and pct_out > 0.5 and (pct_in or 0) < 0.2:
            reason = "high_pct_out_low_pct_in"
        elif lfc is not None and lfc < -0.5:
            reason = "negative_logfc"

        neg_markers.append({
            "gene": gene,
            "rank": rank,
            "pct_in": pct_in,
            "pct_out": pct_out,
            "reason": reason,
        })

    return neg_markers


def compute_marker_quality_score(
    positive_markers: List[Dict[str, Any]],
    n_cells: int,
    low_cells_threshold: int = 100,
) -> float:
    """
    Compute a [0, 1] marker quality score for a cell type cluster.

    Considers:
    - Number of positive markers (more = better, up to ~10)
    - Mean composite score of top markers
    - Cluster size penalty for small clusters
    """
    if not positive_markers:
        return 0.0

    # Component 1: coverage (how many good markers)
    n_markers = min(len(positive_markers), 10)
    coverage = n_markers / 10.0

    # Component 2: mean composite score
    scores = [m.get("score") for m in positive_markers if m.get("score") is not None]
    if scores:
        mean_score = float(np.mean(scores))
        # Clip and scale to [0, 1]: typical max score ~3
        mean_score_norm = min(mean_score / 3.0, 1.0)
    else:
        mean_score_norm = 0.0

    # Component 3: size bonus/penalty
    if n_cells >= low_cells_threshold:
        size_factor = 1.0
    else:
        size_factor = max(0.3, n_cells / low_cells_threshold)

    quality = round((0.4 * coverage + 0.5 * mean_score_norm + 0.1) * size_factor, 4)
    return min(quality, 1.0)


def safe_mean_topk(series: pd.Series, k: int = 5) -> Optional[float]:
    s = series.head(k).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return None
    val = float(s.mean())
    return None if np.isnan(val) else round(val, 6)


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        f = float(val)
        if np.isinf(f):
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None
