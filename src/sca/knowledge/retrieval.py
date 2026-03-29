"""
Evidence retrieval for grounded inference.

Given a query (marker gene list + context), retrieves relevant KB entries
and ranks candidate cell types by evidence overlap.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def retrieve_candidates(
    query_genes: List[str],
    tissue_general: Optional[str] = None,
    species: str = "Homo sapiens",
    top_k: int = 5,
    kb=None,  # MarkerKB instance
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k candidate cell types from the marker KB
    ranked by gene overlap score.

    Returns list of dicts:
        label, cell_ontology_id, overlap_score, marker_genes, source
    """
    if kb is None:
        from sca.knowledge.marker_kb import get_default_kb
        kb = get_default_kb()

    if not kb.is_loaded:
        logger.warning("MarkerKB not loaded — retrieval returning empty")
        return []

    seen_labels: set = set()
    candidates: List[Dict[str, Any]] = []

    for label in kb.all_labels():
        if label in seen_labels:
            continue
        seen_labels.add(label)

        score = kb.score_gene_list(query_genes, label, species, tissue_general)
        if score == 0.0:
            continue

        entries = kb.query_by_label(label, species, tissue_general)
        if not entries:
            continue

        best_entry = max(entries, key=lambda e: e.get("weight", 1.0))
        candidates.append({
            "label": label,
            "cell_ontology_id": best_entry.get("cell_ontology_id"),
            "parent_label": best_entry.get("parent_label"),
            "overlap_score": score,
            "marker_genes": best_entry.get("marker_genes", []),
            "source": best_entry.get("source", "unknown"),
            "evidence_level": best_entry.get("evidence_level", "unknown"),
        })

    candidates.sort(key=lambda x: x["overlap_score"], reverse=True)
    return candidates[:top_k]


def retrieve_candidate_cell_types(
    query_markers: List[str],
    tissue: Optional[str] = None,
    top_k: int = 5,
    kb=None,
) -> List[Dict[str, Any]]:
    """
    Alias for retrieve_candidates with spec-aligned parameter names.

    Parameters
    ----------
    query_markers : list of marker gene names (ranked by importance)
    tissue : tissue_general string for tissue-match bonus
    top_k : number of top candidates to return
    kb : optional MarkerKB instance; loads default if None
    """
    return retrieve_candidates(
        query_genes=query_markers,
        tissue_general=tissue,
        top_k=top_k,
        kb=kb,
    )
