"""
Evidence scoring utilities — quantify how well markers support a candidate cell type.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def compute_overlap_score(
    query_genes: List[str],
    kb_genes: List[str],
    top_k_weight: int = 5,
) -> float:
    """
    Compute a weighted gene overlap score between query markers and KB entry genes.

    Strategy:
    - Jaccard overlap over full gene sets
    - Bonus for overlap in top-ranked query genes (positional weight)

    Returns float in [0, 1].
    """
    if not query_genes or not kb_genes:
        return 0.0

    q_set = {g.upper() for g in query_genes}
    kb_set = {g.upper() for g in kb_genes}

    intersection = q_set & kb_set
    union = q_set | kb_set
    jaccard = len(intersection) / len(union) if union else 0.0

    # Weighted bonus: top-ranked query genes (by list order) count more
    top_query = {g.upper() for g in query_genes[:top_k_weight]}
    top_overlap = top_query & kb_set
    top_bonus = len(top_overlap) / max(len(top_query), 1) * 0.3

    score = min(jaccard + top_bonus, 1.0)
    return round(score, 4)


def compute_tissue_compatibility_score(
    query_tissue: Optional[str],
    kb_tissues: Optional[List[str]],
) -> float:
    """
    Return a tissue compatibility bonus in [0, 0.2].

    0.2 if the query tissue exactly matches a KB tissue.
    0.1 if there's a partial string match.
    0.0 if no match or information unavailable.
    """
    if not query_tissue or not kb_tissues:
        return 0.0

    q = query_tissue.strip().lower()
    for t in kb_tissues:
        t_norm = (t or "").strip().lower()
        if q == t_norm:
            return 0.2
        if q in t_norm or t_norm in q:
            return 0.1
    return 0.0


def compute_evidence_support_level(
    overlap_score: float,
    n_positive_markers: int,
    marker_quality_score: Optional[float] = None,
) -> str:
    """
    Derive a qualitative evidence support level from numeric signals.

    Returns: 'strong' | 'moderate' | 'weak' | 'conflicting'
    """
    if overlap_score >= 0.4 and n_positive_markers >= 5:
        base = "strong"
    elif overlap_score >= 0.2 or n_positive_markers >= 3:
        base = "moderate"
    elif overlap_score > 0.0 or n_positive_markers >= 1:
        base = "weak"
    else:
        base = "conflicting"

    # Downgrade if marker quality is very low
    if marker_quality_score is not None and marker_quality_score < 0.2:
        if base == "strong":
            base = "moderate"
        elif base == "moderate":
            base = "weak"

    return base


def compute_confidence_score(
    overlap_score: float,
    marker_quality_score: Optional[float],
    n_cells: int,
    low_cells_threshold: int = 100,
) -> float:
    """
    Compute a numeric confidence score in [0, 1].

    This is a heuristic calibration score, not a model probability.
    """
    base = overlap_score * 0.5 + (marker_quality_score or 0.0) * 0.4
    size_factor = min(1.0, n_cells / low_cells_threshold) if low_cells_threshold > 0 else 1.0
    score = base * (0.7 + 0.3 * size_factor)
    return round(min(score, 1.0), 4)


def classify_confidence_label(confidence_score: float) -> str:
    if confidence_score >= 0.7:
        return "high"
    if confidence_score >= 0.4:
        return "medium"
    return "low"


def decide_action(
    confidence_label: str,
    novelty_flag: bool,
    evidence_support_level: str,
) -> str:
    """
    Recommend an action based on confidence + evidence signals.

    Returns: 'accept' | 'review' | 'unresolved' | 'novel_candidate'
    """
    if novelty_flag:
        return "novel_candidate"
    if confidence_label == "high" and evidence_support_level in ("strong", "moderate"):
        return "accept"
    if confidence_label == "low" or evidence_support_level == "conflicting":
        return "unresolved"
    return "review"


def build_annotation_output_v2(
    cell_type: str,
    cell_ontology_id: Optional[str],
    parent_cell_type: Optional[str],
    supporting_markers: List[str],
    contradictory_markers: List[str],
    overlap_score: float,
    marker_quality_score: Optional[float],
    n_cells: int,
    rationale: str,
    novelty_flag: bool = False,
    low_cells_threshold: int = 100,
) -> Dict[str, Any]:
    """
    Build a full annotation_output_v2 dict from component signals.
    """
    confidence_score = compute_confidence_score(
        overlap_score, marker_quality_score, n_cells, low_cells_threshold
    )
    confidence_label = classify_confidence_label(confidence_score)
    evidence_support_level = compute_evidence_support_level(
        overlap_score, len(supporting_markers), marker_quality_score
    )
    decision = decide_action(confidence_label, novelty_flag, evidence_support_level)

    return {
        "cell_type": cell_type,
        "cell_ontology_id": cell_ontology_id,
        "parent_cell_type": parent_cell_type,
        "supporting_markers": supporting_markers,
        "contradictory_markers": contradictory_markers,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "need_manual_review": decision in ("review", "unresolved", "novel_candidate"),
        "decision": decision,
        "novelty_flag": novelty_flag,
        "evidence_support_level": evidence_support_level,
        "rationale": rationale,
    }
