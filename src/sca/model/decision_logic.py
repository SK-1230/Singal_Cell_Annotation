"""
Decision logic for SCA-Specialist grounded inference.

Merges model output (parsed annotation_output_v2) with retrieval evidence
to produce a final, calibrated decision.

Key functions:
    merge_model_output_and_retrieval_evidence(...)
    decide_accept_review_or_unresolved(...)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default decision thresholds — can be overridden via config or function args.
DEFAULT_ACCEPT_THRESHOLD = 0.80
DEFAULT_REVIEW_THRESHOLD = 0.45
DEFAULT_NOVELTY_THRESHOLD = 0.25


def merge_model_output_and_retrieval_evidence(
    model_output: Dict[str, Any],
    retrieved_candidates: List[Dict[str, Any]],
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Merge model output with retrieval evidence to produce a final grounded result.

    Parameters
    ----------
    model_output : parsed annotation_output_v2 dict from the model
    retrieved_candidates : list of retrieval hits, each with keys:
        label, overlap_score, cell_ontology_id, parent_label, marker_genes
    accept_threshold : confidence score >= this → accept
    review_threshold : confidence score >= this (but < accept) → review
    novelty_threshold : if top retrieval score < this, may flag novel_candidate

    Returns
    -------
    Enriched dict with extra grounded fields:
        retrieved_candidates, top_retrieval_match, retrieval_support_score,
        final_confidence_score, final_decision, ontology_validation_status
    """
    cell_type = (model_output.get("cell_type") or "").strip().lower()
    model_confidence_score = model_output.get("confidence_score", 0.0) or 0.0
    model_novelty_flag = model_output.get("novelty_flag", False)

    # Find best retrieval match
    top_retrieval_score = 0.0
    top_retrieval_match: Optional[Dict[str, Any]] = None
    retrieval_agrees = False

    for cand in retrieved_candidates:
        score = cand.get("overlap_score", 0.0) or 0.0
        if score > top_retrieval_score:
            top_retrieval_score = score
            top_retrieval_match = cand

    if top_retrieval_match is not None:
        cand_label = (top_retrieval_match.get("label") or "").strip().lower()
        retrieval_agrees = (
            cell_type == cand_label
            or cell_type in cand_label
            or cand_label in cell_type
        )

    retrieval_support_score = round(top_retrieval_score, 4)

    # Blend model confidence with retrieval support
    if top_retrieval_score > 0.0:
        if retrieval_agrees:
            # Retrieval agrees: boost score slightly
            blend_weight = 0.75
            final_confidence_score = round(
                blend_weight * model_confidence_score
                + (1 - blend_weight) * min(top_retrieval_score * 1.5, 1.0),
                4,
            )
        else:
            # Retrieval disagrees: temper confidence
            final_confidence_score = round(model_confidence_score * 0.80, 4)
    else:
        # No retrieval support: use model score with penalty
        final_confidence_score = round(model_confidence_score * 0.85, 4)

    final_confidence_score = max(0.0, min(1.0, final_confidence_score))

    # Ontology validation
    ontology_validation_status = _validate_ontology(
        model_output, top_retrieval_match
    )

    # Final decision
    novelty_flag = model_novelty_flag or (
        top_retrieval_score < novelty_threshold and top_retrieval_score > 0.0
    )
    final_decision = decide_accept_review_or_unresolved(
        final_confidence_score=final_confidence_score,
        evidence_support_level=model_output.get("evidence_support_level", "weak"),
        novelty_flag=novelty_flag,
        accept_threshold=accept_threshold,
        review_threshold=review_threshold,
    )

    result = dict(model_output)
    result.update(
        {
            "retrieved_candidates": retrieved_candidates,
            "top_retrieval_match": top_retrieval_match,
            "retrieval_support_score": retrieval_support_score,
            "final_confidence_score": final_confidence_score,
            "final_decision": final_decision,
            "ontology_validation_status": ontology_validation_status,
            "novelty_flag": novelty_flag,
        }
    )
    return result


def decide_accept_review_or_unresolved(
    final_confidence_score: float,
    evidence_support_level: str,
    novelty_flag: bool = False,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> str:
    """
    Determine the final decision label from confidence + evidence signals.

    Returns: 'accept' | 'review' | 'unresolved' | 'novel_candidate'
    """
    if novelty_flag:
        return "novel_candidate"
    if final_confidence_score >= accept_threshold and evidence_support_level in (
        "strong",
        "moderate",
    ):
        return "accept"
    if final_confidence_score >= review_threshold:
        return "review"
    # Low confidence or conflicting/weak evidence
    return "unresolved"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_ontology(
    model_output: Dict[str, Any],
    top_retrieval_match: Optional[Dict[str, Any]],
) -> str:
    """
    Compare model-predicted ontology ID with retrieval evidence.

    Returns one of:
        'matched'       — model CL ID matches best retrieval candidate
        'parent_match'  — model cell_type matches parent_label from retrieval
        'unmatched'     — IDs or labels don't agree
        'missing'       — model output had no ontology ID
        'no_retrieval'  — no retrieval results available
    """
    model_cl_id = (model_output.get("cell_ontology_id") or "").strip()
    model_cell_type = (model_output.get("cell_type") or "").strip().lower()

    if not model_cl_id:
        return "missing"
    if top_retrieval_match is None:
        return "no_retrieval"

    retrieval_cl_id = (top_retrieval_match.get("cell_ontology_id") or "").strip()
    retrieval_label = (top_retrieval_match.get("label") or "").strip().lower()
    retrieval_parent = (top_retrieval_match.get("parent_label") or "").strip().lower()

    if retrieval_cl_id and model_cl_id == retrieval_cl_id:
        return "matched"
    if model_cell_type == retrieval_label:
        return "matched"
    if model_cell_type == retrieval_parent:
        return "parent_match"
    if retrieval_label and model_cell_type in retrieval_label:
        return "parent_match"

    return "unmatched"
