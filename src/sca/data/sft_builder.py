"""
SFT sample builder for annotation_output_v2.

Converts marker_examples_v2 records into training messages (chat format)
with the richer v2 output schema.

Main entry point:
    build_sft_record_v2(rec, system_prompt, ...)
        → {dataset_id, cell_type_clean, messages, messages_no_think, ...}

Also provides:
    build_assistant_answer_v2(rec)  — construct ground-truth JSON answer
    build_distill_record(rec, ...)  — distillation-ready format
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sca.data.marker_features import summarize_positive_markers, build_marker_feature_vector
from sca.knowledge.evidence_scoring import (
    compute_evidence_support_level,
    compute_confidence_score,
    classify_confidence_label,
    decide_action,
)
from sca.model.prompt_builder import build_training_user_prompt_v2, SYSTEM_PROMPT_V2


# ---------------------------------------------------------------------------
# Confidence score heuristic from v2 marker record
# ---------------------------------------------------------------------------

def _heuristic_confidence_score(rec: Dict[str, Any]) -> float:
    """
    Derive a heuristic confidence score from the v2 marker record fields.
    Used when building training labels — not a model output.
    """
    quality = rec.get("marker_quality_score") or 0.0
    n_cells = rec.get("n_cells", 0)
    hardness = rec.get("hardness_flags", {})

    pos_markers = rec.get("positive_markers", [])
    pos_summary = summarize_positive_markers(pos_markers)
    n_pos = pos_summary["n_markers"]

    # Build a rough overlap score from quality and marker count
    coverage = min(n_pos / 10.0, 1.0)
    overlap_proxy = quality * 0.6 + coverage * 0.4

    score = compute_confidence_score(
        overlap_score=overlap_proxy,
        marker_quality_score=quality,
        n_cells=n_cells,
        low_cells_threshold=100,
    )

    # Penalize for hardness flags
    if hardness.get("low_cells"):
        score *= 0.8
    if hardness.get("ontology_unmapped"):
        score *= 0.9
    if hardness.get("rare_label_in_dataset"):
        score *= 0.85

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Build ground-truth assistant answer
# ---------------------------------------------------------------------------

def build_assistant_answer_v2(
    rec: Dict[str, Any],
    with_empty_think: bool = False,
) -> str:
    """
    Construct the ground-truth assistant JSON answer in annotation_output_v2 format.

    The answer is fully deterministic — derived from the marker record fields
    using heuristic rules. This becomes the training target.
    """
    cell_type = rec.get("cell_type_clean", "unknown")
    cell_ontology_id = rec.get("cell_ontology_id") or None
    parent_cell_type = rec.get("cell_ontology_parent_label") or None

    pos_markers = rec.get("positive_markers", [])
    supporting_markers = [m["gene"] for m in pos_markers[:4] if m.get("gene")]

    neg_markers = rec.get("negative_markers", [])
    contradictory_markers = [m["gene"] for m in neg_markers[:2] if m.get("gene")]

    confidence_score = _heuristic_confidence_score(rec)
    confidence_label = classify_confidence_label(confidence_score)

    pos_summary = summarize_positive_markers(pos_markers)
    n_pos = pos_summary["n_markers"]
    quality = rec.get("marker_quality_score") or 0.0
    overlap_proxy = min(n_pos / 10.0, 1.0) * 0.5 + quality * 0.5

    evidence_support_level = compute_evidence_support_level(
        overlap_score=overlap_proxy,
        n_positive_markers=n_pos,
        marker_quality_score=quality,
    )

    hardness = rec.get("hardness_flags", {})
    novelty_flag = bool(
        hardness.get("ontology_unmapped") and hardness.get("rare_label_in_dataset")
    )

    decision = decide_action(confidence_label, novelty_flag, evidence_support_level)
    need_manual_review = decision in ("review", "unresolved", "novel_candidate")

    # Build rationale
    top_gene_names = ", ".join(supporting_markers) if supporting_markers else "none"
    rationale = (
        f"The marker profile is most consistent with {cell_type}. "
        f"Top supporting markers: {top_gene_names}. "
        f"Marker quality score: {quality:.2f}. "
        f"Confidence is {confidence_label} based on {n_pos} positive markers "
        f"and cluster size of {rec.get('n_cells', 0)} cells."
    )
    if cell_ontology_id:
        rationale += f" Cell Ontology ID: {cell_ontology_id}."

    answer_dict: Dict[str, Any] = {
        "cell_type": cell_type,
        "cell_ontology_id": cell_ontology_id,
        "parent_cell_type": parent_cell_type,
        "supporting_markers": supporting_markers,
        "contradictory_markers": contradictory_markers,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "need_manual_review": need_manual_review,
        "decision": decision,
        "novelty_flag": novelty_flag,
        "evidence_support_level": evidence_support_level,
        "rationale": rationale,
    }

    answer_text = json.dumps(answer_dict, ensure_ascii=False)
    if with_empty_think:
        return "<think>\n\n</think>\n\n" + answer_text
    return answer_text


# ---------------------------------------------------------------------------
# Build full SFT record
# ---------------------------------------------------------------------------

def build_sft_record_v2(
    rec: Dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT_V2,
) -> Dict[str, Any]:
    """
    Build a full SFT record (both standard and no-think variants) from a
    marker_examples_v2 record.

    Returns a dict containing:
        dataset_id, cell_type_clean, cell_ontology_id, n_cells,
        confidence_label, confidence_score, evidence_support_level, decision,
        marker_quality_score, hardness_flags,
        messages             — standard chat format
        messages_no_think    — with /no_think and empty <think> block
    """
    user_std = build_training_user_prompt_v2(rec, add_no_think_suffix=False)
    user_no_think = build_training_user_prompt_v2(rec, add_no_think_suffix=True)

    assistant_std = build_assistant_answer_v2(rec, with_empty_think=False)
    assistant_no_think = build_assistant_answer_v2(rec, with_empty_think=True)

    # Parse back to get structured fields for the manifest
    import json as _json
    try:
        answer_obj = _json.loads(assistant_std)
    except Exception:
        answer_obj = {}

    messages_std = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_std},
        {"role": "assistant", "content": assistant_std},
    ]
    messages_no_think = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_no_think},
        {"role": "assistant", "content": assistant_no_think},
    ]

    return {
        "dataset_id": rec.get("dataset_id", "unknown"),
        "dataset_title": rec.get("dataset_title", ""),
        "organism": rec.get("organism", "unknown"),
        "tissue_general": rec.get("tissue_general", "unknown"),
        "tissue": rec.get("tissue", "unknown"),
        "disease": rec.get("disease", "unknown"),
        "cell_type_clean": rec.get("cell_type_clean", "unknown"),
        "cell_ontology_id": rec.get("cell_ontology_id"),
        "cell_ontology_label": rec.get("cell_ontology_label"),
        "cell_ontology_parent_label": rec.get("cell_ontology_parent_label"),
        "n_cells": rec.get("n_cells", 0),
        "de_method": rec.get("de_method", "unknown"),
        "marker_quality_score": rec.get("marker_quality_score"),
        "hardness_flags": rec.get("hardness_flags", {}),
        "confidence_label": answer_obj.get("confidence_label"),
        "confidence_score": answer_obj.get("confidence_score"),
        "evidence_support_level": answer_obj.get("evidence_support_level"),
        "decision": answer_obj.get("decision"),
        "novelty_flag": answer_obj.get("novelty_flag", False),
        "messages": messages_std,
        "messages_no_think": messages_no_think,
    }


# ---------------------------------------------------------------------------
# Distillation record
# ---------------------------------------------------------------------------

def build_distill_record(
    rec: Dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT_V2,
    teacher_rationale: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a distillation-ready record for teacher-student training.

    Fields:
        input_prompt       — the user prompt (standard, no /no_think)
        target_json        — ground-truth answer as JSON string
        teacher_rationale  — optional rationale string (empty if not provided)
        schema_version     — "annotation_output_v2"
        metadata           — lightweight context
    """
    user_prompt = build_training_user_prompt_v2(rec, add_no_think_suffix=False)
    target_json = build_assistant_answer_v2(rec, with_empty_think=False)

    return {
        "input_prompt": user_prompt,
        "target_json": target_json,
        "teacher_rationale": teacher_rationale or "",
        "schema_version": "annotation_output_v2",
        "metadata": {
            "dataset_id": rec.get("dataset_id"),
            "cell_type_clean": rec.get("cell_type_clean"),
            "cell_ontology_id": rec.get("cell_ontology_id"),
            "n_cells": rec.get("n_cells"),
            "marker_quality_score": rec.get("marker_quality_score"),
        },
    }
