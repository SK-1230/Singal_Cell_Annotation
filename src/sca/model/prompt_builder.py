"""
Prompt builder for SCA-Specialist v2.

Provides:
    build_training_user_prompt_v2(rec)
        — for SFT training, uses marker_examples_v2 record
    build_grounded_infer_prompt_v2(query, retrieved_evidence)
        — for grounded inference, includes retrieval results

Both return plain strings suitable for the "user" turn in chat format.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sca.data.marker_features import format_markers_for_prompt


SYSTEM_PROMPT_V2 = (
    "You are a transcriptomics specialist for single-cell RNA-seq cell type annotation. "
    "Given marker evidence for a cluster, identify the most likely cell type. "
    "Always return a valid JSON object matching the required schema exactly. "
    "Be precise, evidence-grounded, and acknowledge uncertainty when evidence is weak."
)


def build_training_user_prompt_v2(
    rec: Dict[str, Any],
    include_stats: bool = True,
    include_negative_markers: bool = True,
    add_no_think_suffix: bool = False,
) -> str:
    """
    Build a v2 user prompt from a marker_examples_v2 record.

    Uses positive marker statistics (logFC, pct_in, pct_out) and optionally
    negative markers. Does NOT include retrieved evidence (that's for inference).

    Parameters
    ----------
    rec : dict from marker_examples_v2.jsonl
    include_stats : whether to include per-marker logFC/pct stats
    include_negative_markers : whether to include absent/low markers section
    add_no_think_suffix : append /no_think for Qwen3 no-think mode
    """
    organism = rec.get("organism", "unknown")
    tissue = rec.get("tissue_general", "unknown")
    disease = rec.get("disease", "unknown")
    n_cells = rec.get("n_cells", 0)

    positive_markers = rec.get("positive_markers", [])
    negative_markers = rec.get("negative_markers", []) if include_negative_markers else []

    pos_block, neg_block = format_markers_for_prompt(
        positive_markers,
        negative_markers,
        top_k_positive=10,
        top_k_negative=3,
        include_stats=include_stats,
    )

    lines = [
        "You are given evidence for a single-cell RNA-seq cluster.",
        "",
        f"Organism: {organism}",
        f"Tissue: {tissue}",
        f"Disease/Context: {disease}",
        f"Cluster size: {n_cells}",
        "",
    ]

    if pos_block:
        lines.append("Positive markers (ranked by evidence strength):")
        lines.append(pos_block)
        lines.append("")

    if neg_block:
        lines.append("Negative / weakly consistent markers:")
        lines.append(neg_block)
        lines.append("")

    lines += [
        "Task:",
        "Identify the cell type and return a valid JSON object with these keys:",
        "cell_type, cell_ontology_id, parent_cell_type, supporting_markers,",
        "contradictory_markers, confidence_label, confidence_score,",
        "need_manual_review, decision, novelty_flag, evidence_support_level, rationale.",
        "",
        "confidence_label must be one of: high, medium, low",
        "decision must be one of: accept, review, unresolved, novel_candidate",
        "evidence_support_level must be one of: strong, moderate, weak, conflicting",
    ]

    prompt = "\n".join(lines)
    if add_no_think_suffix:
        prompt += "\n/no_think"
    return prompt


def build_grounded_infer_prompt_v2(
    query: Dict[str, Any],
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
    add_no_think_suffix: bool = False,
) -> str:
    """
    Build a v2 inference prompt with optional retrieval evidence injected.

    Parameters
    ----------
    query : dict with keys: organism, tissue_general, disease, n_cells,
            positive_markers (list of dicts), negative_markers (list of dicts)
    retrieved_evidence : list of dicts from retrieval.retrieve_candidates()
        each has: label, cell_ontology_id, overlap_score, marker_genes
    add_no_think_suffix : append /no_think for Qwen3 no-think mode
    """
    organism = query.get("organism", "unknown")
    tissue = query.get("tissue_general", "unknown")
    disease = query.get("disease", "unknown")
    n_cells = query.get("n_cells", 0)

    positive_markers = query.get("positive_markers", [])
    negative_markers = query.get("negative_markers", [])

    pos_block, neg_block = format_markers_for_prompt(
        positive_markers,
        negative_markers,
        top_k_positive=10,
        top_k_negative=3,
        include_stats=True,
    )

    lines = [
        "You are given evidence for a single-cell RNA-seq cluster.",
        "",
        f"Organism: {organism}",
        f"Tissue: {tissue}",
        f"Disease/Context: {disease}",
        f"Cluster size: {n_cells}",
        "",
    ]

    if pos_block:
        lines.append("Positive markers (ranked by evidence strength):")
        lines.append(pos_block)
        lines.append("")

    if neg_block:
        lines.append("Negative / weakly consistent markers:")
        lines.append(neg_block)
        lines.append("")

    # Inject retrieved candidates if available
    if retrieved_evidence:
        lines.append("Retrieved candidate evidence (from local knowledge base):")
        for cand in retrieved_evidence[:5]:
            label = cand.get("label", "unknown")
            cl_id = cand.get("cell_ontology_id", "")
            score = cand.get("overlap_score", 0.0)
            overlap_genes = cand.get("marker_genes", [])[:5]
            gene_str = ", ".join(overlap_genes) if overlap_genes else "—"
            cl_str = f" [{cl_id}]" if cl_id else ""
            lines.append(f"- {label}{cl_str} | overlap_score={score:.3f} | markers: {gene_str}")
        lines.append("")

    lines += [
        "Task:",
        "Identify the cell type and return a valid JSON object with these keys:",
        "cell_type, cell_ontology_id, parent_cell_type, supporting_markers,",
        "contradictory_markers, confidence_label, confidence_score,",
        "need_manual_review, decision, novelty_flag, evidence_support_level, rationale.",
        "",
        "confidence_label must be one of: high, medium, low",
        "decision must be one of: accept, review, unresolved, novel_candidate",
        "evidence_support_level must be one of: strong, moderate, weak, conflicting",
    ]

    prompt = "\n".join(lines)
    if add_no_think_suffix:
        prompt += "\n/no_think"
    return prompt
