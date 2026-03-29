"""
JSON schema validation utilities for SCA outputs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Required keys for each schema version
MARKER_RECORD_V2_REQUIRED_KEYS = {
    "record_id", "dataset_id", "cell_type_clean", "n_cells",
    "de_method", "positive_markers", "negative_markers",
    "avg_logfc_top5", "marker_quality_score", "hardness_flags",
}

ANNOTATION_OUTPUT_V2_REQUIRED_KEYS = {
    "cell_type", "cell_ontology_id", "parent_cell_type",
    "supporting_markers", "contradictory_markers",
    "confidence_label", "confidence_score",
    "need_manual_review", "decision", "novelty_flag",
    "evidence_support_level", "rationale",
}

ONTOLOGY_INDEX_REQUIRED_KEYS = {
    "cell_ontology_id", "label", "synonyms",
    "parent_label", "parent_id", "organ_scope", "label_level",
}

MARKER_KB_ENTRY_REQUIRED_KEYS = {
    "kb_entry_id", "source", "species", "tissue_general",
    "cell_type_label", "cell_ontology_id", "parent_label",
    "marker_genes", "weight", "entry_type",
}


def validate_record(
    record: Dict[str, Any],
    required_keys: set,
    schema_name: str = "unknown",
) -> Tuple[bool, List[str]]:
    """Check that a record contains all required keys. Returns (is_valid, missing_keys)."""
    missing = [k for k in required_keys if k not in record]
    if missing:
        logger.debug("Schema '%s' validation failed — missing keys: %s", schema_name, missing)
        return False, missing
    return True, []


def validate_marker_record_v2(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    return validate_record(record, MARKER_RECORD_V2_REQUIRED_KEYS, "marker_record_v2")


def validate_annotation_output_v2(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    return validate_record(record, ANNOTATION_OUTPUT_V2_REQUIRED_KEYS, "annotation_output_v2")


def load_json_schema(schema_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load a JSON schema file from disk."""
    path = Path(schema_path)
    if not path.exists():
        logger.warning("Schema file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load schema %s: %r", path, e)
        return None
