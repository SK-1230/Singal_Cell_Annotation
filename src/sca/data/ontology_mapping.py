"""
Ontology mapping utilities — bridge between normalized labels and Cell Ontology IDs.

Designed to be lightweight (no OWL/OBO parser required):
works from the local label_aliases.tsv and optional ontology_index.jsonl.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from sca.data.label_normalization import normalize_label_text, init_alias_table, map_label_to_alias_table

logger = logging.getLogger(__name__)

# Optional: in-memory ontology index loaded from 07_build_ontology_resources output
_ontology_index: Optional[Dict[str, dict]] = None  # keyed by cell_ontology_id
_label_to_id: Optional[Dict[str, str]] = None       # normalized_label → cell_ontology_id


def load_ontology_index(jsonl_path: str | Path) -> None:
    """
    Load the ontology index JSONL built by 07_build_ontology_resources.py.
    Populates module-level lookup dicts.
    """
    global _ontology_index, _label_to_id
    from sca.common.jsonl_utils import read_jsonl

    path = Path(jsonl_path)
    records = read_jsonl(path)
    if not records:
        logger.warning("ontology_index.jsonl is empty or missing: %s", path)
        return

    _ontology_index = {}
    _label_to_id = {}

    for rec in records:
        cl_id = str(rec.get("cell_ontology_id", "")).strip()
        label = normalize_label_text(str(rec.get("label", "")))
        if cl_id:
            _ontology_index[cl_id] = rec
        if label and cl_id:
            _label_to_id[label] = cl_id
        # Also index synonyms
        for syn in rec.get("synonyms", []):
            syn_norm = normalize_label_text(syn)
            if syn_norm and cl_id:
                _label_to_id[syn_norm] = cl_id

    logger.info("Loaded ontology index: %d CL entries, %d label mappings", len(_ontology_index), len(_label_to_id))


def map_normalized_label_to_cl(
    normalized_label: str | None,
    alias_tsv_path: str | Path | None = None,
) -> Optional[str]:
    """
    Map a normalized label text to a Cell Ontology ID.

    Priority:
    1. ontology_index label/synonym lookup (if loaded)
    2. alias table lookup (if tsv_path provided or already initialized)
    3. Returns None if no mapping found
    """
    if normalized_label is None:
        return None

    # Try in-memory ontology index first
    if _label_to_id is not None:
        cl_id = _label_to_id.get(normalized_label)
        if cl_id:
            return cl_id

    # Fall back to alias table
    if alias_tsv_path is not None:
        init_alias_table(alias_tsv_path)

    hit = map_label_to_alias_table(normalized_label)
    if hit:
        return hit.get("cell_ontology_id") or None

    return None


def get_parent_label(
    cl_id: str | None = None,
    normalized_label: str | None = None,
    alias_tsv_path: str | Path | None = None,
) -> Optional[str]:
    """
    Get the parent label for a cell type.

    Tries: ontology_index → alias table.
    """
    if cl_id and _ontology_index is not None:
        rec = _ontology_index.get(cl_id, {})
        parent = rec.get("parent_label")
        if parent:
            return str(parent)

    # Fall back to alias table
    if alias_tsv_path is not None:
        init_alias_table(alias_tsv_path)

    key = normalized_label
    if key is None and cl_id and _ontology_index:
        rec = _ontology_index.get(cl_id, {})
        key = normalize_label_text(rec.get("label", ""))

    hit = map_label_to_alias_table(key)
    if hit:
        return hit.get("parent_label") or None

    return None


def get_label_level(
    cl_id: str | None = None,
    normalized_label: str | None = None,
) -> Optional[str]:
    """
    Return a rough ontology level label: 'leaf', 'intermediate', 'broad', or None.

    Currently derived from:
    - presence of a parent in ontology_index (if loaded)
    - heuristic on label complexity
    """
    if cl_id and _ontology_index is not None:
        rec = _ontology_index.get(cl_id, {})
        level = rec.get("label_level")
        if level:
            return str(level)
        # Heuristic: if it has a parent, it's a leaf-ish
        if rec.get("parent_label"):
            return "leaf"
        return "broad"

    if normalized_label:
        # Heuristic: longer / more specific labels tend to be leaf
        words = normalized_label.split()
        if len(words) >= 3:
            return "leaf"
        if len(words) >= 2:
            return "intermediate"
        return "broad"

    return None


def get_ontology_entry(cl_id: str) -> Optional[dict]:
    """Return the full ontology index entry for a CL ID."""
    if _ontology_index is None:
        return None
    return _ontology_index.get(cl_id)
