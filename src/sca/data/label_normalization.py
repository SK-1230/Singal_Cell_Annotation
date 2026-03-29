"""
Label normalization for cell type annotations.

Provides:
- Text normalization (lowercase, strip, collapse whitespace)
- Alias table lookup (raw → canonical + ontology info)
- Label status classification: canonical / alias / unmapped / ambiguous
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Alias table is loaded once per process
_alias_table: Optional[pd.DataFrame] = None
_alias_index: Optional[Dict[str, dict]] = None


def normalize_label_text(raw: str | None) -> str | None:
    """
    Normalize raw cell type text:
    - Strip leading/trailing whitespace
    - Collapse internal whitespace
    - Lowercase

    Returns None for None / empty inputs.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    return s if s else None


def load_alias_table(tsv_path: str | Path) -> pd.DataFrame:
    """
    Load the label_aliases.tsv resource file.
    Expected columns: raw_label, normalized_label, cell_ontology_id, parent_label, notes
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        logger.warning("label_aliases.tsv not found at %s — alias lookup disabled", tsv_path)
        return pd.DataFrame(columns=["raw_label", "normalized_label", "cell_ontology_id", "parent_label", "notes"])

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df = df.fillna("")
    # Normalize the raw_label lookup key
    df["_lookup_key"] = df["raw_label"].apply(lambda x: normalize_label_text(x) or "")
    logger.info("Loaded alias table from %s | %d entries", tsv_path.name, len(df))
    return df


def _build_alias_index(df: pd.DataFrame) -> Dict[str, dict]:
    """Build a fast lookup dict from normalized raw_label → alias row."""
    index: Dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row.get("_lookup_key", "")).strip()
        if key:
            index[key] = {
                "normalized_label": str(row.get("normalized_label", "")).strip(),
                "cell_ontology_id": str(row.get("cell_ontology_id", "")).strip() or None,
                "parent_label": str(row.get("parent_label", "")).strip() or None,
                "notes": str(row.get("notes", "")).strip(),
            }
    return index


def init_alias_table(tsv_path: str | Path) -> None:
    """
    Initialize the module-level alias table. Call once at startup.
    Safe to call multiple times — only loads if not already loaded.
    """
    global _alias_table, _alias_index
    if _alias_table is None:
        _alias_table = load_alias_table(tsv_path)
        _alias_index = _build_alias_index(_alias_table)


def map_label_to_alias_table(label: str | None) -> dict | None:
    """
    Look up a (raw or already-normalized) label in the alias table.

    Returns a dict with keys:
        normalized_label, cell_ontology_id, parent_label, notes
    or None if not found.
    """
    if _alias_index is None:
        logger.debug("Alias table not initialized — call init_alias_table() first")
        return None

    key = normalize_label_text(label)
    if key is None:
        return None

    return _alias_index.get(key)


def classify_label_status(
    label: str | None,
    cell_ontology_id: str | None = None,
    alias_hit: dict | None = None,
    ambiguous_keywords: Optional[List[str]] = None,
) -> str:
    """
    Classify a label as one of:
    - 'canonical'  — already in canonical form and ontology-mapped
    - 'alias'      — recognized alias, mapped to ontology
    - 'unmapped'   — recognized but no ontology ID
    - 'ambiguous'  — matches an ambiguous/generic keyword
    - 'empty'      — None or empty string

    Parameters
    ----------
    label : raw or normalized label text
    cell_ontology_id : CL ID if already resolved
    alias_hit : result dict from map_label_to_alias_table()
    ambiguous_keywords : list of substrings marking ambiguous labels
    """
    default_ambiguous = [
        "unknown", "unassigned", "ambiguous", "doublet",
        "multiplet", "low quality", "low-quality",
        "debris", "artifact", "other",
    ]
    ambiguous_keywords = ambiguous_keywords or default_ambiguous

    if not label:
        return "empty"

    low = label.lower().strip()
    if any(kw in low for kw in ambiguous_keywords):
        return "ambiguous"

    if cell_ontology_id:
        if alias_hit and alias_hit.get("notes", ""):
            return "alias" if "alias" in alias_hit["notes"] else "canonical"
        return "canonical"

    if alias_hit is not None:
        if alias_hit.get("cell_ontology_id"):
            return "alias"
        return "unmapped"

    return "unmapped"


def normalize_and_map(
    raw_label: str | None,
    tsv_path: str | Path | None = None,
) -> dict:
    """
    One-stop normalization + alias table lookup.

    Returns a dict with:
        cell_type_clean      : str | None  — normalized text
        cell_ontology_id     : str | None
        cell_ontology_label  : str | None  — canonical label from alias table
        cell_ontology_parent_label : str | None
        cell_type_status     : str  — canonical/alias/unmapped/ambiguous/empty
        cell_type_level      : str | None  — rough level (leaf / broad), currently basic
    """
    if tsv_path is not None:
        init_alias_table(tsv_path)

    normalized = normalize_label_text(raw_label)
    if normalized is None:
        return {
            "cell_type_clean": None,
            "cell_ontology_id": None,
            "cell_ontology_label": None,
            "cell_ontology_parent_label": None,
            "cell_type_status": "empty",
            "cell_type_level": None,
        }

    alias_hit = map_label_to_alias_table(normalized)

    cell_ontology_id = alias_hit.get("cell_ontology_id") if alias_hit else None
    cell_ontology_label = alias_hit.get("normalized_label") if alias_hit else normalized
    parent_label = alias_hit.get("parent_label") if alias_hit else None

    status = classify_label_status(
        label=normalized,
        cell_ontology_id=cell_ontology_id,
        alias_hit=alias_hit,
    )

    # Simple heuristic level: if parent exists we're a leaf, else broad
    level = "leaf" if parent_label else "broad"

    return {
        "cell_type_clean": normalized,
        "cell_ontology_id": cell_ontology_id or None,
        "cell_ontology_label": cell_ontology_label,
        "cell_ontology_parent_label": parent_label,
        "cell_type_status": status,
        "cell_type_level": level,
    }
