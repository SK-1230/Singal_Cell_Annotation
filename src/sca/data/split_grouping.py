"""
split_grouping.py — group key resolution for study/collection-level split.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def resolve_group_key(record: Dict[str, Any], primary_key: str = "collection_doi",
                      fallback_keys: Optional[List[str]] = None) -> str:
    """
    Resolve a group key from a record dict.

    Priority: primary_key → fallback_keys[0] → ... → dataset_id

    Parameters
    ----------
    record : a single SFT full record dict
    primary_key : preferred key (e.g. "collection_doi")
    fallback_keys : ordered list of fallback keys
    """
    if fallback_keys is None:
        fallback_keys = ["collection_name", "dataset_id"]

    for key in [primary_key] + fallback_keys:
        val = record.get(key)
        if val and str(val).strip() and str(val).strip().lower() not in {"none", "nan", "unknown", ""}:
            return str(val).strip()

    # Last resort: use dataset_id
    return str(record.get("dataset_id", "unknown"))


def build_group_id_map(records: List[Dict[str, Any]],
                       primary_key: str = "collection_doi",
                       fallback_keys: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Build a mapping from record index to group_id for all records.
    """
    return {
        str(i): resolve_group_key(rec, primary_key, fallback_keys)
        for i, rec in enumerate(records)
    }


def get_unique_groups(records: List[Dict[str, Any]],
                      primary_key: str = "collection_doi",
                      fallback_keys: Optional[List[str]] = None) -> List[str]:
    """Return sorted list of unique group IDs from records."""
    groups = {resolve_group_key(rec, primary_key, fallback_keys) for rec in records}
    return sorted(groups)
