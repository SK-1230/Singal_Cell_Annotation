"""
curation_rules.py — dataset-level metadata guardrails.

Functions to check whether a dataset/row passes the Phase-A curation rules
defined in data_prep_config.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def passes_metadata_guardrails(row: pd.Series, cfg) -> bool:
    """
    Return True if the dataset row passes all Phase-A metadata guardrails.

    Parameters
    ----------
    row : a single row from the candidate datasets DataFrame
    cfg : data_prep_config module
    """
    # --- disease keyword exclusion ---
    exclude_disease_kws = getattr(cfg, "EXCLUDE_DISEASE_KEYWORDS", [])
    diseases_str = str(row.get("diseases", "") or "")
    if _contains_any_keyword(diseases_str, exclude_disease_kws):
        return False

    # --- strict normal-only ---
    strict_normal = getattr(cfg, "STRICT_NORMAL_ONLY", False)
    if strict_normal:
        if not diseases_str or diseases_str.strip().lower() not in {"normal", "healthy"}:
            return False

    # --- max n_tissues ---
    max_tissues = getattr(cfg, "MAX_ALLOWED_N_TISSUES", None)
    if max_tissues is not None:
        n_tissues = row.get("n_tissues", 0) or 0
        if int(n_tissues) > max_tissues:
            return False

    # --- max n_diseases ---
    max_diseases = getattr(cfg, "MAX_ALLOWED_N_DISEASES", None)
    if max_diseases is not None:
        n_diseases = row.get("n_diseases", 0) or 0
        if int(n_diseases) > max_diseases:
            return False

    # --- title keyword exclusion ---
    exclude_title_kws = getattr(cfg, "EXCLUDE_TITLE_KEYWORDS", [])
    title_str = str(row.get("dataset_title", "") or "")
    if _contains_any_keyword(title_str, exclude_title_kws):
        return False

    # --- collection keyword exclusion ---
    exclude_coll_kws = getattr(cfg, "EXCLUDE_COLLECTION_KEYWORDS", [])
    coll_str = str(row.get("collection_name", "") or "")
    if _contains_any_keyword(coll_str, exclude_coll_kws):
        return False

    return True


def score_reference_preference(row: pd.Series, cfg) -> float:
    """
    Return a bonus score [0.0, 1.0] if the dataset looks like a reference/atlas.
    """
    prefer_kws = getattr(cfg, "PREFER_REFERENCE_KEYWORDS", [])
    if not prefer_kws:
        return 0.0

    title_str = str(row.get("dataset_title", "") or "")
    coll_str = str(row.get("collection_name", "") or "")
    combined = (title_str + " " + coll_str).lower()

    for kw in prefer_kws:
        if kw.lower() in combined:
            return 1.0

    return 0.0
