"""
target_labeling.py — build unified target label columns for SFT training.

Functions to derive cell_type_target_id and cell_type_target_label from
existing ontology columns (cell_ontology_label, cell_ontology_id, etc.)
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_target_label_columns(
    obs: pd.DataFrame,
    mode: str = "ontology_label",
    fallback_to_parent: bool = True,
    fallback_to_source_clean: bool = True,
) -> pd.DataFrame:
    """
    Build cell_type_target_label and cell_type_target_id columns.

    Priority:
    1. cell_ontology_label  (if mode == "ontology_label")
    2. cell_ontology_parent_label  (if fallback_to_parent)
    3. cell_type_clean  (fallback)
    4. cell_type_source_clean  (if fallback_to_source_clean, last resort before None)

    Parameters
    ----------
    obs : AnnData.obs DataFrame (must already contain ontology columns)
    mode : "ontology_label" or "clean"
    fallback_to_parent : whether to use parent label when primary is missing
    fallback_to_source_clean : whether to use cell_type_source_clean as last resort
        before allowing a None result. Rescues cell types that have no ontology
        mapping and no parent, but have a valid raw annotation (e.g. "erythroblast").
    """
    obs = obs.copy()

    if mode == "clean":
        obs["cell_type_target_label"] = obs.get("cell_type_clean", pd.Series([None] * len(obs), index=obs.index))
        obs["cell_type_target_id"] = obs.get("cell_ontology_id", pd.Series([None] * len(obs), index=obs.index))
        return obs

    # mode == "ontology_label"
    target_label = obs.get("cell_ontology_label", pd.Series([None] * len(obs), index=obs.index)).copy()
    target_id = obs.get("cell_ontology_id", pd.Series([None] * len(obs), index=obs.index)).copy()

    if fallback_to_parent:
        parent_label = obs.get("cell_ontology_parent_label", pd.Series([None] * len(obs), index=obs.index))
        missing_mask = target_label.isna() | (target_label.astype(str).str.strip() == "")
        target_label = target_label.where(~missing_mask, parent_label)

    # Fallback to cell_type_clean
    clean_col = obs.get("cell_type_clean", pd.Series([None] * len(obs), index=obs.index))
    still_missing = target_label.isna() | (target_label.astype(str).str.strip() == "")
    target_label = target_label.where(~still_missing, clean_col)

    # Last-resort fallback: cell_type_source_clean (raw annotation before cleaning)
    # This rescues cell types that have no ontology mapping and whose cell_type_clean
    # is also empty (e.g. cells labeled "erythroblast" in specific datasets that
    # were not included in the ontology index).
    if fallback_to_source_clean and "cell_type_source_clean" in obs.columns:
        source_clean_col = obs["cell_type_source_clean"]
        still_missing = target_label.isna() | (target_label.astype(str).str.strip() == "")
        target_label = target_label.where(~still_missing, source_clean_col)
        n_rescued = int(still_missing.sum()) - int(
            (target_label.isna() | (target_label.astype(str).str.strip() == "")).sum()
        )
        if n_rescued > 0:
            logger.debug("build_target_label_columns: rescued %d records via source_clean fallback", n_rescued)

    obs["cell_type_target_label"] = target_label
    obs["cell_type_target_id"] = target_id

    n_mapped = int(target_id.notna().sum())
    n_total = len(obs)
    logger.debug(
        "build_target_label_columns: target_id mapped %d/%d (%.1f%%)",
        n_mapped, n_total, 100.0 * n_mapped / n_total if n_total > 0 else 0.0,
    )

    return obs


def compute_target_mapping_stats(obs: pd.DataFrame) -> dict:
    """
    Compute target label statistics for manifest/profile output.
    """
    n_total = len(obs)
    n_target_id_mapped = int(obs["cell_type_target_id"].notna().sum()) if "cell_type_target_id" in obs.columns else 0
    n_target_labels = int(obs["cell_type_target_label"].nunique()) if "cell_type_target_label" in obs.columns else 0
    n_target_ids = int(obs["cell_type_target_id"].dropna().nunique()) if "cell_type_target_id" in obs.columns else 0
    target_mapped_ratio = round(n_target_id_mapped / n_total, 4) if n_total > 0 else 0.0

    return {
        "n_target_id_mapped": n_target_id_mapped,
        "target_mapped_ratio": target_mapped_ratio,
        "n_target_labels": n_target_labels,
        "n_target_ids": n_target_ids,
    }
