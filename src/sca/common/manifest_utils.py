"""
Manifest read/write utilities — track per-file processing state for resume support.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest CSV. Returns empty DataFrame if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        logger.debug("Loaded manifest %s | rows=%d", path.name, len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load manifest %s: %r — returning empty", path, e)
        return pd.DataFrame()


def save_manifest(
    rows: List[Dict[str, Any]],
    path: str | Path,
    dedup_key: str = "file_name",
    status_priority: Optional[Dict[str, int]] = None,
) -> None:
    """Save manifest rows to CSV, deduplicating by `dedup_key`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    if not df.empty and dedup_key in df.columns and "status" in df.columns:
        default_priority = {
            "success": 5,
            "cleaned": 5,
            "exists_cleaned_complete": 5,
            "exists_cleaned_repaired": 5,
            "exists_jsonl_recovered": 4,
            "success_zero_examples": 3,
            "skipped": 2,
            "failed": 1,
        }
        priority = status_priority or default_priority
        df["_rank"] = df["status"].map(lambda x: priority.get(str(x), 0))
        df = (
            df.sort_values(by=[dedup_key, "_rank"])
            .drop_duplicates(subset=[dedup_key], keep="last")
            .drop(columns=["_rank"])
            .reset_index(drop=True)
        )

    df.to_csv(path, index=False)
    logger.debug("Saved manifest %s | rows=%d", path.name, len(df))


def manifest_to_map(df: pd.DataFrame, key: str = "file_name") -> Dict[str, Dict[str, Any]]:
    """Convert manifest DataFrame to a dict keyed by `key`."""
    if df.empty or key not in df.columns:
        return {}
    return {str(row[key]): row.to_dict() for _, row in df.iterrows()}
