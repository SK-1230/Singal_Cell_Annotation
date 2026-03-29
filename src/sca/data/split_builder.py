"""
Benchmark split builder for Phase 2 evaluation.

Provides logic to extract specialized test subsets from a v2 full records set:
    build_test_rare_subset()       — globally rare cell type labels
    build_test_unmapped_subset()   — ontology-unmapped labels
    build_benchmark_manifest()     — CSV summary of all split sizes/stats
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rare-label subset
# ---------------------------------------------------------------------------

def build_test_rare_subset(
    test_full: List[Dict[str, Any]],
    global_records: Optional[List[Dict[str, Any]]] = None,
    rare_max_global_count: int = 3,
) -> List[Dict[str, Any]]:
    """
    Extract test records whose cell_type_clean label is globally rare.

    "Globally rare" = the label appears <= rare_max_global_count times
    across the full record set (train + val + test combined).

    Parameters
    ----------
    test_full       : list of test split records
    global_records  : full combined records (train+val+test) for counting;
                      if None, counts only within test_full
    rare_max_global_count : a label with count <= this is considered rare
    """
    count_source = global_records if global_records is not None else test_full
    label_counts: Counter = Counter(
        str(r.get("cell_type_clean", "unknown")) for r in count_source
    )

    rare_labels = {
        label for label, cnt in label_counts.items() if cnt <= rare_max_global_count
    }

    rare_records = [
        r for r in test_full
        if str(r.get("cell_type_clean", "unknown")) in rare_labels
    ]

    logger.info(
        "Rare test subset: %d rare labels | %d records (from %d test records)",
        len(rare_labels), len(rare_records), len(test_full)
    )
    return rare_records


# ---------------------------------------------------------------------------
# Ontology-unmapped subset
# ---------------------------------------------------------------------------

def build_test_unmapped_subset(
    test_full: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract test records where the label has no Cell Ontology mapping.

    Criteria (any of):
    - cell_ontology_id is None / empty
    - cell_type_status == "unmapped"
    - hardness_flags.ontology_unmapped == True
    """
    unmapped = []
    for r in test_full:
        cl_id = r.get("cell_ontology_id")
        status = r.get("cell_type_status", "")
        flags = r.get("hardness_flags", {})

        is_unmapped = (
            not cl_id
            or status == "unmapped"
            or (isinstance(flags, dict) and flags.get("ontology_unmapped", False))
        )
        if is_unmapped:
            unmapped.append(r)

    logger.info(
        "Unmapped test subset: %d records (from %d test records)",
        len(unmapped), len(test_full)
    )
    return unmapped


# ---------------------------------------------------------------------------
# Benchmark manifest
# ---------------------------------------------------------------------------

def build_benchmark_manifest(
    split_sizes: Dict[str, int],
    split_dataset_counts: Dict[str, int],
    split_extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build benchmark_manifest rows describing each split.

    Parameters
    ----------
    split_sizes         : dict of split_name → n_records
    split_dataset_counts: dict of split_name → n_unique_datasets
    split_extra         : optional additional columns per split

    Returns list of dicts suitable for CSV export.
    """
    rows = []
    for split_name, n_records in split_sizes.items():
        row: Dict[str, Any] = {
            "split_name": split_name,
            "n_records": n_records,
            "n_datasets": split_dataset_counts.get(split_name, 0),
        }
        if split_extra and split_name in split_extra:
            row.update(split_extra[split_name])
        rows.append(row)
    return rows


def write_benchmark_manifest(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("split_name,n_records,n_datasets\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved benchmark_manifest.csv: %d splits → %s", len(rows), path)


# ---------------------------------------------------------------------------
# Convert full records to message-only format
# ---------------------------------------------------------------------------

def extract_messages(records: List[Dict[str, Any]], key: str = "messages") -> List[Dict[str, Any]]:
    return [{"messages": r[key]} for r in records if key in r]


def extract_no_think_messages(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return extract_messages(records, key="messages_no_think")


# ---------------------------------------------------------------------------
# Dataset profile for v2 records
# ---------------------------------------------------------------------------

def build_v2_dataset_profiles(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extended dataset profile for v2 records, adding ontology stats."""
    from collections import defaultdict
    ds_map: Dict[str, List] = defaultdict(list)
    for r in records:
        ds_map[r.get("dataset_id", "unknown")].append(r)

    profiles = []
    for dataset_id, recs in ds_map.items():
        n_mapped = sum(1 for r in recs if r.get("cell_ontology_id"))
        quality_scores = [r.get("marker_quality_score") for r in recs if r.get("marker_quality_score") is not None]
        avg_quality = round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None

        tissues = [r.get("tissue_general", "unknown") for r in recs]
        from collections import Counter as _C
        main_tissue = _C(tissues).most_common(1)[0][0] if tissues else "unknown"

        profiles.append({
            "dataset_id": dataset_id,
            "dataset_title": recs[0].get("dataset_title", ""),
            "main_tissue_general": main_tissue,
            "n_records": len(recs),
            "n_cell_types": len({r.get("cell_type_clean", "") for r in recs}),
            "n_ontology_mapped": n_mapped,
            "ontology_mapped_ratio": round(n_mapped / len(recs), 4) if recs else 0.0,
            "avg_marker_quality_score": avg_quality,
        })

    return sorted(profiles, key=lambda x: (x["main_tissue_general"], -x["n_records"], x["dataset_id"]))
