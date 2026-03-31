"""
07_build_ontology_resources.py
================================
Phase 1: Build a lightweight local ontology index from:
  - resources/ontology/label_aliases.tsv
  - resources/ontology/organ_hierarchy.tsv

Outputs:
  - data/knowledge/ontology_index.jsonl        (runtime lookup)
  - resources/ontology/cell_ontology_min.jsonl (static snapshot)

Usage:
    python -u scripts/data_prep/07_build_ontology_resources.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import data_prep_config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_label_aliases(tsv_path: Path) -> pd.DataFrame:
    if not tsv_path.exists():
        raise FileNotFoundError(f"label_aliases.tsv not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    logging.info("Loaded label_aliases.tsv | %d rows", len(df))
    return df


def load_organ_hierarchy(tsv_path: Path) -> pd.DataFrame:
    if not tsv_path.exists():
        logging.warning("organ_hierarchy.tsv not found: %s — skipping organ scope", tsv_path)
        return pd.DataFrame()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    logging.info("Loaded organ_hierarchy.tsv | %d rows", len(df))
    return df


def build_organ_scope_map(organ_df: pd.DataFrame) -> Dict[str, str]:
    """Build a tissue_specific → organ_general mapping."""
    if organ_df.empty:
        return {}
    scope_map: Dict[str, str] = {}
    for _, row in organ_df.iterrows():
        general = str(row.get("organ_general", "")).strip()
        specific = str(row.get("organ_specific", "")).strip().lower()
        if general and specific:
            scope_map[specific] = general
    return scope_map


def determine_label_level(row: pd.Series) -> str:
    """Heuristic label level from alias row."""
    notes = str(row.get("notes", "")).lower()
    normalized = str(row.get("normalized_label", "")).strip()
    parent = str(row.get("parent_label", "")).strip()

    if not parent:
        return "broad"

    # More words in label → more specific
    word_count = len(normalized.split())
    if word_count >= 3:
        return "leaf"
    if word_count >= 2:
        return "intermediate"
    return "leaf"


def build_ontology_index(
    alias_df: pd.DataFrame,
    organ_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Build ontology_index records, one per unique cell_ontology_id.

    Fields per record:
        cell_ontology_id, label, synonyms, parent_label, parent_id,
        organ_scope, label_level
    """
    organ_scope_map = build_organ_scope_map(organ_df)

    # Group by cell_ontology_id to collect all aliases as synonyms
    cl_groups: Dict[str, Dict[str, Any]] = {}

    for _, row in alias_df.iterrows():
        cl_id = str(row.get("cell_ontology_id", "")).strip()
        if not cl_id:
            continue

        norm_label = str(row.get("normalized_label", "")).strip()
        raw_label = str(row.get("raw_label", "")).strip()
        parent_label = str(row.get("parent_label", "")).strip()
        notes = str(row.get("notes", "")).strip()

        if cl_id not in cl_groups:
            cl_groups[cl_id] = {
                "cell_ontology_id": cl_id,
                "label": norm_label,
                "synonyms": [],
                "parent_label": parent_label or None,
                "parent_id": None,  # Could be enriched from a full OWL export
                "organ_scope": None,
                "label_level": determine_label_level(row),
                "_raw_labels": set(),
            }
        else:
            # Additional rows for same CL ID are synonyms
            if raw_label and raw_label not in cl_groups[cl_id]["_raw_labels"]:
                cl_groups[cl_id]["synonyms"].append(raw_label)
                cl_groups[cl_id]["_raw_labels"].add(raw_label)

            # Prefer more informative parent
            if parent_label and not cl_groups[cl_id]["parent_label"]:
                cl_groups[cl_id]["parent_label"] = parent_label

        cl_groups[cl_id]["_raw_labels"].add(raw_label)

    # Resolve organ_scope from parent_label
    for cl_id, entry in cl_groups.items():
        parent = str(entry.get("parent_label", "") or "").lower()
        scope = organ_scope_map.get(parent)
        if not scope and parent:
            # Try direct label
            scope = organ_scope_map.get(entry["label"].lower())
        entry["organ_scope"] = scope

    # Cleanup internal fields
    records = []
    for entry in cl_groups.values():
        entry.pop("_raw_labels", None)
        entry["synonyms"] = sorted(set(entry["synonyms"]))
        records.append(entry)

    # Sort by CL ID for determinism
    records.sort(key=lambda r: r["cell_ontology_id"])
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Wrote %d records to %s", len(records), path)


def write_manifest(records: List[Dict[str, Any]], path: Path, elapsed: float) -> None:
    summary = {
        "n_ontology_entries": len(records),
        "n_with_parent": sum(1 for r in records if r.get("parent_label")),
        "n_with_organ_scope": sum(1 for r in records if r.get("organ_scope")),
        "n_leaf_level": sum(1 for r in records if r.get("label_level") == "leaf"),
        "elapsed_sec": round(elapsed, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved manifest: %s", path)


def main() -> None:
    t0 = time.time()

    alias_df = load_label_aliases(cfg.LABEL_ALIASES_TSV)
    organ_df = load_organ_hierarchy(cfg.ORGAN_HIERARCHY_TSV)

    records = build_ontology_index(alias_df, organ_df)
    logging.info("Built ontology index: %d unique CL entries", len(records))

    # Write runtime index
    write_jsonl(records, cfg.ONTOLOGY_INDEX_JSONL)

    # Write static snapshot
    write_jsonl(records, cfg.CELL_ONTOLOGY_MIN_JSONL)

    # Write manifest
    manifest_path = cfg.KNOWLEDGE_DIR / "07_build_ontology_resources_manifest.json"
    elapsed = time.time() - t0
    write_manifest(records, manifest_path, elapsed)

    print(f"\n07 Ontology Resources Build Complete")
    print(f"  n_entries       : {len(records)}")
    print(f"  ontology_index  : {cfg.ONTOLOGY_INDEX_JSONL}")
    print(f"  cell_onto_min   : {cfg.CELL_ONTOLOGY_MIN_JSONL}")
    print(f"  elapsed         : {elapsed:.2f}s")


if __name__ == "__main__":
    main()

# python -u scripts/data_prep/07_build_ontology_resources.py 2>&1 | tee data/meta/07_build_ontology_resources.log


# nohup bash -lc 'export PYTHONPATH="$PWD/src:$PYTHONPATH"; python -u scripts/data_prep/07_build_ontology_resources.py >> data/meta/07_build_ontology_resources.log 2>&1' >/dev/null 2>&1 &
# grep -E "Generated|Building markers|Make markers:" data/meta/07_build_ontology_resources.log | tail -10