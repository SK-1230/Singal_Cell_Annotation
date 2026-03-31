"""
08_build_marker_kb.py
======================
Phase 1: Build a merged local Marker Knowledge Base (KB) from:

  1. resources/markers/external_marker_kb.jsonl   (manually curated or exported external DB)
  2. data/intermediate/marker_examples_v2.jsonl   (train-derived markers)

Outputs:
  - data/knowledge/train_marker_kb.jsonl     (train-derived only)
  - data/knowledge/merged_marker_kb.jsonl    (all sources merged)
  - data/knowledge/08_build_marker_kb_manifest.json

Usage:
    python -u scripts/data_prep/08_build_marker_kb.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import data_prep_config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

try:
    from sca.data.label_normalization import normalize_label_text, normalize_and_map, init_alias_table
    _ONTOLOGY_AVAILABLE = True
except ImportError:
    _ONTOLOGY_AVAILABLE = False


def _entry_id(source: str, species: str, tissue: str, label: str) -> str:
    raw = f"{source}|{species}|{tissue}|{label}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def load_external_kb(path: Path) -> List[Dict[str, Any]]:
    """Load external_marker_kb.jsonl and convert to merged KB schema."""
    if not path.exists():
        logging.warning("external_marker_kb.jsonl not found: %s", path)
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            label_raw = str(rec.get("cell_type_label", "")).strip()
            label_norm = normalize_label_text(label_raw) if _ONTOLOGY_AVAILABLE else label_raw.lower()
            species = str(rec.get("species", "Homo sapiens")).strip()
            tissue = str(rec.get("tissue_general", "unknown")).strip().lower()
            cl_id = str(rec.get("cell_ontology_id", "") or "").strip() or None
            marker_genes = rec.get("marker_genes", [])
            weight = float(rec.get("marker_weight", 1.0))
            evidence_level = str(rec.get("evidence_level", "unknown")).strip()

            # Try to enrich ontology if not already present
            parent_label: Optional[str] = None
            if _ONTOLOGY_AVAILABLE and label_norm:
                mapped = normalize_and_map(label_norm, tsv_path=cfg.LABEL_ALIASES_TSV)
                if not cl_id:
                    cl_id = mapped.get("cell_ontology_id")
                parent_label = mapped.get("cell_ontology_parent_label")

            entry = {
                "kb_entry_id": _entry_id("external", species, tissue, label_norm or label_raw),
                "source": str(rec.get("source", "external")).strip(),
                "species": species,
                "tissue_general": tissue,
                "cell_type_label": label_norm or label_raw,
                "cell_ontology_id": cl_id,
                "parent_label": parent_label,
                "marker_genes": [str(g).strip() for g in marker_genes if g],
                "weight": round(weight, 4),
                "entry_type": "external",
                "evidence_level": evidence_level,
            }
            entries.append(entry)

    logging.info("Loaded %d external KB entries from %s", len(entries), path.name)
    return entries


def build_train_kb_from_v2_markers(v2_jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Derive KB entries from marker_examples_v2.jsonl (train-derived experience).
    Each cell type in each dataset contributes one entry with its top positive markers.
    """
    if not v2_jsonl_path.exists():
        logging.warning("marker_examples_v2.jsonl not found: %s", v2_jsonl_path)
        return []

    entries = []

    with open(v2_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = str(rec.get("cell_type_clean", "")).strip()
            if not label:
                continue

            species = str(rec.get("organism", "Homo sapiens")).strip()
            tissue = str(rec.get("tissue_general", "unknown")).strip().lower()
            cl_id = rec.get("cell_ontology_id") or None
            parent_label = rec.get("cell_ontology_parent_label") or None
            quality = float(rec.get("marker_quality_score") or 0.0)

            # Extract gene names from positive_markers list
            pos_markers = rec.get("positive_markers", [])
            if isinstance(pos_markers, list):
                genes = [str(m.get("gene", "")).strip() for m in pos_markers if m.get("gene")]
            else:
                genes = []

            if not genes:
                continue

            entry = {
                "kb_entry_id": _entry_id("train_derived", species, tissue, label),
                "source": f"train_derived:{rec.get('dataset_id', 'unknown')}",
                "species": species,
                "tissue_general": tissue,
                "cell_type_label": label,
                "cell_ontology_id": cl_id,
                "parent_label": parent_label,
                "marker_genes": genes,
                "weight": round(min(quality, 1.0), 4),
                "entry_type": "train_derived",
                "evidence_level": "empirical",
            }
            entries.append(entry)

    logging.info("Built %d train-derived KB entries from %s", len(entries), v2_jsonl_path.name)
    return entries


def dedup_merge(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge entries with the same (species, tissue, label).
    External entries take priority; train_derived entries supplement.
    If same label appears multiple times, keep the highest-weight entry.
    """
    # Group by (species, tissue_general, cell_type_label)
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for entry in entries:
        key = (
            str(entry.get("species", "")).lower(),
            str(entry.get("tissue_general", "")).lower(),
            str(entry.get("cell_type_label", "")).lower(),
        )
        groups.setdefault(key, []).append(entry)

    merged = []
    for key, group_entries in groups.items():
        # Sort: external first, then by weight descending
        group_entries.sort(
            key=lambda e: (0 if e.get("entry_type") == "external" else 1, -e.get("weight", 0))
        )
        best = group_entries[0].copy()
        # Accumulate genes from all entries for this key
        all_genes: List[str] = []
        seen_genes: set = set()
        for e in group_entries:
            for g in e.get("marker_genes", []):
                g_upper = g.upper()
                if g_upper not in seen_genes:
                    all_genes.append(g)
                    seen_genes.add(g_upper)
        best["marker_genes"] = all_genes[:20]  # cap at 20
        merged.append(best)

    merged.sort(key=lambda e: (e.get("tissue_general", ""), e.get("cell_type_label", "")))
    return merged


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Wrote %d records to %s", len(records), path)


def main() -> None:
    t0 = time.time()

    if _ONTOLOGY_AVAILABLE and cfg.LABEL_ALIASES_TSV.exists():
        init_alias_table(cfg.LABEL_ALIASES_TSV)

    # Step 1: load external KB
    external_entries = load_external_kb(cfg.EXTERNAL_MARKER_KB_JSONL)

    # Step 2: build train-derived KB from v2 markers
    train_entries = build_train_kb_from_v2_markers(cfg.MARKER_EXAMPLES_V2_JSONL)

    # Step 3: write train_marker_kb.jsonl
    write_jsonl(train_entries, cfg.TRAIN_MARKER_KB_JSONL)

    # Step 4: merge all and write merged_marker_kb.jsonl
    all_entries = external_entries + train_entries
    merged_entries = dedup_merge(all_entries)
    write_jsonl(merged_entries, cfg.MERGED_MARKER_KB_JSONL)

    elapsed = time.time() - t0

    # Write manifest
    manifest = {
        "n_external_entries": len(external_entries),
        "n_train_derived_entries": len(train_entries),
        "n_merged_entries": len(merged_entries),
        "elapsed_sec": round(elapsed, 2),
        "external_kb_path": str(cfg.EXTERNAL_MARKER_KB_JSONL),
        "v2_marker_path": str(cfg.MARKER_EXAMPLES_V2_JSONL),
        "train_kb_path": str(cfg.TRAIN_MARKER_KB_JSONL),
        "merged_kb_path": str(cfg.MERGED_MARKER_KB_JSONL),
    }
    manifest_path = cfg.KNOWLEDGE_DIR / "08_build_marker_kb_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n08 Marker KB Build Complete")
    print(f"  external entries  : {len(external_entries)}")
    print(f"  train entries     : {len(train_entries)}")
    print(f"  merged entries    : {len(merged_entries)}")
    print(f"  merged_kb         : {cfg.MERGED_MARKER_KB_JSONL}")
    print(f"  elapsed           : {elapsed:.2f}s")


if __name__ == "__main__":
    main()

# python -u scripts/data_prep/08_build_marker_kb.py 2>&1 | tee data/meta/08_build_marker_kb.log

# nohup bash -lc 'export PYTHONPATH="$PWD/src:$PYTHONPATH"; python -u scripts/data_prep/08_build_marker_kb.py >> data/meta/08_build_marker_kb.log 2>&1' >/dev/null 2>&1 &  t
# # grep -E "Generated|Building markers|Make markers:" data/meta/08_build_marker_kb.log | tail -10