"""
Local Marker Knowledge Base (KB) — for evidence retrieval during inference.

Supports:
- Loading merged_marker_kb.jsonl
- Querying by cell type label + tissue + species
- Simple overlap-based retrieval scoring
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MarkerKB:
    """
    In-memory marker knowledge base.

    Each entry has:
        kb_entry_id, source, species, tissue_general, cell_type_label,
        cell_ontology_id, parent_label, marker_genes, weight, entry_type
    """

    def __init__(self) -> None:
        self._entries: List[dict] = []
        self._index_by_label: Dict[str, List[int]] = {}   # normalized_label → list of entry indices
        self._index_by_cl_id: Dict[str, List[int]] = {}   # cl_id → list of entry indices
        self._loaded = False

    def load(self, jsonl_path: str | Path) -> None:
        from sca.common.jsonl_utils import read_jsonl
        from sca.data.label_normalization import normalize_label_text

        path = Path(jsonl_path)
        records = read_jsonl(path)
        if not records:
            logger.warning("merged_marker_kb.jsonl empty or missing: %s", path)
            return

        self._entries = records
        self._index_by_label = {}
        self._index_by_cl_id = {}

        for idx, rec in enumerate(records):
            label_norm = normalize_label_text(str(rec.get("cell_type_label", ""))) or ""
            cl_id = str(rec.get("cell_ontology_id", "")).strip()

            if label_norm:
                self._index_by_label.setdefault(label_norm, []).append(idx)
            if cl_id:
                self._index_by_cl_id.setdefault(cl_id, []).append(idx)

        self._loaded = True
        logger.info(
            "MarkerKB loaded: %d entries, %d unique labels from %s",
            len(self._entries), len(self._index_by_label), path.name
        )

    def query_by_label(
        self,
        normalized_label: str,
        species: Optional[str] = None,
        tissue_general: Optional[str] = None,
    ) -> List[dict]:
        """Return all KB entries matching the normalized label."""
        indices = self._index_by_label.get(normalized_label, [])
        return self._filter_entries(indices, species, tissue_general)

    def query_by_cl_id(
        self,
        cl_id: str,
        species: Optional[str] = None,
        tissue_general: Optional[str] = None,
    ) -> List[dict]:
        """Return all KB entries matching the CL ID."""
        indices = self._index_by_cl_id.get(cl_id, [])
        return self._filter_entries(indices, species, tissue_general)

    def _filter_entries(
        self,
        indices: List[int],
        species: Optional[str],
        tissue_general: Optional[str],
    ) -> List[dict]:
        results = []
        for i in indices:
            entry = self._entries[i]
            if species and entry.get("species", "").lower() != species.lower():
                continue
            if tissue_general:
                kb_tissue = entry.get("tissue_general", "").lower()
                if kb_tissue not in ("", "unknown", "all") and kb_tissue != tissue_general.lower():
                    continue
            results.append(entry)
        return results

    def get_marker_genes_for_label(
        self,
        normalized_label: str,
        species: Optional[str] = "Homo sapiens",
        tissue_general: Optional[str] = None,
    ) -> Set[str]:
        """Return the union of all marker genes for a given label."""
        entries = self.query_by_label(normalized_label, species, tissue_general)
        genes: Set[str] = set()
        for entry in entries:
            genes.update(entry.get("marker_genes", []))
        return genes

    def score_gene_list(
        self,
        query_genes: List[str],
        normalized_label: str,
        species: Optional[str] = "Homo sapiens",
        tissue_general: Optional[str] = None,
    ) -> float:
        """
        Compute a simple overlap score between query_genes and KB marker genes.
        Returns Jaccard-like score in [0, 1].
        """
        kb_genes = self.get_marker_genes_for_label(normalized_label, species, tissue_general)
        if not kb_genes or not query_genes:
            return 0.0
        query_set = set(g.upper() for g in query_genes)
        kb_set = set(g.upper() for g in kb_genes)
        intersection = len(query_set & kb_set)
        union = len(query_set | kb_set)
        return round(intersection / union, 4) if union > 0 else 0.0

    def all_labels(self) -> List[str]:
        return list(self._index_by_label.keys())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._entries)


# Module-level singleton
_default_kb: Optional[MarkerKB] = None


def get_default_kb() -> MarkerKB:
    global _default_kb
    if _default_kb is None:
        _default_kb = MarkerKB()
    return _default_kb


def init_default_kb(jsonl_path: str | Path) -> MarkerKB:
    """Initialize and return the module-level default KB."""
    kb = get_default_kb()
    if not kb.is_loaded:
        kb.load(jsonl_path)
    return kb
