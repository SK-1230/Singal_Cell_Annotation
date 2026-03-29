"""
Ontology store — in-memory index of Cell Ontology entries.

Built from:
  resources/ontology/label_aliases.tsv
  resources/ontology/organ_hierarchy.tsv
  data/knowledge/ontology_index.jsonl  (built by 07_build_ontology_resources.py)

Provides fast lookup by CL ID or normalized label.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OntologyStore:
    """
    Lightweight in-memory Cell Ontology store.

    Usage:
        store = OntologyStore()
        store.load(ontology_index_path)
        entry = store.get_by_id("CL:0000084")
        cl_id = store.label_to_id("t cell")
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, dict] = {}
        self._by_label: Dict[str, str] = {}  # normalized_label → cl_id
        self._loaded = False

    def load(self, jsonl_path: str | Path) -> None:
        from sca.common.jsonl_utils import read_jsonl
        from sca.data.label_normalization import normalize_label_text

        path = Path(jsonl_path)
        records = read_jsonl(path)
        if not records:
            logger.warning("ontology_index.jsonl empty or missing: %s", path)
            return

        for rec in records:
            cl_id = str(rec.get("cell_ontology_id", "")).strip()
            label = normalize_label_text(str(rec.get("label", ""))) or ""
            if cl_id:
                self._by_id[cl_id] = rec
            if label and cl_id:
                self._by_label[label] = cl_id
            for syn in rec.get("synonyms", []):
                syn_norm = normalize_label_text(syn) or ""
                if syn_norm and cl_id:
                    self._by_label[syn_norm] = cl_id

        self._loaded = True
        logger.info(
            "OntologyStore loaded: %d entries, %d label mappings from %s",
            len(self._by_id), len(self._by_label), path.name
        )

    def get_by_id(self, cl_id: str) -> Optional[dict]:
        return self._by_id.get(cl_id)

    def label_to_id(self, normalized_label: str) -> Optional[str]:
        return self._by_label.get(normalized_label)

    def get_parent_label(self, cl_id: str) -> Optional[str]:
        entry = self._by_id.get(cl_id)
        if entry:
            return entry.get("parent_label")
        return None

    def get_label_level(self, cl_id: str) -> Optional[str]:
        entry = self._by_id.get(cl_id)
        if entry:
            return entry.get("label_level")
        return None

    def get_organ_scope(self, cl_id: str) -> Optional[str]:
        entry = self._by_id.get(cl_id)
        if entry:
            return entry.get("organ_scope")
        return None

    def all_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def all_labels(self) -> List[str]:
        return list(self._by_label.keys())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._by_id)


# Module-level singleton (lazy-initialized)
_default_store: Optional[OntologyStore] = None


def get_default_store() -> OntologyStore:
    global _default_store
    if _default_store is None:
        _default_store = OntologyStore()
    return _default_store


def init_default_store(jsonl_path: str | Path) -> OntologyStore:
    """Initialize and return the module-level default store."""
    store = get_default_store()
    if not store.is_loaded:
        store.load(jsonl_path)
    return store
