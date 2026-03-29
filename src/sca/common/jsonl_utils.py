"""
JSONL read/write utilities for the SCA project.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

logger = logging.getLogger(__name__)


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records to a JSONL file. Returns number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    logger.debug("Wrote %d records to %s", count, path)
    return count


def append_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    """Append records to an existing JSONL file. Returns number of records appended."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, skip_invalid: bool = True) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_invalid:
                    logger.warning("Skip invalid JSON at line %d in %s: %s", line_no, path.name, e)
                else:
                    raise
    return records


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Lazily iterate records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass
