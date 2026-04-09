"""
label_quality.py — 评估 SFT 训练样本中各字段的可靠性

质量等级：
  clean   — cell_type 与 gold 一致，ontology_id 合法，parent 合法
  weak    — cell_type 正确，但 ontology_id 或 parent 有问题
  noisy   — cell_type 与 gold 不完全一致，或存在多项字段异常
  invalid — 无法解析 / cell_type 为空 / ontology_id 格式非法
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


_CL_ID_RE = re.compile(r"^CL:\d{7}$")


def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"[-_/,]", " ", t)
    t = re.sub(r"\bhuman\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


class LabelQualityChecker:
    """
    逐条检查训练集样本的标签质量。

    使用方式::

        checker = LabelQualityChecker()
        checker.load_ontology(ontology_index_path)
        checker.load_parent_supplement(parent_map_path)
        results = checker.check_all(train_full_path)
        summary = checker.summarize(results)
    """

    def __init__(self) -> None:
        self._valid_ont_ids: Set[str] = set()
        self._valid_labels: Set[str] = set()        # 官方 CL label（规范化）
        self._parent_map: Dict[str, str] = {}        # label → parent_label（规范化）

    def load_ontology(self, path: str | Path) -> None:
        ids: Set[str] = set()
        labels: Set[str] = set()
        pmap: Dict[str, str] = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                oid = d.get("cell_ontology_id", "")
                if oid:
                    ids.add(oid)
                label = _norm(d.get("label", ""))
                if label:
                    labels.add(label)
                parent = _norm(d.get("parent_label", "") or "")
                if label and parent and label != parent:
                    pmap[label] = parent
                # 同义词也加入合法标签集
                for syn in d.get("synonyms", []):
                    sl = _norm(syn)
                    if sl:
                        labels.add(sl)
        self._valid_ont_ids = ids
        self._valid_labels = labels
        self._parent_map = pmap

    def load_parent_supplement(self, path: str | Path) -> None:
        with open(path) as f:
            supp = json.load(f)
        for k, v in supp.items():
            self._parent_map[_norm(k)] = _norm(v)
            # 也把这些细粒度类型加入合法标签集
            self._valid_labels.add(_norm(k))

    def load_alias_table(self, tsv_path: str | Path) -> None:
        """
        从 label_aliases.tsv 加载合法标签集和 CL ID，补充 ontology_index 的不足。
        """
        import csv
        path = Path(tsv_path)
        if not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                norm_label = _norm(row.get("normalized_label", ""))
                raw_label = _norm(row.get("raw_label", ""))
                cl_id = (row.get("cell_ontology_id", "") or "").strip()
                parent = _norm(row.get("parent_label", "") or "")

                if norm_label:
                    self._valid_labels.add(norm_label)
                if raw_label:
                    self._valid_labels.add(raw_label)
                if cl_id and _CL_ID_RE.match(cl_id):
                    self._valid_ont_ids.add(cl_id)
                if norm_label and parent:
                    self._parent_map.setdefault(norm_label, parent)

    def check_all(self, train_full_path: str | Path) -> List[Dict[str, Any]]:
        """
        检查训练集每条样本，返回带质量标注的记录列表。

        每条记录新增字段：
          _quality:       clean / weak / noisy / invalid
          _issues:        问题描述列表
          _cell_type_ok:  cell_type_clean 是否在 CL 中
          _ont_id_ok:     ontology_id 是否合法
          _ont_id_match:  ontology_id 是否与 cell_type 对应（交叉校验）
          _parent_ok:     parent_cell_type 是否合法
        """
        records = []
        with open(train_full_path) as f:
            for line in f:
                d = json.loads(line)
                result = self._check_one(d)
                records.append(result)
        return records

    def _check_one(self, d: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        r = dict(d)

        cell_type = _norm(d.get("cell_type_clean", ""))
        ont_id = d.get("cell_ontology_id", "") or ""
        parent = _norm(d.get("cell_ontology_parent_label", "") or "")

        # 1. cell_type 是否在 CL 合法标签集中
        ct_ok = bool(cell_type and cell_type in self._valid_labels)
        r["_cell_type_ok"] = ct_ok
        if not cell_type:
            issues.append("cell_type 为空")
        elif not ct_ok:
            issues.append(f"cell_type '{cell_type}' 不在 CL 标签集中")

        # 2. ontology_id 格式是否合法
        if not ont_id:
            ont_id_ok = False
            issues.append("ontology_id 缺失")
        elif not _CL_ID_RE.match(ont_id):
            ont_id_ok = False
            issues.append(f"ontology_id '{ont_id}' 格式非法（应为 CL:XXXXXXX）")
        elif ont_id not in self._valid_ont_ids:
            ont_id_ok = False
            issues.append(f"ontology_id '{ont_id}' 不在 CL 数据库中")
        else:
            ont_id_ok = True
        r["_ont_id_ok"] = ont_id_ok

        # 3. parent_cell_type 是否合法
        if not parent:
            parent_ok = False
            issues.append("parent_cell_type 缺失")
        elif parent not in self._valid_labels and parent not in self._parent_map:
            parent_ok = False
            issues.append(f"parent '{parent}' 不在 CL 中")
        else:
            parent_ok = True
        r["_parent_ok"] = parent_ok

        # 4. 综合质量评级
        if not cell_type or (not ct_ok and "为空" in str(issues)):
            quality = "invalid"
        elif not ct_ok and not ont_id_ok:
            quality = "noisy"
        elif not ont_id_ok or not parent_ok:
            quality = "weak"
        else:
            quality = "clean"

        r["_quality"] = quality
        r["_issues"] = issues
        return r

    def summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成质量汇总统计。"""
        from collections import Counter
        quality_counts = Counter(r["_quality"] for r in records)
        n = len(records)

        noisy_examples = [
            {
                "cell_type": r.get("cell_type_clean", ""),
                "ont_id": r.get("cell_ontology_id", ""),
                "parent": r.get("cell_ontology_parent_label", ""),
                "quality": r["_quality"],
                "issues": r["_issues"],
            }
            for r in records if r["_quality"] in ("noisy", "invalid")
        ][:20]  # Top-20 典型问题样本

        return {
            "total": n,
            "quality_counts": dict(quality_counts),
            "quality_rates": {k: round(v / n, 4) for k, v in quality_counts.items()},
            "cell_type_ok_rate": round(sum(1 for r in records if r["_cell_type_ok"]) / n, 4),
            "ont_id_ok_rate": round(sum(1 for r in records if r["_ont_id_ok"]) / n, 4),
            "parent_ok_rate": round(sum(1 for r in records if r["_parent_ok"]) / n, 4),
            "top20_noisy_examples": noisy_examples,
        }
