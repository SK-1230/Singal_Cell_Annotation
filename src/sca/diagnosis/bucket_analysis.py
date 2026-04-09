"""
bucket_analysis.py — 将测试样本按不同维度分桶，计算各桶的指标

Bucket 维度：
  1. seen_status:     训练集中见过几次（zero_shot / singleton / rare / frequent）
  2. tissue:          组织类型（brain / lung / liver / kidney / ...）
  3. hardness:        是否在 hard 子集中
  4. marker_quality:  marker_quality_score 高低
  5. ontology_mapped: gold ontology_id 是否有效映射
  6. parent_known:    gold cell_type 的父节点是否已知
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .metrics import compute_metrics, MetricsDict


def _norm(s: Any) -> str:
    if s is None:
        return ""
    import re
    t = str(s).strip().lower()
    t = re.sub(r"[-_/,]", " ", t)
    t = re.sub(r"\bhuman\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


class BucketAnalyzer:
    """
    将推理结果与测试集元数据、训练集统计结合，按多维度分桶分析。

    使用方式::

        analyzer = BucketAnalyzer()
        analyzer.load_train(train_full_path)
        analyzer.load_test_meta(test_full_path)
        analyzer.load_infer_results(results_jsonl_path)
        analyzer.load_ontology(ontology_index_path)
        report = analyzer.run()
    """

    def __init__(self) -> None:
        self._train_counts: Dict[str, int] = {}          # cell_type_clean → 训练集出现次数
        self._test_meta: Dict[Tuple, Dict] = {}           # (dataset_id, cell_type_clean) → meta
        self._results: List[Dict[str, Any]] = []          # 推理结果列表
        self._valid_ont_ids: set = set()                  # 官方 CL 中的合法 ontology id
        self._parent_map: Dict[str, str] = {}             # cell_type → parent_cell_type

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_train(self, path: str | Path) -> None:
        """统计训练集中每种 cell_type 的出现次数。"""
        counts: Dict[str, int] = defaultdict(int)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                ct = _norm(d.get("cell_type_clean", ""))
                if ct:
                    counts[ct] += 1
        self._train_counts = dict(counts)

    def load_test_meta(self, path: str | Path) -> None:
        """加载测试集完整元数据，用于补充分桶所需字段。"""
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                key = (d.get("dataset_id", ""), _norm(d.get("cell_type_clean", "")))
                self._test_meta[key] = d

    def load_infer_results(self, path: str | Path) -> None:
        """加载推理结果 JSONL。"""
        results = []
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))
        self._results = results

    def load_ontology(self, path: str | Path) -> None:
        """加载 Cell Ontology 索引，提取合法 ID 集合和父节点映射。"""
        ids: set = set()
        pmap: Dict[str, str] = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                oid = d.get("cell_ontology_id", "")
                if oid:
                    ids.add(oid)
                label = _norm(d.get("label", ""))
                parent = _norm(d.get("parent_label", "") or "")
                if label and parent and label != parent:
                    pmap[label] = parent
        self._valid_ont_ids = ids
        self._parent_map = pmap

    def load_parent_supplement(self, path: str | Path) -> None:
        """加载人工补充的父节点映射（优先级高于 ontology_index）。"""
        with open(path) as f:
            supp = json.load(f)
        for k, v in supp.items():
            self._parent_map[_norm(k)] = _norm(v)

    # ------------------------------------------------------------------
    # 分桶逻辑
    # ------------------------------------------------------------------

    def _enrich_result(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """为推理结果附加分桶所需的元数据字段。"""
        r = dict(r)
        gold_ct = _norm(r.get("gold_cell_type", ""))
        dataset_id = r.get("dataset_id", "")
        key = (dataset_id, gold_ct)

        meta = self._test_meta.get(key, {})
        r["_meta"] = meta

        # seen_status
        cnt = self._train_counts.get(gold_ct, 0)
        if cnt == 0:
            r["_seen_status"] = "zero_shot"
        elif cnt == 1:
            r["_seen_status"] = "singleton"
        elif cnt <= 4:
            r["_seen_status"] = "rare"
        else:
            r["_seen_status"] = "frequent"

        # tissue
        r["_tissue"] = r.get("tissue_general") or meta.get("tissue_general", "unknown")

        # hardness flags from meta
        hf = meta.get("hardness_flags") or {}
        r["_hard"] = any(hf.values()) if isinstance(hf, dict) else False

        # marker quality
        mq = meta.get("marker_quality_score")
        if mq is None:
            r["_marker_quality"] = "unknown"
        elif mq >= 0.7:
            r["_marker_quality"] = "high"
        else:
            r["_marker_quality"] = "low"

        # ontology mapped
        gold_oid = r.get("gold_ontology_id", "")
        r["_ont_mapped"] = bool(gold_oid and gold_oid in self._valid_ont_ids)

        # parent known
        r["_parent_known"] = gold_ct in self._parent_map

        return r

    # ------------------------------------------------------------------
    # 运行分析
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        执行所有分桶分析，返回结构化报告字典。

        Returns:
            {
              "overall": MetricsDict,
              "by_seen_status": {bucket: MetricsDict, ...},
              "by_tissue": {...},
              "by_hardness": {...},
              "by_marker_quality": {...},
              "by_ont_mapped": {...},
              "by_parent_known": {...},
              "top3_bottlenecks": [...],
              "enriched_results": [...],
            }
        """
        enriched = [self._enrich_result(r) for r in self._results]

        buckets: Dict[str, Dict[str, List]] = {
            "by_seen_status":    defaultdict(list),
            "by_tissue":         defaultdict(list),
            "by_hardness":       defaultdict(list),
            "by_marker_quality": defaultdict(list),
            "by_ont_mapped":     defaultdict(list),
            "by_parent_known":   defaultdict(list),
        }

        for r in enriched:
            buckets["by_seen_status"][r["_seen_status"]].append(r)
            buckets["by_tissue"][r["_tissue"]].append(r)
            buckets["by_hardness"]["hard" if r["_hard"] else "normal"].append(r)
            buckets["by_marker_quality"][r["_marker_quality"]].append(r)
            buckets["by_ont_mapped"]["mapped" if r["_ont_mapped"] else "unmapped"].append(r)
            buckets["by_parent_known"]["parent_known" if r["_parent_known"] else "parent_unknown"].append(r)

        report: Dict[str, Any] = {
            "overall": compute_metrics(enriched),
        }
        for dim, groups in buckets.items():
            report[dim] = {k: compute_metrics(v) for k, v in sorted(groups.items())}

        report["top3_bottlenecks"] = self._find_bottlenecks(report)
        report["enriched_results"] = enriched
        return report

    def _find_bottlenecks(self, report: Dict[str, Any]) -> List[str]:
        """自动识别 severe_error 最高的 top-3 分桶。"""
        candidates: List[Tuple[float, str]] = []
        overall_severe = report["overall"].get("severe_error", 0)

        for dim_name in [
            "by_seen_status", "by_tissue", "by_hardness",
            "by_marker_quality", "by_ont_mapped",
        ]:
            for bucket, m in report.get(dim_name, {}).items():
                sv = m.get("severe_error", 0)
                n = m.get("n", 0)
                if n >= 3:  # 至少 3 个样本才有统计意义
                    delta = sv - overall_severe
                    candidates.append((delta, f"{dim_name}/{bucket} (n={n}, severe={sv*100:.1f}%)"))

        candidates.sort(reverse=True)
        return [desc for _, desc in candidates[:3]]
