"""
metrics.py — 诊断模块的核心指标计算函数

所有指标均基于推理结果 JSONL 中已有的字段（cell_type_exact、
cell_type_severe_error 等），不重新实现评估逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, List

MetricsDict = Dict[str, Any]

# 结果字段名常量（与 infer_qwen3_kb_retrieval.py 保持一致）
F_EXACT = "cell_type_exact"
F_COMPAT = "ontology_compatible"
F_LINEAGE = "cell_type_same_lineage"
F_SEVERE = "cell_type_severe_error"
F_PARSE = "parse_ok"
F_PRED_GENERAL = "pred_more_general"    # 可能不在结果里，需要推导
F_PRED_SPECIFIC = "pred_more_specific"  # 同上
F_ONT_ID = "ont_id_correct"             # 自定义字段，由外部脚本注入


def compute_metrics(results: List[Dict[str, Any]]) -> MetricsDict:
    """
    对一批推理结果计算所有诊断指标。

    Args:
        results: 推理结果字典列表，字段来自 infer_qwen3_kb_retrieval.py 输出

    Returns:
        指标字典，所有比率保留 4 位小数
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    exact = sum(1 for r in results if r.get(F_EXACT, False))
    compat = sum(1 for r in results if r.get(F_COMPAT, False))
    lineage = sum(1 for r in results if r.get(F_LINEAGE, False))
    severe = sum(1 for r in results if r.get(F_SEVERE, False))
    parse_ok = sum(1 for r in results if r.get(F_PARSE, True))

    # ontology_id 正确率（仅对有 gold_ontology_id 的样本）
    ont_results = [
        r for r in results
        if r.get("gold_ontology_id") and r.get("gold_ontology_id") != ""
    ]
    if ont_results:
        ont_correct = sum(
            1 for r in ont_results
            if _normalize(r.get("pred_ontology_id", "")) == _normalize(r.get("gold_ontology_id", ""))
        )
        ont_id_acc = round(ont_correct / len(ont_results), 4)
    else:
        ont_id_acc = None

    return {
        "n": n,
        "exact": round(exact / n, 4),
        "ontology_compat": round(compat / n, 4),
        "same_lineage": round(lineage / n, 4),
        "severe_error": round(severe / n, 4),
        "parse_ok": round(parse_ok / n, 4),
        "ont_id_acc": ont_id_acc,
        "n_ont_evaluated": len(ont_results),
    }


def compute_metrics_delta(baseline: MetricsDict, variant: MetricsDict) -> MetricsDict:
    """
    计算两组指标的差值（variant - baseline）。
    正值表示 variant 更好，负值表示 variant 更差。
    """
    keys = ["exact", "ontology_compat", "same_lineage", "severe_error", "parse_ok"]
    delta: MetricsDict = {"n_baseline": baseline.get("n"), "n_variant": variant.get("n")}
    for k in keys:
        b = baseline.get(k)
        v = variant.get(k)
        if b is not None and v is not None:
            # severe_error：delta 为负数时表示 variant 的错误率更低（更好）
            delta[f"delta_{k}"] = round(v - b, 4)
    return delta


def format_metrics_row(label: str, m: MetricsDict) -> str:
    """将指标格式化为 Markdown 表格一行。"""
    exact = _pct(m.get("exact"))
    compat = _pct(m.get("ontology_compat"))
    lineage = _pct(m.get("same_lineage"))
    severe = _pct(m.get("severe_error"))
    parse = _pct(m.get("parse_ok"))
    ont_id = _pct(m.get("ont_id_acc"))
    n = m.get("n", "?")
    return f"| {label} | {n} | {exact} | {compat} | {lineage} | {severe} | {parse} | {ont_id} |"


def metrics_table_header() -> str:
    header = "| 模式 | n | Exact | Ontology Compat | Same Lineage | Severe Error | Parse OK | Ont ID Acc |"
    sep = "|------|---|-------|-----------------|--------------|--------------|----------|------------|"
    return header + "\n" + sep


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def _normalize(s: str) -> str:
    return str(s).strip().lower()
