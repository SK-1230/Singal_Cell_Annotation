"""
report_utils.py — Markdown 报告生成工具函数

所有 section 以字符串返回，由 generate_diagnosis_report.py 拼接。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .metrics import compute_metrics, format_metrics_row, metrics_table_header, MetricsDict


def section(title: str, level: int = 2) -> str:
    prefix = "#" * level
    return f"\n{prefix} {title}\n"


def kv_table(data: Dict[str, Any], title: Optional[str] = None) -> str:
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| 字段 | 值 |")
    lines.append("|------|-----|")
    for k, v in data.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def bucket_table(
    bucket_metrics: Dict[str, MetricsDict],
    title: str,
    sort_by: str = "severe_error",
    sort_desc: bool = True,
) -> str:
    """将分桶指标渲染为 Markdown 表格。"""
    lines = [f"\n**{title}**\n", metrics_table_header()]
    rows = list(bucket_metrics.items())
    rows.sort(key=lambda x: x[1].get(sort_by, 0) or 0, reverse=sort_desc)
    for bucket, m in rows:
        lines.append(format_metrics_row(bucket, m))
    return "\n".join(lines)


def ablation_comparison_table(
    modes: Dict[str, MetricsDict],
    title: str = "消融对比",
) -> str:
    """多模式对比表格。"""
    lines = [f"\n**{title}**\n", metrics_table_header()]
    for mode, m in modes.items():
        lines.append(format_metrics_row(mode, m))
    return "\n".join(lines)


def verdict_block(question: str, verdict: str, evidence: str) -> str:
    """生成标准化诊断结论块。"""
    return (
        f"\n> **问题**: {question}\n"
        f">\n"
        f"> **结论**: {verdict}\n"
        f">\n"
        f"> **依据**: {evidence}\n"
    )


def top_bottlenecks_block(bottlenecks: List[str]) -> str:
    if not bottlenecks:
        return "\n_未发现明显瓶颈_\n"
    lines = ["\n**Top-3 最致命错误来源**：\n"]
    for i, b in enumerate(bottlenecks, 1):
        lines.append(f"{i}. `{b}`")
    return "\n".join(lines)


def noisy_samples_block(examples: List[Dict[str, Any]], limit: int = 20) -> str:
    if not examples:
        return "\n_无典型问题样本_\n"
    lines = [
        "\n| # | cell_type | ont_id | quality | 问题 |",
        "|---|-----------|--------|---------|------|",
    ]
    for i, ex in enumerate(examples[:limit], 1):
        ct = ex.get("cell_type", "")
        oid = ex.get("ont_id", "")
        q = ex.get("quality", "")
        issues = "; ".join(ex.get("issues", []))
        lines.append(f"| {i} | {ct} | {oid} | {q} | {issues} |")
    return "\n".join(lines)


def summary_answers(answers: Dict[str, str]) -> str:
    """渲染最终诊断问答摘要。"""
    lines = []
    for q, a in answers.items():
        lines.append(f"\n- **{q}**  \n  {a}")
    return "\n".join(lines)
