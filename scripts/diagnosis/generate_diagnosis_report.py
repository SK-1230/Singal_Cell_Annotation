"""
generate_diagnosis_report.py — 整合所有诊断子模块结果，生成统一诊断报告

从各子模块的输出目录读取 Markdown 报告片段，
拼接成一份完整的 diagnosis_report.md，并回答 5 个核心问题。

用法
----
  # 自动在所有诊断子目录中搜索结果
  python scripts/diagnosis/generate_diagnosis_report.py

  # 指定各子目录
  python scripts/diagnosis/generate_diagnosis_report.py \\
      --buckets  output/diagnosis/buckets \\
      --schema   output/diagnosis/schema_ablation \\
      --kb       output/diagnosis/kb_ablation \\
      --ontology output/diagnosis/ontology_ablation \\
      --noise    output/diagnosis/label_noise \\
      --coverage output/diagnosis/data_coverage \\
      --output   output/diagnosis/final_report

输出
----
  output/diagnosis/final_report/
    diagnosis_report.md   — 统一诊断报告
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
import diag_config as cfg


REPORT_TEMPLATE = """\
# SCA 诊断报告

> 生成时间：{timestamp}
> 当前最优模型：`{adapter}`
> 测试集：`{test_file}`（n={n_test}）

---

## 核心问题回答

{core_answers}

---

## 1. 错误分桶分析

{bucket_section}

---

## 2. Schema 消融（训练目标是否过宽）

{schema_section}

---

## 3. KB 检索消融（是增益还是噪声）

{kb_section}

---

## 4. Ontology Target 消融

{ontology_section}

---

## 5. 标签噪声分析

{noise_section}

---

## 6. 数据覆盖分析

{coverage_section}

---

## 7. 综合建议（优先级排序）

{recommendations}

---

_本报告由 `generate_diagnosis_report.py` 自动生成。_
"""


def _read_md(path: Path, max_lines: int = 100) -> str:
    """读取 Markdown 文件（限制行数，避免报告过长）。"""
    if not path.exists():
        return f"_未找到文件：`{path}`_"
    lines = path.read_text(encoding="utf-8").splitlines()
    # 跳过一级标题（避免重复）
    lines = [l for l in lines if not l.startswith("# ")]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"\n_（已截断，完整内容见 `{path}`）_"]
    return "\n".join(lines)


def _count_test() -> int:
    try:
        with open(cfg.TEST_FULL) as f:
            return sum(1 for _ in f)
    except Exception:
        return -1


def _extract_core_answers(
    bucket_dir: Optional[Path],
    schema_dir: Optional[Path],
    kb_dir: Optional[Path],
    ontology_dir: Optional[Path],
    noise_dir: Optional[Path],
) -> str:
    """
    从各子报告中提取 verdict_block，回答 5 个核心问题。
    如果子报告不存在则标记为"待运行"。
    """

    def _extract_verdict(md_path: Optional[Path], keyword: str) -> str:
        if not md_path or not md_path.exists():
            return "_待运行_"
        text = md_path.read_text(encoding="utf-8")
        # 找到包含 keyword 的 verdict_block
        m = re.search(
            r"> \*\*问题\*\*.*?" + re.escape(keyword) + r".*?\n(.*?)(?=\n---|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(0)[:400].strip()
        # fallback: 找第一个结论
        m2 = re.search(r"> \*\*结论\*\*: (.*?)(?=\n|$)", text)
        if m2:
            return m2.group(1).strip()
        return "_见详细报告_"

    q1_verdict = _extract_verdict(
        schema_dir / "schema_comparison.md" if schema_dir else None,
        "schema"
    )
    q2_verdict = _extract_verdict(
        kb_dir / "kb_comparison.md" if kb_dir else None,
        "KB"
    )
    q3_verdict = _extract_verdict(
        ontology_dir / "ontology_comparison.md" if ontology_dir else None,
        "ontology"
    )
    q4_verdict = _extract_verdict(
        bucket_dir / "bucket_report.md" if bucket_dir else None,
        "zero-shot"
    )
    q5_verdict = _extract_verdict(
        noise_dir / "label_noise_report.md" if noise_dir else None,
        "噪声"
    )

    lines = [
        "| # | 核心问题 | 结论 |",
        "|---|---------|------|",
        f"| 1 | 训练目标过宽（full_json）是否是主瓶颈？ | {_one_line(q1_verdict)} |",
        f"| 2 | KB 检索是否形成稳定正增益？ | {_one_line(q2_verdict)} |",
        f"| 3 | ontology_id 是否应改为后处理？ | {_one_line(q3_verdict)} |",
        f"| 4 | zero-shot/rare/tissue-shift 哪个影响最大？ | {_one_line(q4_verdict)} |",
        f"| 5 | 自蒸馏标签噪声是否严重到必须先处理？ | {_one_line(q5_verdict)} |",
    ]
    return "\n".join(lines)


def _one_line(s: str) -> str:
    """将多行文本压缩为一行。"""
    s = re.sub(r"\s+", " ", s.replace("\n", " ")).strip()
    return s[:120] + ("..." if len(s) > 120 else "")


def _auto_recommendations(
    schema_dir: Optional[Path],
    kb_dir: Optional[Path],
    noise_dir: Optional[Path],
    coverage_dir: Optional[Path],
) -> str:
    """根据各子报告自动生成优先级建议。"""
    recs = []

    # schema 建议
    if schema_dir and (schema_dir / "schema_comparison.md").exists():
        text = (schema_dir / "schema_comparison.md").read_text(encoding="utf-8")
        if "是主瓶颈" in text:
            recs.append("1. **立即收缩主任务**：切换到 `cell_type_only` 训练目标（Phase 1）")
        elif "不是主瓶颈" in text:
            recs.append("1. ~~收缩主任务~~ schema 不是主瓶颈，跳过此项")
        else:
            recs.append("1. **可选**：尝试 cell_type_only 训练以验证 schema 影响（需重训）")

    # KB 建议
    if kb_dir and (kb_dir / "kb_comparison.md").exists():
        text = (kb_dir / "kb_comparison.md").read_text(encoding="utf-8")
        if "噪声源" in text:
            recs.append("2. **禁用 KB**：当前 KB 是噪声，建议 `--no-kb` 训练或大幅改进检索")
        elif "稳定正增益" in text:
            recs.append("2. **优化 KB**：KB 有效，可升级检索方式（BM25 → dense retrieval）")
        else:
            recs.append("2. **暂不投入 KB**：KB 增益不稳定，优先处理数据/标签问题")

    # 标签建议
    if noise_dir and (noise_dir / "label_noise_report.md").exists():
        text = (noise_dir / "label_noise_report.md").read_text(encoding="utf-8")
        if "必须优先处理" in text:
            recs.append("3. **净化标签**（Phase 2）：noise+invalid > 20%，必须先清洗监督数据")
        elif "建议在扩数据前先净化" in text:
            recs.append("3. **净化标签**（Phase 2，扩数据前执行）：ontology_id 合法率低")
        else:
            recs.append("3. ~~净化标签~~ 标签质量尚可，可以直接扩数据")

    # 数据覆盖建议
    if coverage_dir and (coverage_dir / "data_coverage_report.md").exists():
        recs.append(
            "4. **扩充数据**（Phase 3）："
            "优先补 zero-shot 细胞类型和训练样本最少的 tissue，"
            "目标：从 ~500 条扩充至 2000+ 条"
        )

    recs.append("5. **保持现有框架**：训练 → KB 推理 → 评估闭环不变，只做针对性修改")

    return "\n".join(recs)


def run(
    bucket_dir: Optional[Path],
    schema_dir: Optional[Path],
    kb_dir: Optional[Path],
    ontology_dir: Optional[Path],
    noise_dir: Optional[Path],
    coverage_dir: Optional[Path],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    core_answers = _extract_core_answers(
        bucket_dir, schema_dir, kb_dir, ontology_dir, noise_dir
    )

    report = REPORT_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        adapter=str(cfg.BEST_ADAPTER),
        test_file=str(cfg.TEST_FULL),
        n_test=_count_test(),
        core_answers=core_answers,
        bucket_section=_read_md(bucket_dir / "bucket_report.md") if bucket_dir else "_待运行 analyze_error_buckets.py_",
        schema_section=_read_md(schema_dir / "schema_comparison.md") if schema_dir else "_待运行 ablate_output_schema.py_",
        kb_section=_read_md(kb_dir / "kb_comparison.md") if kb_dir else "_待运行 ablate_kb_retrieval.py_",
        ontology_section=_read_md(ontology_dir / "ontology_comparison.md") if ontology_dir else "_待运行 ablate_ontology_target.py_",
        noise_section=_read_md(noise_dir / "label_noise_report.md") if noise_dir else "_待运行 analyze_label_noise.py_",
        coverage_section=_read_md(coverage_dir / "data_coverage_report.md") if coverage_dir else "_待运行 analyze_data_coverage.py_",
        recommendations=_auto_recommendations(schema_dir, kb_dir, noise_dir, coverage_dir),
    )

    out_path = output_dir / "diagnosis_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[report] 统一诊断报告已写入：{out_path}")


def _parse_args():
    p = argparse.ArgumentParser(description="生成统一诊断报告")
    p.add_argument("--buckets",  default=str(_PROJECT / "output/diagnosis/buckets"))
    p.add_argument("--schema",   default=str(_PROJECT / "output/diagnosis/schema_ablation"))
    p.add_argument("--kb",       default=str(_PROJECT / "output/diagnosis/kb_ablation"))
    p.add_argument("--ontology", default=str(_PROJECT / "output/diagnosis/ontology_ablation"))
    p.add_argument("--noise",    default=str(_PROJECT / "output/diagnosis/label_noise"))
    p.add_argument("--coverage", default=str(_PROJECT / "output/diagnosis/data_coverage"))
    p.add_argument("--output",   default=str(_PROJECT / "output/diagnosis/final_report"))
    return p.parse_args()


def _to_path_or_none(s: str) -> Optional[Path]:
    p = Path(s)
    return p if p.exists() else None


if __name__ == "__main__":
    args = _parse_args()
    run(
        bucket_dir=_to_path_or_none(args.buckets),
        schema_dir=_to_path_or_none(args.schema),
        kb_dir=_to_path_or_none(args.kb),
        ontology_dir=_to_path_or_none(args.ontology),
        noise_dir=_to_path_or_none(args.noise),
        coverage_dir=_to_path_or_none(args.coverage),
        output_dir=Path(args.output),
    )
