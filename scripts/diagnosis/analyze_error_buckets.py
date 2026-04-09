"""
analyze_error_buckets.py — 将当前最优模型的错误按多维度分桶，
                            找出性能下降的主要来源

用法
----
  # 分析当前最优推理结果（默认）
  python scripts/diagnosis/analyze_error_buckets.py

  # 分析指定推理结果
  python scripts/diagnosis/analyze_error_buckets.py \\
      --results output/infer_kb_retrieval_20260405_211943/results.jsonl \\
      --output-dir output/diagnosis/buckets

输出
----
  <output_dir>/
    bucket_report.md      — Markdown 详细报告（含每个分桶的指标）
    bucket_summary.csv    — CSV 汇总表（可用 Excel/pandas 查看）
    enriched_results.jsonl — 附加了分桶标签的完整推理结果
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# 将项目 src/ 目录加入路径
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT / "src"))

from sca.diagnosis.bucket_analysis import BucketAnalyzer
from sca.diagnosis.metrics import metrics_table_header, format_metrics_row
from sca.diagnosis.report_utils import (
    section, bucket_table, top_bottlenecks_block, kv_table
)

# 引入诊断配置
sys.path.insert(0, str(_HERE))
import diag_config as cfg


# ─────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────

def run(results_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bucket] 加载推理结果：{results_path}")
    analyzer = BucketAnalyzer()
    analyzer.load_train(cfg.TRAIN_FULL)
    analyzer.load_test_meta(cfg.TEST_FULL)
    analyzer.load_infer_results(results_path)
    analyzer.load_ontology(cfg.ONTOLOGY_INDEX)
    if cfg.PARENT_SUPPLEMENT.exists():
        analyzer.load_parent_supplement(cfg.PARENT_SUPPLEMENT)

    print("[bucket] 执行分桶分析...")
    report = analyzer.run()

    # ── 写 Markdown 报告 ──────────────────────────────
    md_lines = [
        "# 错误分桶分析报告\n",
        f"> 推理结果：`{results_path}`\n",
        section("1. 整体指标"),
        kv_table({k: v for k, v in report["overall"].items() if k != "n"}),
        f"\n样本总数：**{report['overall']['n']}**\n",
        section("2. Top-3 最致命错误来源"),
        top_bottlenecks_block(report["top3_bottlenecks"]),
        section("3. 按 Seen Status 分桶（zero-shot / singleton / rare / frequent）"),
        bucket_table(report["by_seen_status"], "Seen Status"),
        section("4. 按组织类型分桶"),
        bucket_table(report["by_tissue"], "Tissue"),
        section("5. Hard vs Normal"),
        bucket_table(report["by_hardness"], "Hardness"),
        section("6. Marker Quality 高低"),
        bucket_table(report["by_marker_quality"], "Marker Quality"),
        section("7. Ontology 是否有映射"),
        bucket_table(report["by_ont_mapped"], "Ontology Mapped"),
        section("8. 父节点是否已知"),
        bucket_table(report["by_parent_known"], "Parent Known"),
        section("9. 诊断结论"),
        _auto_conclusion(report),
    ]
    md_path = output_dir / "bucket_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[bucket] Markdown 报告已写入：{md_path}")

    # ── 写 CSV ───────────────────────────────────────
    csv_rows = []
    metric_keys = ["n", "exact", "ontology_compat", "same_lineage", "severe_error", "parse_ok"]
    for dim in [
        "by_seen_status", "by_tissue", "by_hardness",
        "by_marker_quality", "by_ont_mapped", "by_parent_known",
    ]:
        for bucket, m in report.get(dim, {}).items():
            row = {"dimension": dim, "bucket": bucket}
            row.update({k: m.get(k, "") for k in metric_keys})
            csv_rows.append(row)

    csv_path = output_dir / "bucket_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"[bucket] CSV 汇总已写入：{csv_path}")

    # ── 写增强结果 ────────────────────────────────────
    enrich_path = output_dir / "enriched_results.jsonl"
    with open(enrich_path, "w", encoding="utf-8") as f:
        for r in report["enriched_results"]:
            # 只保留诊断所需的轻量字段（去掉原始 _meta 大字典）
            out = {
                k: v for k, v in r.items()
                if not k.startswith("_meta")
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[bucket] 增强结果已写入：{enrich_path}")

    # ── 打印关键结论到终端 ────────────────────────────
    overall = report["overall"]
    print("\n" + "=" * 60)
    print(f"  整体 Exact:        {overall.get('exact', 0)*100:.1f}%")
    print(f"  整体 Severe Error: {overall.get('severe_error', 0)*100:.1f}%")
    print(f"  Top-3 瓶颈：")
    for b in report.get("top3_bottlenecks", []):
        print(f"    - {b}")
    print("=" * 60)


def _auto_conclusion(report: dict) -> str:
    """根据分桶结果自动生成文字结论。"""
    lines = []
    overall_severe = report["overall"].get("severe_error", 0)
    overall_exact = report["overall"].get("exact", 0)

    # zero-shot 影响
    seen = report.get("by_seen_status", {})
    zs = seen.get("zero_shot", {})
    if zs.get("n", 0) > 0:
        lines.append(
            f"- **Zero-shot 影响**：zero_shot 样本 {zs['n']} 个，"
            f"severe_error={zs.get('severe_error',0)*100:.1f}%"
            f"（全局 {overall_severe*100:.1f}%）。"
            + ("**Zero-shot 是主要瓶颈**，需优先扩充对应细胞类型的训练数据。"
               if zs.get("severe_error", 0) > overall_severe + 0.1 else
               "Zero-shot 对整体影响有限。")
        )

    # tissue 偏移
    tissue_severes = {
        k: v.get("severe_error", 0)
        for k, v in report.get("by_tissue", {}).items()
        if v.get("n", 0) >= 5
    }
    if tissue_severes:
        worst_tissue = max(tissue_severes, key=tissue_severes.get)
        wt_severe = tissue_severes[worst_tissue]
        lines.append(
            f"- **组织偏移**：`{worst_tissue}` 的 severe_error 最高（{wt_severe*100:.1f}%），"
            + ("显著高于全局，说明该组织训练样本严重不足。"
               if wt_severe > overall_severe + 0.1 else "与全局差异不大。")
        )

    # hard subset
    hard = report.get("by_hardness", {})
    h_hard = hard.get("hard", {})
    h_normal = hard.get("normal", {})
    if h_hard.get("n", 0) > 0 and h_normal.get("n", 0) > 0:
        diff = h_hard.get("severe_error", 0) - h_normal.get("severe_error", 0)
        lines.append(
            f"- **Hard subset**：hard 样本 severe_error={h_hard.get('severe_error',0)*100:.1f}%，"
            f"normal={h_normal.get('severe_error',0)*100:.1f}%，差值 {diff*100:+.1f}pp。"
            + ("Hard 样本是显著瓶颈。" if diff > 0.1 else "Hard vs Normal 差异不大。")
        )

    if not lines:
        lines.append("数据量不足，无法自动生成可靠结论，请人工检阅上方各分桶表格。")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="错误分桶分析")
    p.add_argument(
        "--results",
        default=str(cfg.BEST_INFER_RESULTS),
        help="推理结果 JSONL 路径",
    )
    p.add_argument(
        "--output-dir",
        default=str(cfg.PROJECT_DIR / "output/diagnosis/buckets"),
        help="输出目录",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(Path(args.results), Path(args.output_dir))
