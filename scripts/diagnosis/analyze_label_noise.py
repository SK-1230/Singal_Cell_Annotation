"""
analyze_label_noise.py — 评估 SFT 训练数据中的标签噪声

检查每条训练样本的字段可靠性：
  - cell_type 是否在 Cell Ontology 合法标签中
  - ontology_id 格式是否合法且在 CL 中存在
  - parent_cell_type 是否为合法上位类
  - 综合评级：clean / weak / noisy / invalid

用法
----
  python scripts/diagnosis/analyze_label_noise.py

输出
----
  output/diagnosis/label_noise/
    label_noise_report.md     — Markdown 报告
    label_quality.csv         — 每条样本的质量评级 CSV
    top20_noisy_samples.json  — 典型问题样本（人工复核用）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_HERE))

from sca.diagnosis.label_quality import LabelQualityChecker
from sca.diagnosis.report_utils import (
    section, kv_table, noisy_samples_block, verdict_block
)
import diag_config as cfg


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[noise] 加载 Cell Ontology 资源...")
    checker = LabelQualityChecker()
    checker.load_ontology(cfg.ONTOLOGY_INDEX)
    if cfg.PARENT_SUPPLEMENT.exists():
        checker.load_parent_supplement(cfg.PARENT_SUPPLEMENT)
    # 补充 label_aliases.tsv 中的合法标签与 CL ID
    alias_tsv = _PROJECT / "resources/ontology/label_aliases.tsv"
    if alias_tsv.exists():
        checker.load_alias_table(alias_tsv)
        print(f"[noise] 已加载 alias table: {alias_tsv}")

    print(f"[noise] 检查训练集标签：{cfg.TRAIN_FULL}")
    records = checker.check_all(cfg.TRAIN_FULL)
    summary = checker.summarize(records)

    # ── 写 CSV ──────────────────────────────────────
    csv_path = output_dir / "label_quality.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "cell_type_clean", "cell_ontology_id", "cell_ontology_parent_label",
            "_quality", "_cell_type_ok", "_ont_id_ok", "_parent_ok", "_issues",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fieldnames}
            row["_issues"] = "; ".join(r.get("_issues", []))
            writer.writerow(row)
    print(f"[noise] CSV 已写入：{csv_path}")

    # ── 写 Top-20 问题样本 ─────────────────────────
    noisy_path = output_dir / "top20_noisy_samples.json"
    with open(noisy_path, "w", encoding="utf-8") as f:
        json.dump(summary["top20_noisy_examples"], f, ensure_ascii=False, indent=2)
    print(f"[noise] 问题样本已写入：{noisy_path}")

    # ── 生成 Markdown 报告 ────────────────────────
    qc = summary["quality_counts"]
    qr = summary["quality_rates"]
    n = summary["total"]

    noisy_rate = qr.get("noisy", 0) + qr.get("invalid", 0)
    weak_rate = qr.get("weak", 0)

    md_lines = [
        "# 标签噪声分析报告\n",
        f"> 训练集：`{cfg.TRAIN_FULL}`  \n> 样本总数：{n}\n",
        section("1. 质量分布"),
        kv_table({
            "clean": f"{qc.get('clean', 0)} ({qr.get('clean', 0)*100:.1f}%)",
            "weak": f"{qc.get('weak', 0)} ({qr.get('weak', 0)*100:.1f}%)",
            "noisy": f"{qc.get('noisy', 0)} ({qr.get('noisy', 0)*100:.1f}%)",
            "invalid": f"{qc.get('invalid', 0)} ({qr.get('invalid', 0)*100:.1f}%)",
        }),
        "",
        kv_table({
            "cell_type 在 CL 中": f"{summary['cell_type_ok_rate']*100:.1f}%",
            "ontology_id 合法": f"{summary['ont_id_ok_rate']*100:.1f}%",
            "parent 合法": f"{summary['parent_ok_rate']*100:.1f}%",
        }),
        section("2. 诊断结论"),
        _auto_verdict(noisy_rate, weak_rate, summary),
        section("3. Top-20 典型问题样本"),
        noisy_samples_block(summary["top20_noisy_examples"]),
    ]

    md_path = output_dir / "label_noise_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[noise] 报告已写入：{md_path}")

    # ── 打印关键结论 ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"  clean   : {qc.get('clean', 0):3d} ({qr.get('clean', 0)*100:.1f}%)")
    print(f"  weak    : {qc.get('weak', 0):3d} ({qr.get('weak', 0)*100:.1f}%)")
    print(f"  noisy   : {qc.get('noisy', 0):3d} ({qr.get('noisy', 0)*100:.1f}%)")
    print(f"  invalid : {qc.get('invalid', 0):3d} ({qr.get('invalid', 0)*100:.1f}%)")
    print(f"  ontology_id 合法率：{summary['ont_id_ok_rate']*100:.1f}%")
    print("=" * 60)


def _auto_verdict(noisy_rate: float, weak_rate: float, summary: dict) -> str:
    ont_ok = summary["ont_id_ok_rate"]
    ct_ok = summary["cell_type_ok_rate"]

    if noisy_rate >= 0.2:
        verdict = "**标签噪声严重，必须优先处理**"
        evidence = (
            f"noisy + invalid 占比 {noisy_rate*100:.1f}%，超过 20% 阈值。"
            "建议立即重建训练标签（Phase 2），否则扩数据效果有限。"
        )
    elif noisy_rate + weak_rate >= 0.3:
        verdict = "**标签质量中等，建议在扩数据前先净化**"
        evidence = (
            f"noisy + invalid + weak 合计 {(noisy_rate + weak_rate)*100:.1f}%。"
            "ontology_id 合法率仅 {:.1f}%，建议清洗后单独验证 clean 子集效果。".format(
                ont_ok * 100
            )
        )
    else:
        verdict = "**标签质量尚可，噪声不是首要瓶颈**"
        evidence = (
            f"noisy + invalid 仅 {noisy_rate*100:.1f}%。"
            "主要问题可能在数据量或 zero-shot 覆盖，而非标签质量。"
        )

    lines = [verdict_block("当前自蒸馏标签噪声是否严重到必须先处理？", verdict, evidence)]

    if ont_ok < 0.5:
        lines.append(
            f"\n> **注意**：ontology_id 合法率仅 {ont_ok*100:.1f}%（<50%），"
            "建议将 ontology_id 直接生成目标移除，改用规则后处理（见 ablate_ontology_target.py）。\n"
        )
    return "\n".join(lines)


def _parse_args():
    p = argparse.ArgumentParser(description="标签噪声分析")
    p.add_argument(
        "--output-dir",
        default=str(_PROJECT / "output/diagnosis/label_noise"),
        help="输出目录",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(Path(args.output_dir))
