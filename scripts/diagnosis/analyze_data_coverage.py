"""
analyze_data_coverage.py — 量化数据覆盖缺口

统计维度：
  - 每个 tissue 的 train/val/test 样本数
  - 每个细胞类型的训练集出现次数
  - zero-shot 类型列表
  - rare 类型列表（train ≤ 2）
  - ontology 映射率
  - marker quality 分布
  - 最该优先补充的数据类型

用法
----
  python scripts/diagnosis/analyze_data_coverage.py

输出
----
  output/diagnosis/data_coverage/
    data_coverage_report.md   — Markdown 报告
    tissue_coverage.csv
    celltype_coverage.csv
    zero_shot_types.txt
    rare_types.txt
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Any, Counter, Dict, List, Set, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_HERE))

from sca.diagnosis.report_utils import section, kv_table
import diag_config as cfg


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_jsonl(cfg.TRAIN_FULL)
    val = _load_jsonl(cfg.VAL_FULL)
    test = _load_jsonl(cfg.TEST_FULL)

    # ── 细胞类型统计 ──────────────────────────────
    train_ct: Counter[str] = collections.Counter(
        r.get("cell_type_clean", "").lower() for r in train
    )
    test_cts: Set[str] = {r.get("cell_type_clean", "").lower() for r in test}
    val_cts: Set[str] = {r.get("cell_type_clean", "").lower() for r in val}

    zero_shot = sorted(test_cts - set(train_ct.keys()))
    rare = sorted(ct for ct, n in train_ct.items() if 0 < n <= 2)
    singletons = sorted(ct for ct, n in train_ct.items() if n == 1)

    # ── 组织统计 ──────────────────────────────────
    def _tissue_counts(rows):
        c: Counter[str] = collections.Counter()
        for r in rows:
            t = r.get("tissue_general", "unknown")
            c[t] += 1
        return c

    tissue_train = _tissue_counts(train)
    tissue_val = _tissue_counts(val)
    tissue_test = _tissue_counts(test)
    all_tissues = sorted(
        set(tissue_train) | set(tissue_val) | set(tissue_test)
    )

    # ── Marker quality 分布 ───────────────────────
    mq_buckets = {"high(≥0.7)": 0, "mid(0.5-0.7)": 0, "low(<0.5)": 0, "N/A": 0}
    for r in train:
        mq = r.get("marker_quality_score")
        if mq is None:
            mq_buckets["N/A"] += 1
        elif mq >= 0.7:
            mq_buckets["high(≥0.7)"] += 1
        elif mq >= 0.5:
            mq_buckets["mid(0.5-0.7)"] += 1
        else:
            mq_buckets["low(<0.5)"] += 1

    # ── Ontology 映射率 ───────────────────────────
    valid_ont_ids: Set[str] = set()
    with open(cfg.ONTOLOGY_INDEX) as f:
        for line in f:
            d = json.loads(line)
            oid = d.get("cell_ontology_id", "")
            if oid:
                valid_ont_ids.add(oid)

    ont_valid_train = sum(
        1 for r in train if r.get("cell_ontology_id", "") in valid_ont_ids
    )
    ont_valid_test = sum(
        1 for r in test if r.get("cell_ontology_id", "") in valid_ont_ids
    )

    # ── 确定最该补的数据类型 ──────────────────────
    # 按测试集 tissue 中训练样本最少的排序
    priority_tissues = sorted(
        all_tissues,
        key=lambda t: tissue_train.get(t, 0),
    )

    # ── 写 CSV ────────────────────────────────────
    tissue_csv = output_dir / "tissue_coverage.csv"
    with open(tissue_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tissue", "train", "val", "test", "in_test_not_train"])
        for t in all_tissues:
            in_test = t in {r.get("tissue_general", "") for r in test}
            in_train = t in tissue_train
            w.writerow([
                t,
                tissue_train.get(t, 0),
                tissue_val.get(t, 0),
                tissue_test.get(t, 0),
                "yes" if in_test and not in_train else "",
            ])

    ct_csv = output_dir / "celltype_coverage.csv"
    all_cts = sorted(
        set(train_ct.keys()) | test_cts | val_cts,
        key=lambda x: -train_ct.get(x, 0),
    )
    with open(ct_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_type", "train_count", "in_val", "in_test", "status"])
        for ct in all_cts:
            tc = train_ct.get(ct, 0)
            in_val = ct in val_cts
            in_test = ct in test_cts
            if tc == 0:
                status = "zero_shot"
            elif tc == 1:
                status = "singleton"
            elif tc <= 2:
                status = "rare"
            elif tc <= 5:
                status = "few_shot"
            else:
                status = "seen"
            w.writerow([ct, tc, in_val, in_test, status])

    (output_dir / "zero_shot_types.txt").write_text("\n".join(zero_shot), encoding="utf-8")
    (output_dir / "rare_types.txt").write_text("\n".join(rare), encoding="utf-8")

    # ── Markdown 报告 ─────────────────────────────
    n_train = len(train)
    n_test = len(test)
    n_val = len(val)
    n_unique_train_ct = len(train_ct)

    md_lines = [
        "# 数据覆盖分析报告\n",
        section("1. 数据集规模"),
        kv_table({
            "训练集样本数": n_train,
            "验证集样本数": n_val,
            "测试集样本数": n_test,
            "训练集 unique cell types": n_unique_train_ct,
            "测试集 unique cell types": len(test_cts),
        }),
        section("2. 细胞类型覆盖"),
        kv_table({
            "zero-shot（仅在测试集）": len(zero_shot),
            "singleton（训练集仅 1 条）": len(singletons),
            "rare（训练集 ≤ 2 条）": len(rare),
            "zero-shot 列表": ", ".join(zero_shot[:10]) + ("..." if len(zero_shot) > 10 else ""),
        }),
        section("3. 组织覆盖"),
        "\n**各组织样本数（train / val / test）：**\n",
        "| tissue | train | val | test |",
        "|--------|-------|-----|------|",
    ]
    for t in all_tissues:
        md_lines.append(
            f"| {t} | {tissue_train.get(t,0)} | {tissue_val.get(t,0)} | {tissue_test.get(t,0)} |"
        )

    md_lines += [
        "",
        section("4. Marker Quality 分布（训练集）"),
        kv_table(mq_buckets),
        section("5. Ontology ID 映射率"),
        kv_table({
            "训练集合法 CL ID": f"{ont_valid_train}/{n_train} ({ont_valid_train/n_train*100:.1f}%)",
            "测试集合法 CL ID": f"{ont_valid_test}/{n_test} ({ont_valid_test/n_test*100:.1f}%)",
        }),
        section("6. 优先补充建议"),
        _priority_advice(
            zero_shot, singletons, tissue_train, tissue_test,
            ont_valid_train, n_train
        ),
    ]

    md_path = output_dir / "data_coverage_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[coverage] 报告已写入：{md_path}")

    # ── 终端摘要 ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  训练集样本：{n_train}  |  unique types：{n_unique_train_ct}")
    print(f"  zero-shot：{len(zero_shot)}  |  singleton：{len(singletons)}  |  rare：{len(rare)}")
    print(f"  ontology_id 合法率（训练）：{ont_valid_train/n_train*100:.1f}%")
    print(f"  最需补充的 tissue：{priority_tissues[:3]}")
    print("=" * 60)


def _priority_advice(
    zero_shot: List[str],
    singletons: List[str],
    tissue_train: Dict,
    tissue_test: Dict,
    ont_valid: int,
    n_train: int,
) -> str:
    lines = []

    # 测试集有但训练集样本最少的 tissue
    test_tissues_sorted = sorted(
        [t for t in tissue_test if tissue_test[t] > 0],
        key=lambda t: tissue_train.get(t, 0),
    )
    if test_tissues_sorted:
        lines.append(
            f"1. **最该补充的组织**（测试集有但训练样本少）：`{', '.join(test_tissues_sorted[:3])}`"
        )

    if zero_shot:
        lines.append(
            f"2. **Zero-shot 细胞类型**（必须补）：共 {len(zero_shot)} 种，"
            f"包括：`{', '.join(zero_shot[:5])}`"
            + ("..." if len(zero_shot) > 5 else "")
        )

    if len(singletons) > 20:
        lines.append(
            f"3. **Singleton 类型**（{len(singletons)} 种只有 1 条训练样本）：建议优先补充测试集中出现的 singleton 类型"
        )

    if ont_valid / n_train < 0.5:
        lines.append(
            f"4. **Ontology ID 质量**：合法率仅 {ont_valid/n_train*100:.1f}%，"
            "建议引入专家标注数据（Tabula Sapiens / HCA）替代自蒸馏标签"
        )

    return "\n".join(lines) if lines else "_无特别建议_"


def _parse_args():
    p = argparse.ArgumentParser(description="数据覆盖分析")
    p.add_argument(
        "--output-dir",
        default=str(_PROJECT / "output/diagnosis/data_coverage"),
        help="输出目录",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(Path(args.output_dir))
