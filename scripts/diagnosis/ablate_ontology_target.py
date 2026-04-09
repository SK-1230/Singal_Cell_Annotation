"""
ablate_ontology_target.py — 验证 ontology_id 是否应继续作为直接生成目标

三种策略对比：
  direct_gen        模型直接生成 cell_type + ontology_id（当前策略）
  rule_postprocess  模型只生成 cell_type，用规则从 CL 映射 ontology_id
  parent_postprocess 模型生成 cell_type + parent，再后处理 ontology_id

本脚本基于已有推理结果做后处理分析，无需重新运行 GPU 推理。
规则映射：优先精确匹配 → 同义词匹配 → label 模糊匹配。

用法
----
  python scripts/diagnosis/ablate_ontology_target.py

  # 指定推理结果
  python scripts/diagnosis/ablate_ontology_target.py \\
      --results output/infer_kb_retrieval_20260405_211943/results.jsonl

输出
----
  output/diagnosis/ontology_ablation/
    ontology_comparison.md   — Markdown 对比报告
    rule_mapped.jsonl        — 规则映射后的结果
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_HERE))

import diag_config as cfg
from sca.diagnosis.metrics import MetricsDict
from sca.diagnosis.report_utils import ablation_comparison_table, section, verdict_block


# ─────────────────────────────────────────────
# Cell Ontology 规则映射器
# ─────────────────────────────────────────────

def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"[-_/,]", " ", t)
    t = re.sub(r"\bhuman\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


class OntologyRuleMapper:
    """
    给定 cell_type 字符串，用规则查找对应 ontology_id。

    优先级：精确 label 匹配 > 同义词匹配 > 父类回退
    """

    def __init__(self) -> None:
        self._label_to_id: Dict[str, str] = {}     # norm_label → id
        self._syn_to_id: Dict[str, str] = {}        # norm_synonym → id
        self._id_to_label: Dict[str, str] = {}

    def load(self, ontology_index_path: Path) -> None:
        with open(ontology_index_path) as f:
            for line in f:
                d = json.loads(line)
                oid = d.get("cell_ontology_id", "")
                label = _norm(d.get("label", ""))
                if oid and label:
                    self._label_to_id[label] = oid
                    self._id_to_label[oid] = label
                for syn in d.get("synonyms", []):
                    sl = _norm(syn)
                    if sl and oid:
                        self._syn_to_id[sl] = oid

    def lookup(self, cell_type: str) -> Optional[str]:
        nl = _norm(cell_type)
        # 1. 精确 label 匹配
        if nl in self._label_to_id:
            return self._label_to_id[nl]
        # 2. 同义词匹配
        if nl in self._syn_to_id:
            return self._syn_to_id[nl]
        # 3. 部分匹配：cell_type 是某个 label 的子串，或某 label 是子串
        for label, oid in self._label_to_id.items():
            if nl in label or label in nl:
                return oid
        return None


# ─────────────────────────────────────────────
# 指标计算（ontology_id 专项）
# ─────────────────────────────────────────────

def _compute_ont_metrics(
    results: List[Dict[str, Any]],
    pred_id_field: str = "pred_ontology_id",
) -> MetricsDict:
    """计算 ontology_id 准确率（仅对有 gold_ontology_id 的样本）。"""
    ont_samples = [r for r in results if r.get("gold_ontology_id", "")]
    n_total = len(results)
    n_ont = len(ont_samples)

    exact = sum(1 for r in results if r.get("cell_type_exact", False))
    compat = sum(1 for r in results if r.get("ontology_compatible", False))
    severe = sum(1 for r in results if r.get("cell_type_severe_error", False))

    if n_ont > 0:
        ont_correct = sum(
            1 for r in ont_samples
            if _norm(r.get(pred_id_field, "")) == _norm(r.get("gold_ontology_id", ""))
        )
        ont_acc = round(ont_correct / n_ont, 4)
    else:
        ont_acc = None

    return {
        "n": n_total,
        "exact": round(exact / n_total, 4) if n_total else 0,
        "ontology_compat": round(compat / n_total, 4) if n_total else 0,
        "severe_error": round(severe / n_total, 4) if n_total else 0,
        "parse_ok": round(sum(1 for r in results if r.get("parse_ok", True)) / n_total, 4) if n_total else 0,
        "ont_id_acc": ont_acc,
        "n_ont_evaluated": n_ont,
    }


# ─────────────────────────────────────────────
# 后处理：规则映射 ontology_id
# ─────────────────────────────────────────────

def _apply_rule_mapping(
    results: List[Dict[str, Any]],
    mapper: OntologyRuleMapper,
) -> List[Dict[str, Any]]:
    """在已有推理结果中，用规则替换 pred_ontology_id。"""
    out = []
    n_mapped = 0
    for r in results:
        r = dict(r)
        pred_ct = r.get("pred_cell_type", "")
        rule_id = mapper.lookup(pred_ct)
        r["pred_ontology_id_rule"] = rule_id or ""
        if rule_id:
            n_mapped += 1
        out.append(r)
    print(f"[ont_ablate] 规则映射成功率：{n_mapped}/{len(results)} ({n_mapped/max(len(results),1)*100:.1f}%)")
    return out


# ─────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────

def _build_report(
    mode_metrics: Dict[str, MetricsDict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Ontology Target 消融报告\n",
        "> 验证 ontology_id 是否应继续作为直接生成目标\n",
        section("1. 指标对比"),
        ablation_comparison_table(mode_metrics, "各策略对比"),
        "",
        section("2. 诊断结论"),
    ]

    direct = mode_metrics.get("direct_gen", {})
    rule = mode_metrics.get("rule_postprocess", {})

    if direct and rule:
        d_id = direct.get("ont_id_acc") or 0
        r_id = rule.get("ont_id_acc") or 0
        delta = r_id - d_id

        if delta >= 0.05:
            verdict = "**应将 ontology_id 改为后处理**"
            evidence = (
                f"规则映射 ont_id_acc={r_id*100:.1f}%，比模型直接生成（{d_id*100:.1f}%）"
                f"高 {delta*100:.1f}pp。"
                "建议下一轮移除 ontology_id 生成监督，改用规则后处理。"
            )
        elif delta >= 0:
            verdict = "**规则映射与直接生成相当，倾向于后处理**"
            evidence = (
                f"规则映射 {r_id*100:.1f}% vs 直接生成 {d_id*100:.1f}%，差异 {delta*100:.1f}pp。"
                "规则映射更可控（不产生幻觉 ID），建议切换。"
            )
        else:
            verdict = "**直接生成略优于规则映射**"
            evidence = (
                f"直接生成 {d_id*100:.1f}% vs 规则映射 {r_id*100:.1f}%，差异 {-delta*100:.1f}pp。"
                "当前模型直接生成略好，可暂时保留，但需确认 ID 是否幻觉。"
            )
        md_lines.append(verdict_block(
            "ontology_id 是否应改为后处理而非直接生成？",
            verdict,
            evidence,
        ))

    # 分析幻觉 ID（模型生成了不在 CL 中的 ID）
    md_lines.append(section("3. 幻觉 ID 分析（直接生成模式）"))
    md_lines.append(
        "\n> 见 `rule_mapped.jsonl` 中 `pred_ontology_id_rule==''` 的样本，\n"
        "> 这些样本的 pred_cell_type 无法被规则映射，说明预测的细胞类型不在官方 CL 中。\n"
    )

    report_path = output_dir / "ontology_comparison.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ont_ablate] 报告已写入：{report_path}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Ontology Target 消融实验")
    p.add_argument(
        "--results",
        default=str(cfg.BEST_INFER_RESULTS),
        help="推理结果 JSONL 路径",
    )
    p.add_argument(
        "--output-dir",
        default=str(_PROJECT / "output/diagnosis/ontology_ablation"),
        help="输出目录",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载推理结果 ──────────────────────────────
    results = []
    with open(args.results) as f:
        for line in f:
            results.append(json.loads(line))
    print(f"[ont_ablate] 加载 {len(results)} 条推理结果")

    # ── 加载 CL 映射器 ────────────────────────────
    mapper = OntologyRuleMapper()
    mapper.load(cfg.ONTOLOGY_INDEX)

    # ── direct_gen：使用模型生成的 pred_ontology_id ──
    m_direct = _compute_ont_metrics(results, pred_id_field="pred_ontology_id")
    print(
        f"[ont_ablate] direct_gen:      "
        f"exact={m_direct.get('exact',0)*100:.1f}%  "
        f"ont_id_acc={( m_direct.get('ont_id_acc') or 0)*100:.1f}%"
    )

    # ── rule_postprocess：用规则替换 pred_ontology_id ──
    mapped_results = _apply_rule_mapping(results, mapper)
    m_rule = _compute_ont_metrics(mapped_results, pred_id_field="pred_ontology_id_rule")
    print(
        f"[ont_ablate] rule_postprocess: "
        f"exact={m_rule.get('exact',0)*100:.1f}%  "
        f"ont_id_acc={(m_rule.get('ont_id_acc') or 0)*100:.1f}%"
    )

    # ── 写规则映射结果 JSONL ───────────────────────
    rule_path = output_dir / "rule_mapped.jsonl"
    with open(rule_path, "w") as f:
        for r in mapped_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ont_ablate] 规则映射结果已写入：{rule_path}")

    # ── 统计幻觉 ID ───────────────────────────────
    ont_samples = [r for r in results if r.get("gold_ontology_id", "")]
    if ont_samples:
        hallucinated = [
            r for r in ont_samples
            if r.get("pred_ontology_id") and not r.get("pred_ontology_id", "").startswith("CL:")
        ]
        fabricated = [
            r for r in ont_samples
            if r.get("pred_ontology_id", "").startswith("CL:")
            and r["pred_ontology_id"] not in mapper._label_to_id.values()
            and r["pred_ontology_id"] not in {
                oid for oid in mapper._label_to_id.values()
            }
        ]
        print(f"\n[ont_ablate] 幻觉 ID 分析（共 {len(ont_samples)} 个有 gold ID 的样本）：")
        print(f"  非 CL: 格式 ID：{len(hallucinated)} 个")

    # ── 生成报告 ──────────────────────────────────
    mode_metrics = {
        "direct_gen": m_direct,
        "rule_postprocess": m_rule,
    }
    _build_report(mode_metrics, output_dir)

    print("\n[ont_ablate] 消融完成。")
    print(f"  direct_gen:       ont_id_acc={(m_direct.get('ont_id_acc') or 0)*100:.1f}%")
    print(f"  rule_postprocess: ont_id_acc={(m_rule.get('ont_id_acc') or 0)*100:.1f}%")


if __name__ == "__main__":
    main()
