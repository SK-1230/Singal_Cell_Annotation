"""
ablate_output_schema.py — 验证"训练目标过宽"是否是主瓶颈

三种 schema 模式对比：
  full_json       当前完整输出（cell_type + 11 个辅助字段）
  cell_type_only  只输出 {"cell_type": "..."}
  cell_type_parent 只输出 {"cell_type": "...", "parent_cell_type": "..."}

实现策略（两阶段）
------------------
阶段一（inference-only，无需 GPU）：
  对当前已训练模型，修改 system prompt 要求简化输出，
  通过子进程调用 infer_qwen3_kb_retrieval.py，
  快速得到近似消融信号（当前模型的"可塑性"测试）。

阶段二（需重训，生成简化训练数据）：
  运行本脚本时加 --prep-train-data，将在 data/sft/ 下
  生成 sft_messages_cell_type_only.jsonl 等文件，
  可直接作为 train_config.yaml 的 train_file 使用。

用法
----
  # 阶段一：仅做推理侧消融（需 GPU，~5-10 min）
  CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/ablate_output_schema.py

  # 阶段二：生成简化训练数据（无需 GPU）
  python scripts/diagnosis/ablate_output_schema.py --prep-train-data

  # 只分析已有结果（无需 GPU）
  python scripts/diagnosis/ablate_output_schema.py --no-run \\
      --full-results output/infer_kb_retrieval_20260405_211943/results.jsonl

输出
----
  output/diagnosis/schema_ablation/
    schema_comparison.md    — Markdown 对比报告
    full_json/              — full_json 模式推理结果（复用已有或新跑）
    cell_type_only/         — cell_type_only 模式推理结果
    cell_type_parent/       — cell_type_parent 模式推理结果
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_HERE))

from sca.diagnosis.metrics import compute_metrics, MetricsDict
from sca.diagnosis.report_utils import ablation_comparison_table, section, verdict_block
import diag_config as cfg


# ─────────────────────────────────────────────
# Schema 定义
# ─────────────────────────────────────────────

SCHEMAS = {
    "full_json": {
        "system_suffix": "",  # 不修改 system prompt
        "description": "当前完整输出（11 字段 JSON）",
    },
    "cell_type_only": {
        "system_suffix": (
            "\n\nIMPORTANT: Output ONLY a minimal JSON with a SINGLE field:\n"
            '{"cell_type": "<cell type name>"}\n'
            "Do NOT include any other fields (no ontology_id, rationale, confidence, etc.)."
        ),
        "description": "只输出 {\"cell_type\": \"...\"}",
    },
    "cell_type_parent": {
        "system_suffix": (
            "\n\nIMPORTANT: Output ONLY a JSON with TWO fields:\n"
            '{"cell_type": "<cell type name>", "parent_cell_type": "<parent cell type>"}\n'
            "Do NOT include any other fields."
        ),
        "description": "只输出 cell_type + parent_cell_type",
    },
}


# ─────────────────────────────────────────────
# 推理结果解析（适配简化 schema）
# ─────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """从模型输出中提取 JSON（兼容 <think> 标签和 Markdown 代码块）。"""
    # 去掉 thinking 部分
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 尝试 Markdown 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 尝试裸 JSON
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _normalize(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"[-_/,]", " ", t)
    t = re.sub(r"\bhuman\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────
# 修改 system prompt，生成临时测试文件
# ─────────────────────────────────────────────

def _make_schema_test_file(
    schema_name: str, test_full_path: Path, tmp_dir: Path
) -> Path:
    """
    将测试集 messages 中的 system prompt 追加 schema 指令，
    生成临时测试 JSONL 文件供推理脚本使用。
    """
    suffix = SCHEMAS[schema_name]["system_suffix"]
    out_path = tmp_dir / f"test_{schema_name}.jsonl"
    with open(test_full_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            msgs = d.get("messages_no_think") or d.get("messages") or []
            new_msgs = []
            for m in msgs:
                if m.get("role") == "system" and suffix:
                    new_msgs.append({"role": "system", "content": m["content"] + suffix})
                elif m.get("role") != "assistant":
                    new_msgs.append(m)
            d["messages_no_think"] = new_msgs
            d["messages"] = new_msgs
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    return out_path


# ─────────────────────────────────────────────
# 调用推理脚本（子进程）
# ─────────────────────────────────────────────

def _run_infer(test_file: Path, output_dir: Path, no_kb: bool = False) -> int:
    """调用 infer_qwen3_kb_retrieval.py，返回返回码。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(cfg.INFER_SCRIPT),
        "--base-model", str(cfg.BASE_MODEL),
        "--adapter", str(cfg.BEST_ADAPTER),
        "--test-file", str(test_file),
        "--output-dir", str(output_dir),
    ]
    if no_kb:
        cmd.append("--no-kb")
    print(f"[schema] 运行推理：{' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


# ─────────────────────────────────────────────
# 解析简化 schema 的推理结果
# ─────────────────────────────────────────────

def _parse_schema_results(
    results_jsonl: Path,
    schema_name: str,
) -> List[Dict[str, Any]]:
    """
    对 full_json 外的 schema，重新解析 raw_output，
    提取 cell_type 并重新计算 cell_type_exact / severe_error 等指标。
    """
    if schema_name == "full_json":
        rows = []
        with open(results_jsonl) as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    rows = []
    with open(results_jsonl) as f:
        for line in f:
            r = json.loads(line)
            raw = r.get("raw_output", "")
            parsed = _extract_json(raw) if raw else None

            pred_ct = ""
            parse_ok = False
            if parsed and isinstance(parsed, dict):
                pred_ct = _normalize(parsed.get("cell_type", ""))
                parse_ok = bool(pred_ct)

            gold_ct = _normalize(r.get("gold_cell_type", ""))
            exact = (pred_ct == gold_ct) if (pred_ct and gold_ct) else False

            r["pred_cell_type"] = pred_ct
            r["parse_ok"] = parse_ok
            r["cell_type_exact"] = exact
            # 对简化 schema，无法计算 ontology_compatible / severe_error
            # 保留原字段但标记为需要注意
            rows.append(r)
    return rows


# ─────────────────────────────────────────────
# 生成简化训练数据
# ─────────────────────────────────────────────

def _prep_train_data() -> None:
    """
    从现有 sft_records_full_v2.jsonl 生成简化版训练数据：
      - sft_messages_cell_type_only.jsonl
      - sft_messages_cell_type_parent.jsonl
    """
    sft_records = _PROJECT / "data/sft/sft_records_full_v2.jsonl"
    if not sft_records.exists():
        print(f"[schema] 找不到 {sft_records}，跳过训练数据生成")
        return

    system_prompt = (
        "You are a transcriptomics assistant specialized in single-cell RNA-seq "
        "cell type annotation. Use the ranked marker genes and biological context "
        "to infer the most likely cell type. Be concise, structured, and avoid overclaiming."
    )

    out_ct_only = _PROJECT / "data/sft/sft_messages_cell_type_only.jsonl"
    out_ct_parent = _PROJECT / "data/sft/sft_messages_cell_type_parent.jsonl"

    n = 0
    with (
        open(sft_records) as fin,
        open(out_ct_only, "w") as f_ct,
        open(out_ct_parent, "w") as f_cp,
    ):
        for line in fin:
            d = json.loads(line)
            # 从原始 messages_no_think 取 user 消息
            orig_msgs = d.get("messages_no_think") or d.get("messages") or []
            user_msg = next((m["content"] for m in orig_msgs if m["role"] == "user"), "")
            if not user_msg:
                continue

            cell_type = d.get("cell_type_clean", "")
            parent = d.get("cell_ontology_parent_label", "")

            # cell_type_only
            assistant_ct = json.dumps({"cell_type": cell_type}, ensure_ascii=False)
            msgs_ct = [
                {"role": "system", "content": system_prompt
                 + "\n\nOutput ONLY: {\"cell_type\": \"<name>\"}"},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "<think>\n\n</think>\n\n" + assistant_ct},
            ]
            f_ct.write(json.dumps({"messages": msgs_ct}, ensure_ascii=False) + "\n")

            # cell_type_parent
            assistant_cp = json.dumps(
                {"cell_type": cell_type, "parent_cell_type": parent or ""},
                ensure_ascii=False,
            )
            msgs_cp = [
                {"role": "system", "content": system_prompt
                 + "\n\nOutput ONLY: {\"cell_type\": \"<name>\", \"parent_cell_type\": \"<parent>\"}"},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "<think>\n\n</think>\n\n" + assistant_cp},
            ]
            f_cp.write(json.dumps({"messages": msgs_cp}, ensure_ascii=False) + "\n")
            n += 1

    print(f"[schema] 生成简化训练数据：{n} 条")
    print(f"  cell_type_only  → {out_ct_only}")
    print(f"  cell_type_parent → {out_ct_parent}")
    print()
    print("[schema] 下一步：修改 train_config.yaml 中的 train_file，")
    print("  设为上述文件路径后重新训练，再用 --no-run + --full-results 比较结果。")


# ─────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────

def _build_report(
    mode_metrics: Dict[str, MetricsDict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Schema 消融报告\n",
        "> 验证\"训练目标过宽\"是否是主瓶颈\n",
        section("1. 指标对比"),
        ablation_comparison_table(mode_metrics, "各 Schema 模式对比"),
        "",
        section("2. 诊断结论"),
    ]

    base = mode_metrics.get("full_json", {})
    ct_only = mode_metrics.get("cell_type_only", {})

    if ct_only and base:
        delta_exact = (ct_only.get("exact", 0) or 0) - (base.get("exact", 0) or 0)
        delta_severe = (ct_only.get("severe_error", 0) or 0) - (base.get("severe_error", 0) or 0)

        if delta_exact >= 0.05 or delta_severe <= -0.05:
            verdict = "**任务定义过宽是主瓶颈**"
            evidence = (
                f"cell_type_only 相比 full_json：exact {delta_exact*100:+.1f}pp，"
                f"severe_error {delta_severe*100:+.1f}pp。"
                "建议立即切换到 cell_type_only 训练目标（Phase 1）。"
            )
        elif delta_exact >= 0.02 or delta_severe <= -0.02:
            verdict = "**任务定义过宽有一定影响，但非唯一瓶颈**"
            evidence = (
                f"cell_type_only 相比 full_json：exact {delta_exact*100:+.1f}pp，"
                f"severe_error {delta_severe*100:+.1f}pp。"
                "轻微影响，可尝试简化但需同步改进数据质量。"
            )
        else:
            verdict = "**任务定义过宽不是主瓶颈**"
            evidence = (
                f"cell_type_only vs full_json exact 差异 {delta_exact*100:+.1f}pp，"
                "简化 schema 无明显收益，瓶颈在其他方面（数据量/标签质量）。"
            )

        md_lines.append(verdict_block(
            "full_json 的复杂 schema 是否拖累了主任务学习？",
            verdict,
            evidence,
        ))

        md_lines.append(
            "\n> **注意**：此结论基于推理侧 schema 修改（非重训），"
            "仅反映现有模型对简化指令的响应能力。"
            "完整消融需生成简化训练数据并重训（运行 `--prep-train-data`）。\n"
        )
    else:
        md_lines.append("\n_缺少对比数据，无法生成自动结论。_\n")

    report_path = output_dir / "schema_comparison.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[schema] 报告已写入：{report_path}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Schema 消融实验")
    p.add_argument(
        "--no-run", action="store_true",
        help="不运行推理，只分析已有结果"
    )
    p.add_argument(
        "--prep-train-data", action="store_true",
        help="生成简化版训练数据文件（无需 GPU）"
    )
    p.add_argument(
        "--full-results",
        default=str(cfg.BEST_INFER_RESULTS),
        help="full_json 模式的推理结果路径（默认复用最优结果）",
    )
    p.add_argument(
        "--output-dir",
        default=str(_PROJECT / "output/diagnosis/schema_ablation"),
        help="输出目录",
    )
    p.add_argument(
        "--schemas",
        nargs="+",
        default=["cell_type_only", "cell_type_parent"],
        choices=["cell_type_only", "cell_type_parent"],
        help="要消融的 schema 模式（full_json 始终作为 baseline）",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if args.prep_train_data:
        _prep_train_data()
        return

    mode_metrics: Dict[str, MetricsDict] = {}

    # ── full_json baseline（复用已有结果）──────────────
    full_results_path = Path(args.full_results)
    if full_results_path.exists():
        rows = _parse_schema_results(full_results_path, "full_json")
        mode_metrics["full_json"] = compute_metrics(rows)
        print(f"[schema] full_json: exact={mode_metrics['full_json'].get('exact',0)*100:.1f}%")
    else:
        print(f"[schema] 找不到 full_json 结果：{full_results_path}")

    if not args.no_run:
        # ── 对每个简化 schema 运行推理 ─────────────────
        with tempfile.TemporaryDirectory(prefix="sca_schema_") as tmp:
            tmp_path = Path(tmp)
            for schema_name in args.schemas:
                print(f"\n[schema] 运行 {schema_name} 模式推理...")
                test_file = _make_schema_test_file(schema_name, cfg.TEST_FULL, tmp_path)
                schema_out_dir = output_dir / schema_name
                rc = _run_infer(test_file, schema_out_dir)
                if rc != 0:
                    print(f"[schema] 推理失败（返回码 {rc}），跳过 {schema_name}")
                    continue
                results_jsonl = schema_out_dir / "results.jsonl"
                if results_jsonl.exists():
                    rows = _parse_schema_results(results_jsonl, schema_name)
                    mode_metrics[schema_name] = compute_metrics(rows)
                    m = mode_metrics[schema_name]
                    print(
                        f"[schema] {schema_name}: "
                        f"exact={m.get('exact',0)*100:.1f}%  "
                        f"severe={m.get('severe_error',0)*100:.1f}%  "
                        f"parse={m.get('parse_ok',0)*100:.1f}%"
                    )
    else:
        # ── 只分析已有结果 ────────────────────────────
        for schema_name in args.schemas:
            schema_results = output_dir / schema_name / "results.jsonl"
            if schema_results.exists():
                rows = _parse_schema_results(schema_results, schema_name)
                mode_metrics[schema_name] = compute_metrics(rows)
                print(
                    f"[schema] {schema_name}: "
                    f"exact={mode_metrics[schema_name].get('exact',0)*100:.1f}%"
                )
            else:
                print(f"[schema] 找不到 {schema_name} 结果，跳过")

    if mode_metrics:
        _build_report(mode_metrics, output_dir)
        print("\n[schema] 消融完成。")
        for mode, m in mode_metrics.items():
            print(f"  {mode:20s}: exact={m.get('exact',0)*100:.1f}%  severe={m.get('severe_error',0)*100:.1f}%")
    else:
        print("[schema] 无可用结果，终止。")


if __name__ == "__main__":
    main()
