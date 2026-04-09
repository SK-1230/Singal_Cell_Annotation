"""
ablate_kb_retrieval.py — 验证 KB 检索是增益还是噪声

三种模式对比：
  jaccard_kb      当前 Jaccard 相似度检索（baseline，复用已有结果）
  no_kb           禁用 KB，纯模型推理
  oracle_hint     用 gold parent cell type 作为提示（理论上限）

用法
----
  # 完整消融（需 GPU，运行 no_kb 和 oracle 两次推理）
  CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/ablate_kb_retrieval.py

  # 只跑 no_kb
  CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/ablate_kb_retrieval.py --modes no_kb

  # 只分析已有结果
  python scripts/diagnosis/ablate_kb_retrieval.py --no-run

输出
----
  output/diagnosis/kb_ablation/
    kb_comparison.md     — Markdown 对比报告
    no_kb/               — no_kb 模式推理结果
    oracle_hint/         — oracle_hint 模式推理结果
"""

from __future__ import annotations

import argparse
import json
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
# Oracle hint：将 gold parent_cell_type 注入 prompt
# ─────────────────────────────────────────────

def _make_oracle_test_file(test_full_path: Path, tmp_dir: Path) -> Path:
    """
    生成 oracle 版测试文件：将 gold parent_cell_type 作为完美检索结果注入
    user 消息末尾，模拟"理想 KB"的上限。
    """
    out_path = tmp_dir / "test_oracle_hint.jsonl"
    with open(test_full_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            d = json.loads(line)
            parent = d.get("cell_ontology_parent_label", "") or ""
            cell_type = d.get("cell_type_clean", "") or ""

            if parent:
                hint = (
                    "\n--- Oracle Hint (ground-truth parent class) ---\n"
                    f"  The cell belongs to the broader category: {parent}\n"
                    "  Use this as supporting context, but determine the exact subtype yourself.\n"
                    "---"
                )
            else:
                hint = ""

            msgs = d.get("messages_no_think") or d.get("messages") or []
            new_msgs = []
            for m in msgs:
                if m.get("role") == "user" and hint:
                    new_msgs.append({
                        "role": "user",
                        "content": m["content"] + "\n" + hint,
                    })
                elif m.get("role") != "assistant":
                    new_msgs.append(m)
            d["messages_no_think"] = new_msgs
            d["messages"] = new_msgs
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    return out_path


# ─────────────────────────────────────────────
# 调用推理脚本
# ─────────────────────────────────────────────

def _run_infer(
    test_file: Path,
    output_dir: Path,
    no_kb: bool = False,
) -> int:
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
    print(f"[kb_ablate] 运行推理：{' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _load_results(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────

def _build_report(
    mode_metrics: Dict[str, MetricsDict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# KB 检索消融报告\n",
        "> 验证 KB 检索是增益、无效、还是噪声\n",
        section("1. 指标对比"),
        ablation_comparison_table(mode_metrics, "各 KB 模式对比"),
        "",
        section("2. 诊断结论"),
    ]

    jaccard = mode_metrics.get("jaccard_kb", {})
    no_kb = mode_metrics.get("no_kb", {})
    oracle = mode_metrics.get("oracle_hint", {})

    if jaccard and no_kb:
        delta_exact = (
            (jaccard.get("exact", 0) or 0) - (no_kb.get("exact", 0) or 0)
        )
        delta_severe = (
            (jaccard.get("severe_error", 0) or 0) - (no_kb.get("severe_error", 0) or 0)
        )

        if delta_exact < -0.02 or delta_severe > 0.02:
            verdict = "**当前 KB 是噪声源**"
            evidence = (
                f"no_kb 比 jaccard_kb 更好（exact {-delta_exact*100:+.1f}pp，"
                f"severe {-delta_severe*100:+.1f}pp）。"
                "建议暂时禁用 KB，或大幅改进检索质量后再启用。"
            )
        elif delta_exact >= 0.05:
            verdict = "**当前 KB 有稳定正增益**"
            evidence = (
                f"jaccard_kb 比 no_kb exact 高 {delta_exact*100:.1f}pp。"
                "KB 有效，可继续优化检索质量。"
            )
        else:
            verdict = "**当前 KB 增益有限（不稳定）**"
            evidence = (
                f"jaccard_kb vs no_kb exact 差异 {delta_exact*100:+.1f}pp，"
                f"severe 差异 {delta_severe*100:+.1f}pp。"
                "KB 未形成稳定收益，当前 Jaccard 检索质量不足，"
                "建议先不投入大量精力优化检索。"
            )
        md_lines.append(verdict_block(
            "KB 检索是否形成稳定正增益？",
            verdict,
            evidence,
        ))

    if oracle and jaccard:
        oracle_exact = oracle.get("exact", 0) or 0
        jac_exact = jaccard.get("exact", 0) or 0
        gap = oracle_exact - jac_exact
        if gap >= 0.05:
            md_lines.append(verdict_block(
                "如果有完美检索（知道 gold parent），性能能提升多少？",
                f"**理论上限 exact={oracle_exact*100:.1f}%（当前 {jac_exact*100:.1f}%，gap={gap*100:.1f}pp）**",
                "改善检索质量有较大提升空间，值得投入优化。",
            ))
        else:
            md_lines.append(verdict_block(
                "如果有完美检索，性能能提升多少？",
                f"**理论上限提升有限（gap={gap*100:.1f}pp）**",
                "即使检索完美，性能提升也有限，瓶颈不在检索质量。",
            ))

    report_path = output_dir / "kb_comparison.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[kb_ablate] 报告已写入：{report_path}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="KB 检索消融实验")
    p.add_argument(
        "--no-run", action="store_true",
        help="不运行推理，只分析已有结果",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["no_kb", "oracle_hint"],
        choices=["no_kb", "oracle_hint"],
        help="要运行的消融模式",
    )
    p.add_argument(
        "--jaccard-results",
        default=str(cfg.BEST_INFER_RESULTS),
        help="jaccard_kb 模式的已有推理结果（默认复用最优结果）",
    )
    p.add_argument(
        "--output-dir",
        default=str(_PROJECT / "output/diagnosis/kb_ablation"),
        help="输出目录",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    mode_metrics: Dict[str, MetricsDict] = {}

    # ── jaccard_kb baseline ──────────────────────────
    jac_path = Path(args.jaccard_results)
    if jac_path.exists():
        rows = _load_results(jac_path)
        mode_metrics["jaccard_kb"] = compute_metrics(rows)
        print(f"[kb_ablate] jaccard_kb: exact={mode_metrics['jaccard_kb'].get('exact',0)*100:.1f}%")
    else:
        print(f"[kb_ablate] 找不到 jaccard_kb 结果：{jac_path}")

    if not args.no_run:
        with tempfile.TemporaryDirectory(prefix="sca_kb_") as tmp:
            tmp_path = Path(tmp)

            # ── no_kb ──────────────────────────────────────
            if "no_kb" in args.modes:
                print("\n[kb_ablate] 运行 no_kb 模式...")
                no_kb_out = output_dir / "no_kb"
                rc = _run_infer(cfg.TEST_FULL, no_kb_out, no_kb=True)
                if rc == 0:
                    results_path = no_kb_out / "results.jsonl"
                    if results_path.exists():
                        rows = _load_results(results_path)
                        mode_metrics["no_kb"] = compute_metrics(rows)
                        m = mode_metrics["no_kb"]
                        print(
                            f"[kb_ablate] no_kb: "
                            f"exact={m.get('exact',0)*100:.1f}%  "
                            f"severe={m.get('severe_error',0)*100:.1f}%"
                        )

            # ── oracle_hint ────────────────────────────────
            if "oracle_hint" in args.modes:
                print("\n[kb_ablate] 运行 oracle_hint 模式...")
                oracle_test = _make_oracle_test_file(cfg.TEST_FULL, tmp_path)
                oracle_out = output_dir / "oracle_hint"
                rc = _run_infer(oracle_test, oracle_out, no_kb=True)
                if rc == 0:
                    results_path = oracle_out / "results.jsonl"
                    if results_path.exists():
                        rows = _load_results(results_path)
                        mode_metrics["oracle_hint"] = compute_metrics(rows)
                        m = mode_metrics["oracle_hint"]
                        print(
                            f"[kb_ablate] oracle_hint: "
                            f"exact={m.get('exact',0)*100:.1f}%  "
                            f"severe={m.get('severe_error',0)*100:.1f}%"
                        )
    else:
        # ── 只读已有结果 ────────────────────────────────
        for mode in args.modes:
            mode_path = output_dir / mode / "results.jsonl"
            if mode_path.exists():
                rows = _load_results(mode_path)
                mode_metrics[mode] = compute_metrics(rows)
                print(f"[kb_ablate] {mode}: exact={mode_metrics[mode].get('exact',0)*100:.1f}%")
            else:
                print(f"[kb_ablate] 找不到 {mode} 结果，跳过")

    if mode_metrics:
        _build_report(mode_metrics, output_dir)
        print("\n[kb_ablate] 消融完成。")
        for mode, m in mode_metrics.items():
            print(
                f"  {mode:15s}: exact={m.get('exact',0)*100:.1f}%  "
                f"severe={m.get('severe_error',0)*100:.1f}%"
            )
    else:
        print("[kb_ablate] 无可用结果，终止。")


if __name__ == "__main__":
    main()
