"""
run_all_diagnosis.py — 一键运行所有诊断脚本

执行顺序：
  1. analyze_error_buckets   （无需 GPU）
  2. ablate_ontology_target  （无需 GPU）
  3. analyze_label_noise     （无需 GPU）
  4. analyze_data_coverage   （无需 GPU）
  5. ablate_kb_retrieval     （需 GPU）
  6. ablate_output_schema    （需 GPU）
  7. generate_diagnosis_report（无需 GPU，整合所有结果）

默认跳过需要 GPU 的步骤（--skip-gpu），只运行无需 GPU 的分析。
加上 --with-gpu 运行完整消融（需确保 GPU 可用）。

用法
----
  # 只运行无需 GPU 的诊断（推荐先跑）
  python scripts/diagnosis/run_all_diagnosis.py

  # 完整诊断（含 GPU 推理消融）
  CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/run_all_diagnosis.py --with-gpu

  # 运行特定步骤
  python scripts/diagnosis/run_all_diagnosis.py --steps buckets noise coverage report

输出
----
  output/diagnosis/
    buckets/
    ontology_ablation/
    label_noise/
    data_coverage/
    kb_ablation/           （仅 --with-gpu）
    schema_ablation/       （仅 --with-gpu）
    final_report/
      diagnosis_report.md  ← 最终统一报告
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
DIAG_OUT = _PROJECT / "output/diagnosis"


def _run_step(script: str, extra_args: List[str] = (), label: str = "") -> bool:
    """运行诊断子脚本，返回是否成功。"""
    script_path = _HERE / script
    cmd = [sys.executable, str(script_path)] + list(extra_args)
    label = label or script
    print(f"\n{'='*60}")
    print(f"  运行：{label}")
    print(f"  命令：{' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "✓ 成功" if result.returncode == 0 else f"✗ 失败（返回码 {result.returncode}）"
    print(f"  {status}，耗时 {elapsed:.1f}s")
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description="一键运行所有诊断脚本")
    p.add_argument(
        "--with-gpu", action="store_true",
        help="同时运行需要 GPU 的消融实验（kb + schema）"
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=["buckets", "ontology", "noise", "coverage", "kb", "schema", "report"],
        help="只运行指定步骤（默认全部）",
    )
    args = p.parse_args()

    all_steps = ["buckets", "ontology", "noise", "coverage"]
    if args.with_gpu:
        all_steps += ["kb", "schema"]
    all_steps.append("report")

    steps_to_run = args.steps if args.steps else all_steps

    results = {}
    t_total = time.time()

    if "buckets" in steps_to_run:
        ok = _run_step(
            "analyze_error_buckets.py",
            ["--output-dir", str(DIAG_OUT / "buckets")],
            "错误分桶分析",
        )
        results["buckets"] = ok

    if "ontology" in steps_to_run:
        ok = _run_step(
            "ablate_ontology_target.py",
            ["--output-dir", str(DIAG_OUT / "ontology_ablation")],
            "Ontology Target 消融（无需 GPU）",
        )
        results["ontology"] = ok

    if "noise" in steps_to_run:
        ok = _run_step(
            "analyze_label_noise.py",
            ["--output-dir", str(DIAG_OUT / "label_noise")],
            "标签噪声分析",
        )
        results["noise"] = ok

    if "coverage" in steps_to_run:
        ok = _run_step(
            "analyze_data_coverage.py",
            ["--output-dir", str(DIAG_OUT / "data_coverage")],
            "数据覆盖分析",
        )
        results["coverage"] = ok

    if "kb" in steps_to_run:
        ok = _run_step(
            "ablate_kb_retrieval.py",
            ["--output-dir", str(DIAG_OUT / "kb_ablation")],
            "KB 检索消融（需 GPU）",
        )
        results["kb"] = ok

    if "schema" in steps_to_run:
        ok = _run_step(
            "ablate_output_schema.py",
            ["--output-dir", str(DIAG_OUT / "schema_ablation")],
            "Schema 消融（需 GPU）",
        )
        results["schema"] = ok

    if "report" in steps_to_run:
        ok = _run_step(
            "generate_diagnosis_report.py",
            [
                "--buckets",  str(DIAG_OUT / "buckets"),
                "--schema",   str(DIAG_OUT / "schema_ablation"),
                "--kb",       str(DIAG_OUT / "kb_ablation"),
                "--ontology", str(DIAG_OUT / "ontology_ablation"),
                "--noise",    str(DIAG_OUT / "label_noise"),
                "--coverage", str(DIAG_OUT / "data_coverage"),
                "--output",   str(DIAG_OUT / "final_report"),
            ],
            "生成统一诊断报告",
        )
        results["report"] = ok

    # ── 汇总 ──────────────────────────────────────
    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  诊断完成，总耗时 {elapsed_total:.0f}s")
    for step, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {step}")

    report_path = DIAG_OUT / "final_report" / "diagnosis_report.md"
    if report_path.exists():
        print(f"\n  最终报告：{report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
