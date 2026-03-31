from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import data_prep_config as cfg

# Phase 2: v2 SFT builder
try:
    from sca.data.sft_builder import build_sft_record_v2, build_distill_record
    from sca.data.marker_features import summarize_positive_markers
    _V2_AVAILABLE = True
except ImportError:
    _V2_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def ensure_output_dirs() -> None:
    Path(cfg.SFT_RECORDS_FULL_JSONL).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.SFT_MESSAGES_JSONL).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.SFT_MESSAGES_NO_THINK_JSONL).parent.mkdir(parents=True, exist_ok=True)


def confidence_from_record(rec: Dict[str, Any]) -> str:
    n_cells = rec.get("n_cells", 0) or 0
    avg_logfc_top5 = rec.get("avg_logfc_top5")

    if avg_logfc_top5 is None:
        return "medium"

    if n_cells >= 100 and avg_logfc_top5 >= 1.5:
        return "high"
    if n_cells >= 50 and avg_logfc_top5 >= 0.8:
        return "medium"
    return "low"

'''
目前有没有明显问题

    有一个点我提醒你，但不算 bug：
        confidence 分布偏高

    你现在是：
        high: 58
        medium: 7
        low: 0

    这说明你当前的 confidence_from_record() 规则偏乐观。
    原因也不难理解：
        你的阈值不算高
        前一步 marker 质量本身也不错
        n_cells 又普遍不小

    这不会影响你“能不能训练”，但会影响数据风格：
        模型会学到“多数情况下都很高置信”
        need_manual_review 基本总是 false

    这不一定理想。

    
建议要不要改
    如果你现在目标是先跑通训练

    不用改。
    这版已经完全可以进入训练阶段。

    如果你想让数据更像真实标注场景

    建议轻微收紧一下 confidence 规则，让一部分样本变成 medium 或 low: V2 版本的 confidence_from_record() 已经更新了规则，整体会更严格一些，预计会有更多 medium 和 low 的样本。你可以先试试看新的分布情况，再决定是否需要进一步调整阈值。
'''
def confidence_from_record_V2(rec: Dict[str, Any]) -> str:
    n_cells = rec.get("n_cells", 0) or 0
    avg_logfc_top5 = rec.get("avg_logfc_top5")

    if avg_logfc_top5 is None:
        return "medium"

    if n_cells >= 500 and avg_logfc_top5 >= 2.0:
        return "high"
    if n_cells >= 100 and avg_logfc_top5 >= 1.0:
        return "medium"
    return "low"


def build_user_prompt(rec: Dict[str, Any], add_no_think_suffix: bool = False) -> str:
    markers = ", ".join(rec.get("marker_genes", []))
    disease = rec.get("disease", "unknown")
    tissue = rec.get("tissue_general", "unknown")
    organism = rec.get("organism", "unknown")

    prompt = (
        "You are given a ranked marker-gene list from a single-cell RNA-seq cluster.\n\n"
        f"Organism: {organism}\n"
        f"Tissue: {tissue}\n"
        f"Disease/Context: {disease}\n"
        f"Top marker genes: {markers}\n\n"
        "Task:\n"
        "1. Predict the most likely cell type.\n"
        "2. List 2-4 supporting markers from the provided genes.\n"
        "3. Estimate confidence as one of: high, medium, low.\n"
        "4. Decide whether manual review is needed.\n"
        "5. Give a short rationale.\n\n"
        "Return valid JSON with keys: "
        "cell_type, supporting_markers, confidence, need_manual_review, rationale."
    )

    if add_no_think_suffix:
        prompt += " /no_think"

    return prompt


def build_assistant_answer(rec: Dict[str, Any], with_empty_think: bool = False) -> str:
    # 优先使用 clean 标签作为训练目标
    label_clean = rec.get("cell_type_clean", "unknown")
    label_gold_major = rec.get("cell_type_gold_major", label_clean)

    supporting_markers = rec.get("marker_genes", [])[:4]
    confidence = confidence_from_record(rec)
    need_manual_review = confidence == "low"

    rationale = (
        f"The ranked marker list is most consistent with {label_clean}. "
        f"Representative supporting markers include {', '.join(supporting_markers)}. "
        f"The original source label is {label_gold_major}."
    )

    answer_dict = {
        "cell_type": label_clean,
        "supporting_markers": supporting_markers,
        "confidence": confidence,
        "need_manual_review": need_manual_review,
        "rationale": rationale,
    }

    answer_text = json.dumps(answer_dict, ensure_ascii=False)

    if with_empty_think:
        return "<think>\n\n</think>\n\n" + answer_text
    return answer_text


def load_marker_records() -> List[Dict[str, Any]]:
    path = Path(cfg.MARKER_EXAMPLES_JSONL)
    if not path.exists():
        raise FileNotFoundError(f"Marker examples file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                logging.warning("Skip invalid JSON line %d: %s", line_no, e)
                continue

            if "marker_genes" not in rec or not rec["marker_genes"]:
                logging.warning("Skip line %d: missing marker_genes", line_no)
                continue

            if "cell_type_clean" not in rec:
                logging.warning("Skip line %d: missing cell_type_clean", line_no)
                continue

            records.append(rec)

    return records


def load_marker_records_v2() -> List[Dict[str, Any]]:
    """Load v2 marker records from marker_examples_v2.jsonl."""
    path = Path(cfg.MARKER_EXAMPLES_V2_JSONL)
    if not path.exists():
        logging.warning("marker_examples_v2.jsonl not found: %s — v2 SFT will be skipped", path)
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                logging.warning("Skip invalid JSON line %d: %s", line_no, e)
                continue

            if not rec.get("positive_markers"):
                logging.debug("Skip line %d: missing positive_markers", line_no)
                continue

            if "cell_type_clean" not in rec:
                logging.warning("Skip line %d: missing cell_type_clean", line_no)
                continue

            records.append(rec)

    logging.info("Loaded %d v2 marker records from %s", len(records), path.name)
    return records


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_summary(
    full_records: List[Dict[str, Any]],
    pure_messages: List[Dict[str, Any]],
    pure_messages_no_think: List[Dict[str, Any]],
) -> None:
    summary_path = Path(cfg.SFT_MESSAGES_JSONL).with_name("05_make_sft_jsonl_run_summary.txt")

    conf_counter = Counter()
    label_counter = Counter()
    dataset_counter = Counter()

    for rec in full_records:
        conf_counter[rec["confidence"]] += 1
        label_counter[rec["cell_type_clean"]] += 1
        dataset_counter[rec["dataset_id"]] += 1

    lines = [
        "=== 05_make_sft_jsonl run summary ===",
        "",
        f"Full records: {len(full_records)}",
        f"Standard messages: {len(pure_messages)}",
        f"No-think messages: {len(pure_messages_no_think)}",
        "",
        "Confidence distribution:",
    ]

    for k in ["high", "medium", "low"]:
        lines.append(f"- {k}: {conf_counter.get(k, 0)}")

    lines.append("")
    lines.append("Examples per dataset:")
    for ds, cnt in sorted(dataset_counter.items()):
        lines.append(f"- {ds}: {cnt}")

    lines.append("")
    lines.append(f"SFT_RECORDS_FULL_JSONL: {cfg.SFT_RECORDS_FULL_JSONL}")
    lines.append(f"SFT_MESSAGES_JSONL: {cfg.SFT_MESSAGES_JSONL}")
    lines.append(f"SFT_MESSAGES_NO_THINK_JSONL: {cfg.SFT_MESSAGES_NO_THINK_JSONL}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved run summary to %s", summary_path)


def save_summary_v2(
    full_records: List[Dict[str, Any]],
    distill_records: List[Dict[str, Any]],
) -> None:
    summary_path = Path(cfg.SFT_MESSAGES_V2_JSONL).with_name("05_make_sft_jsonl_v2_run_summary.txt")

    conf_counter: Counter = Counter()
    label_counter: Counter = Counter()
    decision_counter: Counter = Counter()
    dataset_counter: Counter = Counter()
    evidence_counter: Counter = Counter()

    for rec in full_records:
        conf_counter[rec.get("confidence_label", "unknown")] += 1
        label_counter[rec.get("cell_type_clean", "unknown")] += 1
        decision_counter[rec.get("decision", "unknown")] += 1
        dataset_counter[rec.get("dataset_id", "unknown")] += 1
        evidence_counter[rec.get("evidence_support_level", "unknown")] += 1

    n_ontology_mapped = sum(1 for r in full_records if r.get("cell_ontology_id"))
    n_novelty = sum(1 for r in full_records if r.get("novelty_flag"))

    lines = [
        "=== 05_make_sft_jsonl v2 run summary ===",
        "",
        f"Full records (v2): {len(full_records)}",
        f"Distill records: {len(distill_records)}",
        f"Ontology-mapped: {n_ontology_mapped}",
        f"Novelty-flagged: {n_novelty}",
        "",
        "Confidence distribution:",
    ]
    for k in ["high", "medium", "low"]:
        lines.append(f"  {k}: {conf_counter.get(k, 0)}")

    lines += ["", "Decision distribution:"]
    for k in ["accept", "review", "unresolved", "novel_candidate"]:
        lines.append(f"  {k}: {decision_counter.get(k, 0)}")

    lines += ["", "Evidence support distribution:"]
    for k in ["strong", "moderate", "weak", "conflicting"]:
        lines.append(f"  {k}: {evidence_counter.get(k, 0)}")

    lines += ["", "Examples per dataset:"]
    for ds, cnt in sorted(dataset_counter.items()):
        lines.append(f"  {ds}: {cnt}")

    lines += [
        "",
        f"SFT_RECORDS_FULL_V2: {cfg.SFT_RECORDS_FULL_V2_JSONL}",
        f"SFT_MESSAGES_V2: {cfg.SFT_MESSAGES_V2_JSONL}",
        f"SFT_MESSAGES_NO_THINK_V2: {cfg.SFT_MESSAGES_NO_THINK_V2_JSONL}",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved v2 run summary to %s", summary_path)


def main() -> None:
    ensure_output_dirs()

    marker_records = load_marker_records()
    if not marker_records:
        raise ValueError("No valid marker records loaded from marker_examples.jsonl")

    full_records = []
    pure_messages = []
    pure_messages_no_think = []

    for rec in marker_records:
        confidence = confidence_from_record(rec)

        msg_std = {
            "messages": [
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(rec, add_no_think_suffix=False)},
                {"role": "assistant", "content": build_assistant_answer(rec, with_empty_think=False)},
            ]
        }

        msg_no_think = {
            "messages": [
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(rec, add_no_think_suffix=True)},
                {"role": "assistant", "content": build_assistant_answer(rec, with_empty_think=True)},
            ]
        }

        full = {
            "dataset_id": rec.get("dataset_id", "unknown_dataset"),
            "dataset_title": rec.get("dataset_title", "unknown_title"),
            "organism": rec.get("organism", "unknown"),
            "tissue_general": rec.get("tissue_general", "unknown"),
            "tissue": rec.get("tissue", "unknown"),
            "disease": rec.get("disease", "unknown"),
            "cell_type_clean": rec.get("cell_type_clean", "unknown"),
            "cell_type_gold_major": rec.get("cell_type_gold_major", rec.get("cell_type_clean", "unknown")),
            "n_cells": rec.get("n_cells", 0),
            "marker_genes": rec.get("marker_genes", []),
            "confidence": confidence,
            "messages": msg_std["messages"],
            "messages_no_think": msg_no_think["messages"],
        }

        full_records.append(full)
        pure_messages.append(msg_std)
        pure_messages_no_think.append(msg_no_think)

    write_jsonl(cfg.SFT_RECORDS_FULL_JSONL, full_records)
    write_jsonl(cfg.SFT_MESSAGES_JSONL, pure_messages)
    write_jsonl(cfg.SFT_MESSAGES_NO_THINK_JSONL, pure_messages_no_think)

    # 额外保存一个 csv manifest，方便快速检查
    manifest_path = Path(cfg.SFT_MESSAGES_JSONL).with_name("sft_records_manifest.csv")
    pd.DataFrame([
        {
            "dataset_id": x["dataset_id"],
            "dataset_title": x["dataset_title"],
            "cell_type_clean": x["cell_type_clean"],
            "cell_type_gold_major": x["cell_type_gold_major"],
            "n_cells": x["n_cells"],
            "confidence": x["confidence"],
            "tissue_general": x["tissue_general"],
            "disease": x["disease"],
        }
        for x in full_records
    ]).to_csv(manifest_path, index=False)

    logging.info("Saved full records to %s", cfg.SFT_RECORDS_FULL_JSONL)
    logging.info("Saved standard messages to %s", cfg.SFT_MESSAGES_JSONL)
    logging.info("Saved no-think messages to %s", cfg.SFT_MESSAGES_NO_THINK_JSONL)
    logging.info("Saved manifest to %s", manifest_path)

    save_summary(full_records, pure_messages, pure_messages_no_think)

    # =====================================================================
    # Phase 2: v2 SFT generation from marker_examples_v2.jsonl
    # =====================================================================
    if not _V2_AVAILABLE:
        logging.warning("sca.data.sft_builder not available — skipping v2 SFT generation")
        return

    v2_marker_records = load_marker_records_v2()
    if not v2_marker_records:
        logging.warning("No v2 marker records found — skipping v2 SFT generation")
        return

    logging.info("Building v2 SFT from %d v2 marker records", len(v2_marker_records))

    v2_full_records: List[Dict[str, Any]] = []
    v2_pure_messages: List[Dict[str, Any]] = []
    v2_pure_messages_no_think: List[Dict[str, Any]] = []
    v2_distill_records: List[Dict[str, Any]] = []

    for rec in v2_marker_records:
        try:
            sft_rec = build_sft_record_v2(rec, system_prompt=cfg.SYSTEM_PROMPT)
        except Exception as e:
            logging.warning("Failed building v2 SFT for %s/%s: %r",
                            rec.get("dataset_id"), rec.get("cell_type_clean"), e)
            continue

        v2_full_records.append(sft_rec)
        v2_pure_messages.append({"messages": sft_rec["messages"]})
        v2_pure_messages_no_think.append({"messages": sft_rec["messages_no_think"]})

        try:
            distill = build_distill_record(rec, system_prompt=cfg.SYSTEM_PROMPT)
            v2_distill_records.append(distill)
        except Exception as e:
            logging.debug("Failed building distill record: %r", e)

    write_jsonl(cfg.SFT_RECORDS_FULL_V2_JSONL, v2_full_records)
    write_jsonl(cfg.SFT_MESSAGES_V2_JSONL, v2_pure_messages)
    write_jsonl(cfg.SFT_MESSAGES_NO_THINK_V2_JSONL, v2_pure_messages_no_think)

    # distillation records
    distill_path = Path(cfg.SFT_DIR) / "distill_records_v1.jsonl"
    write_jsonl(distill_path, v2_distill_records)

    # manifest v2
    manifest_v2_path = Path(cfg.SFT_DIR) / "sft_records_manifest_v2.csv"
    pd.DataFrame([
        {
            "dataset_id": x["dataset_id"],
            "dataset_title": x["dataset_title"],
            "cell_type_clean": x["cell_type_clean"],
            "cell_ontology_id": x.get("cell_ontology_id"),
            "n_cells": x["n_cells"],
            "confidence_label": x.get("confidence_label"),
            "confidence_score": x.get("confidence_score"),
            "evidence_support_level": x.get("evidence_support_level"),
            "decision": x.get("decision"),
            "novelty_flag": x.get("novelty_flag"),
            "marker_quality_score": x.get("marker_quality_score"),
            "tissue_general": x["tissue_general"],
            "disease": x["disease"],
        }
        for x in v2_full_records
    ]).to_csv(manifest_v2_path, index=False)

    logging.info("Saved v2 full records: %s (%d rows)", cfg.SFT_RECORDS_FULL_V2_JSONL, len(v2_full_records))
    logging.info("Saved v2 messages: %s", cfg.SFT_MESSAGES_V2_JSONL)
    logging.info("Saved v2 no-think: %s", cfg.SFT_MESSAGES_NO_THINK_V2_JSONL)
    logging.info("Saved distill records: %s (%d rows)", distill_path, len(v2_distill_records))
    logging.info("Saved v2 manifest: %s", manifest_v2_path)

    save_summary_v2(v2_full_records, v2_distill_records)

    print(f"\nv2 SFT generation complete:")
    print(f"  full_records_v2   : {len(v2_full_records)}")
    print(f"  distill_records   : {len(v2_distill_records)}")
    print(f"  manifest_v2       : {manifest_v2_path}")


if __name__ == "__main__":
    main()


# python -u scripts/data_prep/05_make_sft_jsonl.py 2>&1 | tee data/meta/05_make_sft_jsonl.log

# nohup bash -lc 'export PYTHONPATH="$PWD/src:$PYTHONPATH"; python -u scripts/data_prep/05_make_sft_jsonl.py >> data/meta/05_make_sft_jsonl.log 2>&1' >/dev/null 2>&1 &
# grep -E "Generated|Building markers|Make markers:" data/meta/05_make_sft_jsonl.log | tail -10