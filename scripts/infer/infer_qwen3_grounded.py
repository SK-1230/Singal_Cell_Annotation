#!/usr/bin/env python3
"""
infer_qwen3_grounded.py — SCA-Specialist grounded inference pipeline (Phase 3).

Pipeline:
    cluster evidence
      ↓
    retrieval from local KB
      ↓
    grounded prompt assembly
      ↓
    model inference
      ↓
    JSON parse
      ↓
    ontology validation
      ↓
    confidence calibration (if calibrator available)
      ↓
    final decision
      ↓
    save structured result + markdown report

Usage (defaults):
    python scripts/infer/infer_qwen3_grounded.py

Override examples:
    MODEL_PATH=/path/to/base ADAPTER_PATH=/path/to/adapter \\
        python scripts/infer/infer_qwen3_grounded.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[1]
_SRC_DIR = _PROJECT_DIR / "src"
_SCRIPTS_DATA_PREP = _PROJECT_DIR / "scripts" / "data_prep"
for _p in [str(_SRC_DIR), str(_SCRIPTS_DATA_PREP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_prep_config as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable paths / parameters (all overridable via env vars)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(_PROJECT_DIR / "my_models" / "Qwen" / "Qwen3-4B"),
)
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")  # empty → use base model only
TEST_FILE = os.environ.get(
    "TEST_FILE",
    str(cfg.SPLIT_DIR / "test_messages_no_think_v2.jsonl"),
)
MARKER_EXAMPLES_FILE = os.environ.get(
    "MARKER_EXAMPLES_FILE",
    str(cfg.MARKER_EXAMPLES_V2_JSONL),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    str(_PROJECT_DIR / "output" / "infer_qwen3_grounded"),
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))  # 0 = all
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", "5"))

ACCEPT_THRESHOLD = float(
    os.environ.get("FINAL_ACCEPT_THRESHOLD", str(getattr(cfg, "FINAL_ACCEPT_THRESHOLD", 0.80)))
)
REVIEW_THRESHOLD = float(
    os.environ.get("FINAL_REVIEW_THRESHOLD", str(getattr(cfg, "FINAL_REVIEW_THRESHOLD", 0.45)))
)
NOVELTY_THRESHOLD = float(
    os.environ.get("NOVELTY_THRESHOLD", str(getattr(cfg, "NOVELTY_THRESHOLD", 0.25)))
)
CALIBRATOR_PATH = os.environ.get(
    "CALIBRATOR_PATH",
    str(_PROJECT_DIR / "output" / "calibration" / "confidence_calibrator.joblib"),
)


# ---------------------------------------------------------------------------
# Lazy imports (only needed if actually running)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("Loading tokenizer from %s", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    logger.info("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    if ADAPTER_PATH and Path(ADAPTER_PATH).exists():
        logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model, tokenizer


def _load_kb():
    from sca.knowledge.marker_kb import MarkerKB
    kb = MarkerKB()
    kb_path = cfg.MERGED_MARKER_KB_JSONL
    if not kb_path.exists():
        kb_path = cfg.TRAIN_MARKER_KB_JSONL
    if kb_path.exists():
        kb.load(str(kb_path))
        logger.info("Loaded KB from %s (%d entries)", kb_path, kb.size())
    else:
        logger.warning("No KB file found — retrieval will return empty results.")
    return kb


def _load_calibrator():
    cal_path = Path(CALIBRATOR_PATH)
    if not cal_path.exists():
        logger.info("No calibrator found at %s — using raw confidence scores.", cal_path)
        return None
    try:
        import joblib
        cal = joblib.load(cal_path)
        logger.info("Loaded calibrator from %s", cal_path)
        return cal
    except Exception as exc:
        logger.warning("Could not load calibrator: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

def _extract_query_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a query dict from a marker_examples_v2 record."""
    return {
        "organism": rec.get("organism", "Homo sapiens"),
        "tissue_general": rec.get("tissue_general", "unknown"),
        "disease": rec.get("disease", "unknown"),
        "n_cells": rec.get("n_cells", 0),
        "positive_markers": rec.get("positive_markers", []),
        "negative_markers": rec.get("negative_markers", []),
        "cell_type_clean": rec.get("cell_type_clean"),
        "cell_ontology_id": rec.get("cell_ontology_id"),
        "dataset_id": rec.get("dataset_id"),
        "cluster_id": rec.get("cluster_id"),
    }


def _extract_query_from_messages(msg_rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempt to reconstruct a minimal query dict from a messages record.
    Falls back to returning the messages as-is with empty retrieval.
    """
    # We only need the user message text; proper query extraction would
    # require the original marker_examples_v2 record, so this is a thin wrapper.
    messages = msg_rec.get("messages", [])
    user_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break
    # Parse basic fields from the prompt text (best-effort)
    query: Dict[str, Any] = {
        "organism": "Homo sapiens",
        "tissue_general": "unknown",
        "disease": "unknown",
        "n_cells": 0,
        "positive_markers": [],
        "negative_markers": [],
    }
    for line in user_text.splitlines():
        if line.startswith("Organism:"):
            query["organism"] = line.split(":", 1)[1].strip()
        elif line.startswith("Tissue:"):
            query["tissue_general"] = line.split(":", 1)[1].strip()
        elif line.startswith("Disease/Context:"):
            query["disease"] = line.split(":", 1)[1].strip()
        elif line.startswith("Cluster size:"):
            try:
                query["n_cells"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return query


def _run_retrieval(query: Dict[str, Any], kb) -> List[Dict[str, Any]]:
    from sca.knowledge.retrieval import retrieve_candidate_cell_types

    gene_names: List[str] = []
    for m in query.get("positive_markers", []):
        if isinstance(m, dict):
            g = m.get("gene") or m.get("gene_name") or ""
        else:
            g = str(m)
        if g:
            gene_names.append(g)

    if not gene_names:
        return []

    return retrieve_candidate_cell_types(
        query_markers=gene_names,
        tissue=query.get("tissue_general"),
        top_k=TOP_K_RETRIEVAL,
        kb=kb,
    )


def _build_prompt(query: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> str:
    from sca.model.prompt_builder import build_grounded_infer_prompt_v2

    return build_grounded_infer_prompt_v2(
        query=query,
        retrieved_evidence=retrieved,
        add_no_think_suffix=True,
    )


def _generate(prompt: str, system_prompt: str, model, tokenizer) -> str:
    import torch
    from sca.model.prompt_builder import SYSTEM_PROMPT_V2

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated


def _parse_output(raw_text: str) -> Dict[str, Any]:
    from sca.model.output_parser import parse_model_output_v2

    parsed, ok, err = parse_model_output_v2(raw_text)
    if not ok:
        logger.debug("Output parse issue: %s", err)
    return parsed or {}, ok, err


def _apply_calibrator(
    merged: Dict[str, Any],
    calibrator,
) -> float:
    """Apply the sklearn calibrator if available, else return final_confidence_score."""
    if calibrator is None:
        return merged.get("final_confidence_score", merged.get("confidence_score", 0.5))
    try:
        from sca.model.calibration import build_calibration_features

        features = build_calibration_features(merged)
        import numpy as np
        X = np.array([features])
        prob = calibrator.predict_proba(X)[0][1]
        return round(float(prob), 4)
    except Exception as exc:
        logger.debug("Calibration failed: %s", exc)
        return merged.get("final_confidence_score", merged.get("confidence_score", 0.5))


def _merge_and_decide(
    parsed_output: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    calibrator,
) -> Dict[str, Any]:
    from sca.model.decision_logic import (
        merge_model_output_and_retrieval_evidence,
        decide_accept_review_or_unresolved,
    )

    merged = merge_model_output_and_retrieval_evidence(
        model_output=parsed_output,
        retrieved_candidates=retrieved,
        accept_threshold=ACCEPT_THRESHOLD,
        review_threshold=REVIEW_THRESHOLD,
        novelty_threshold=NOVELTY_THRESHOLD,
    )

    # Re-calibrate if calibrator available
    calibrated_score = _apply_calibrator(merged, calibrator)
    merged["final_confidence_score"] = calibrated_score

    # Re-decide with calibrated score
    merged["final_decision"] = decide_accept_review_or_unresolved(
        final_confidence_score=calibrated_score,
        evidence_support_level=merged.get("evidence_support_level", "weak"),
        novelty_flag=merged.get("novelty_flag", False),
        accept_threshold=ACCEPT_THRESHOLD,
        review_threshold=REVIEW_THRESHOLD,
    )
    return merged


def _render_markdown_report(
    idx: int,
    query: Dict[str, Any],
    merged: Dict[str, Any],
    raw_output: str,
    parse_ok: bool,
) -> str:
    cell_type = merged.get("cell_type", "N/A")
    cl_id = merged.get("cell_ontology_id", "N/A")
    final_decision = merged.get("final_decision", "N/A")
    final_score = merged.get("final_confidence_score", "N/A")
    retrieval_score = merged.get("retrieval_support_score", "N/A")
    ontology_status = merged.get("ontology_validation_status", "N/A")
    rationale = merged.get("rationale", "")

    tissue = query.get("tissue_general", "N/A")
    disease = query.get("disease", "N/A")
    organism = query.get("organism", "N/A")
    n_cells = query.get("n_cells", "N/A")
    gold = query.get("cell_type_clean", "N/A")

    retrieved_list = merged.get("retrieved_candidates", [])
    retrieved_lines = []
    for c in retrieved_list[:5]:
        retrieved_lines.append(
            f"  - {c.get('label', '?')} | overlap={c.get('overlap_score', 0):.3f}"
        )

    lines = [
        f"# Sample {idx:04d} — Grounded Annotation Report",
        "",
        "## Context",
        f"- Organism: {organism}",
        f"- Tissue: {tissue}",
        f"- Disease: {disease}",
        f"- Cluster size: {n_cells}",
        f"- Gold label: {gold}",
        "",
        "## Prediction",
        f"- Cell type: **{cell_type}**",
        f"- CL ID: `{cl_id}`",
        f"- Final decision: `{final_decision}`",
        f"- Final confidence: {final_score}",
        "",
        "## Retrieval",
        f"- Retrieval support score: {retrieval_score}",
        f"- Ontology validation: `{ontology_status}`",
    ]
    if retrieved_lines:
        lines.append("- Top candidates:")
        lines.extend(retrieved_lines)

    lines += [
        "",
        "## Rationale",
        rationale or "_No rationale provided._",
        "",
        "## Parse Status",
        f"- Parse OK: {parse_ok}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import datetime

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "per_sample_reports"
    reports_dir.mkdir(exist_ok=True)

    logger.info("=== infer_qwen3_grounded ===")
    logger.info("TEST_FILE   : %s", TEST_FILE)
    logger.info("OUTPUT_DIR  : %s", OUTPUT_DIR)
    logger.info("TOP_K_RETRIEVAL: %d", TOP_K_RETRIEVAL)
    logger.info("ACCEPT_THRESHOLD: %.2f", ACCEPT_THRESHOLD)
    logger.info("REVIEW_THRESHOLD: %.2f", REVIEW_THRESHOLD)

    # Try to load marker_examples_v2 as the primary query source
    use_marker_examples = Path(MARKER_EXAMPLES_FILE).exists()
    if use_marker_examples:
        logger.info("Using marker_examples_v2 as query source: %s", MARKER_EXAMPLES_FILE)
        with open(MARKER_EXAMPLES_FILE, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        queries = [_extract_query_from_record(r) for r in records]
    else:
        logger.info("Falling back to test messages: %s", TEST_FILE)
        if not Path(TEST_FILE).exists():
            logger.error("Neither marker_examples nor test file found. Exiting.")
            sys.exit(1)
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            msg_records = [json.loads(line) for line in f if line.strip()]
        queries = [_extract_query_from_messages(r) for r in msg_records]
        records = msg_records  # used below for gold labels if present

    if MAX_SAMPLES > 0:
        queries = queries[:MAX_SAMPLES]
        records = records[:MAX_SAMPLES]

    logger.info("Samples to process: %d", len(queries))

    kb = _load_kb()
    calibrator = _load_calibrator()
    model, tokenizer = _load_model_and_tokenizer()

    from sca.model.prompt_builder import SYSTEM_PROMPT_V2

    all_results: List[Dict[str, Any]] = []
    n_parse_ok = 0
    n_accept = n_review = n_unresolved = n_novel = 0
    start_time = datetime.datetime.now()

    for idx, (query, rec) in enumerate(zip(queries, records)):
        logger.info("[%d/%d] processing ...", idx + 1, len(queries))

        retrieved = _run_retrieval(query, kb)
        prompt = _build_prompt(query, retrieved)
        raw_output = _generate(prompt, SYSTEM_PROMPT_V2, model, tokenizer)
        parsed, parse_ok, parse_err = _parse_output(raw_output)

        if parse_ok:
            n_parse_ok += 1

        merged = _merge_and_decide(parsed, retrieved, calibrator)
        merged["_raw_output"] = raw_output
        merged["_parse_ok"] = parse_ok
        merged["_parse_err"] = parse_err
        merged["_sample_idx"] = idx
        merged["_dataset_id"] = query.get("dataset_id", rec.get("dataset_id", ""))
        merged["_cluster_id"] = query.get("cluster_id", rec.get("cluster_id", ""))
        merged["_gold_cell_type"] = query.get("cell_type_clean", rec.get("cell_type_clean", ""))

        fd = merged.get("final_decision", "unresolved")
        if fd == "accept":
            n_accept += 1
        elif fd == "review":
            n_review += 1
        elif fd == "novel_candidate":
            n_novel += 1
        else:
            n_unresolved += 1

        # Per-sample markdown report
        report_md = _render_markdown_report(idx, query, merged, raw_output, parse_ok)
        report_path = reports_dir / f"sample_{idx:04d}.md"
        report_path.write_text(report_md, encoding="utf-8")

        all_results.append(merged)

    elapsed = (datetime.datetime.now() - start_time).total_seconds()

    # Save predictions.jsonl (exclude raw output to keep file clean)
    preds_jsonl = out_dir / "predictions.jsonl"
    with open(preds_jsonl, "w", encoding="utf-8") as f:
        for r in all_results:
            row = {k: v for k, v in r.items() if k != "_raw_output"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved %s", preds_jsonl)

    # Save predictions.csv (flat key subset)
    try:
        import csv
        csv_path = out_dir / "predictions.csv"
        fieldnames = [
            "_sample_idx", "_dataset_id", "_cluster_id", "_gold_cell_type",
            "cell_type", "cell_ontology_id", "parent_cell_type",
            "confidence_label", "confidence_score",
            "final_confidence_score", "final_decision",
            "evidence_support_level", "ontology_validation_status",
            "retrieval_support_score", "novelty_flag",
            "need_manual_review", "_parse_ok",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Saved %s", csv_path)
    except Exception as exc:
        logger.warning("Could not write CSV: %s", exc)

    # Save summary.json
    n_total = len(all_results)
    summary = {
        "n_total": n_total,
        "n_parse_ok": n_parse_ok,
        "parse_success_rate": round(n_parse_ok / n_total, 4) if n_total else 0.0,
        "decision_distribution": {
            "accept": n_accept,
            "review": n_review,
            "unresolved": n_unresolved,
            "novel_candidate": n_novel,
        },
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "model_path": MODEL_PATH,
            "adapter_path": ADAPTER_PATH,
            "test_file": TEST_FILE,
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "accept_threshold": ACCEPT_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD,
            "novelty_threshold": NOVELTY_THRESHOLD,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
