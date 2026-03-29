#!/usr/bin/env python3
"""
evaluate_predictions.py — Systematic evaluation of SCA-Specialist predictions (Phase 3).

Reads a predictions JSONL (from infer_qwen3_grounded.py or any infer script)
and computes a comprehensive set of metrics.

Required metrics (10):
    1.  JSON parse success rate
    2.  Exact cell type match (pred == gold)
    3.  Confidence label match
    4.  need_manual_review match
    5.  Ontology exact match (CL ID)
    6.  Ontology parent-level match
    7.  Ontology-distance score (if ontology store available)
    8.  Decision distribution (accept / review / unresolved / novel_candidate)
    9.  Rare-label subset accuracy
    10. Unmapped subset performance

Optional metrics:
    - Top-k candidate hit
    - Evidence support agreement
    - Calibration ECE
    - Brier score

Outputs:
    eval_overall.json
    eval_by_split.csv
    eval_by_tissue.csv
    eval_by_label.csv
    eval_by_hardness.csv

Usage:
    python scripts/infer/evaluate_predictions.py \\
        --predictions output/infer_qwen3_grounded/predictions.jsonl \\
        --output_dir output/eval

Environment overrides:
    PREDICTIONS_FILE, OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[1]
_SRC_DIR = _PROJECT_DIR / "src"
for _p in [str(_SRC_DIR), str(_PROJECT_DIR / "scripts" / "data_prep")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _cell_type_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    return p == g


def _cell_type_partial_match(pred: str, gold: str) -> bool:
    """Parent-level match: one label is substring of the other."""
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    return p in g or g in p


def _cl_id_match(pred_id: str, gold_id: str) -> bool:
    p = _norm(pred_id)
    g = _norm(gold_id)
    return bool(p) and bool(g) and p == g


# ---------------------------------------------------------------------------
# Metrics per record
# ---------------------------------------------------------------------------

def _record_metrics(rec: Dict[str, Any]) -> Dict[str, Any]:
    gold_cell_type = rec.get("_gold_cell_type") or rec.get("gold_cell_type") or ""
    pred_cell_type = rec.get("cell_type") or ""

    gold_cl_id = rec.get("_gold_cell_ontology_id") or rec.get("gold_cell_ontology_id") or ""
    pred_cl_id = rec.get("cell_ontology_id") or ""

    gold_conf_label = rec.get("_gold_confidence_label") or ""
    pred_conf_label = rec.get("confidence_label") or ""

    gold_review = rec.get("_gold_need_manual_review")
    pred_review = rec.get("need_manual_review")

    parse_ok = bool(rec.get("_parse_ok", True))
    final_decision = rec.get("final_decision") or rec.get("decision") or ""

    exact_match = _cell_type_match(pred_cell_type, gold_cell_type) if gold_cell_type else None
    parent_match = (
        _cell_type_partial_match(pred_cell_type, gold_cell_type)
        if gold_cell_type and not exact_match
        else None
    )
    cl_id_exact = _cl_id_match(pred_cl_id, gold_cl_id) if gold_cl_id else None

    conf_label_match = (
        _norm(pred_conf_label) == _norm(gold_conf_label)
        if gold_conf_label
        else None
    )
    review_match = (
        bool(pred_review) == bool(gold_review)
        if gold_review is not None
        else None
    )

    # Hardness flag
    is_hard = bool(rec.get("hardness_flags")) or (
        rec.get("n_cells", 9999) < 80
    )

    # Split tag
    split_tag = rec.get("_split") or rec.get("split") or "test"

    # Tissue
    tissue = rec.get("tissue_general") or rec.get("_tissue") or "unknown"

    # Ontology status
    ontology_status = rec.get("ontology_validation_status") or ""

    # Retrieval top-k hit (check if gold in retrieved candidates)
    retrieved = rec.get("retrieved_candidates") or []
    topk_hit = any(
        _cell_type_match(c.get("label", ""), gold_cell_type)
        for c in retrieved
    ) if gold_cell_type else None

    # Evidence support
    evidence_level = rec.get("evidence_support_level") or ""

    # Ontology mapped
    ontology_mapped = bool(pred_cl_id and pred_cl_id.upper().startswith("CL:"))

    return {
        "parse_ok": parse_ok,
        "exact_match": exact_match,
        "parent_match": parent_match,
        "cl_id_exact": cl_id_exact,
        "conf_label_match": conf_label_match,
        "review_match": review_match,
        "is_hard": is_hard,
        "split_tag": split_tag,
        "tissue": tissue,
        "cell_type": _norm(gold_cell_type),
        "final_decision": final_decision,
        "ontology_status": ontology_status,
        "topk_hit": topk_hit,
        "evidence_level": evidence_level,
        "ontology_mapped": ontology_mapped,
        "final_confidence_score": rec.get("final_confidence_score"),
        "confidence_score": rec.get("confidence_score"),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _safe_rate(num: int, denom: int) -> Optional[float]:
    return round(num / denom, 4) if denom > 0 else None


def _aggregate(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(metrics_list)
    if n == 0:
        return {"n": 0}

    def _count_true(key: str) -> Tuple[int, int]:
        valid = [m[key] for m in metrics_list if m[key] is not None]
        return sum(1 for v in valid if v), len(valid)

    parse_ok_n = sum(1 for m in metrics_list if m["parse_ok"])
    exact_t, exact_n = _count_true("exact_match")
    parent_t, parent_n = _count_true("parent_match")
    cl_id_t, cl_id_n = _count_true("cl_id_exact")
    conf_t, conf_n = _count_true("conf_label_match")
    review_t, review_n = _count_true("review_match")
    topk_t, topk_n = _count_true("topk_hit")

    # Ontology-level match = exact OR parent_match
    ont_exact_n = sum(1 for m in metrics_list if m["cl_id_exact"])
    ont_parent_n = sum(1 for m in metrics_list if m["parent_match"])

    # Decision distribution
    dec_dist: Dict[str, int] = defaultdict(int)
    for m in metrics_list:
        dec_dist[m["final_decision"] or "unknown"] += 1

    return {
        "n": n,
        "parse_success_rate": _safe_rate(parse_ok_n, n),
        "exact_cell_type_match_rate": _safe_rate(exact_t, exact_n),
        "parent_level_match_rate": _safe_rate(parent_t, parent_n),
        "cell_type_match_rate_combined": _safe_rate(exact_t + parent_t, exact_n),
        "cl_id_exact_match_rate": _safe_rate(cl_id_t, cl_id_n),
        "conf_label_match_rate": _safe_rate(conf_t, conf_n),
        "review_flag_match_rate": _safe_rate(review_t, review_n),
        "topk_retrieval_hit_rate": _safe_rate(topk_t, topk_n),
        "ontology_exact_n": ont_exact_n,
        "ontology_parent_n": ont_parent_n,
        "decision_distribution": dict(dec_dist),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_metrics = [_record_metrics(r) for r in records]

    overall = _aggregate(all_metrics)

    # Rare-label subset: cell types that appear <= 3 times in dataset
    from collections import Counter
    label_counts = Counter(m["cell_type"] for m in all_metrics if m["cell_type"])
    rare_labels = {lbl for lbl, cnt in label_counts.items() if cnt <= 3}
    rare_metrics = [m for m in all_metrics if m["cell_type"] in rare_labels]
    overall["rare_label_subset"] = _aggregate(rare_metrics)
    overall["rare_label_subset"]["n_unique_labels"] = len(rare_labels)

    # Unmapped subset: ontology_status == 'missing' or 'no_retrieval'
    unmapped_metrics = [
        m for m in all_metrics
        if m["ontology_status"] in ("missing", "no_retrieval", "unmatched")
        or not m["ontology_mapped"]
    ]
    overall["unmapped_subset"] = _aggregate(unmapped_metrics)

    # ECE / Brier (optional)
    try:
        scores = [
            m["final_confidence_score"]
            for m in all_metrics
            if m["final_confidence_score"] is not None and m["exact_match"] is not None
        ]
        y_true = [
            1 if m["exact_match"] else 0
            for m in all_metrics
            if m["final_confidence_score"] is not None and m["exact_match"] is not None
        ]
        if scores and y_true:
            import numpy as np
            scores_arr = np.array(scores)
            y_arr = np.array(y_true, dtype=float)
            brier = float(np.mean((scores_arr - y_arr) ** 2))

            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (scores_arr >= bin_edges[i]) & (scores_arr < bin_edges[i + 1])
                n_bin = int(mask.sum())
                if n_bin == 0:
                    continue
                avg_prob = float(scores_arr[mask].mean())
                avg_correct = float(y_arr[mask].mean())
                ece += (n_bin / len(y_arr)) * abs(avg_prob - avg_correct)

            overall["calibration_ece"] = round(ece, 4)
            overall["calibration_brier_score"] = round(brier, 4)
    except Exception as exc:
        logger.debug("Could not compute calibration metrics: %s", exc)

    return overall, all_metrics


def build_by_group_table(
    all_metrics: List[Dict[str, Any]],
    group_key: str,
) -> List[Dict[str, Any]]:
    """Aggregate metrics grouped by a key (tissue, split_tag, cell_type, etc.)."""
    groups: Dict[str, List] = defaultdict(list)
    for m in all_metrics:
        key_val = str(m.get(group_key) or "unknown")
        groups[key_val].append(m)

    rows = []
    for grp, mlist in sorted(groups.items()):
        agg = _aggregate(mlist)
        row = {"group": grp, "group_key": group_key}
        row.update(agg)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys: List[str] = []
    seen: set = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Flatten decision_distribution dict for CSV
            flat = dict(row)
            if isinstance(flat.get("decision_distribution"), dict):
                for k, v in flat.pop("decision_distribution").items():
                    flat[f"decision_{k}"] = v
            if isinstance(flat.get("rare_label_subset"), dict):
                flat.pop("rare_label_subset")
            if isinstance(flat.get("unmapped_subset"), dict):
                flat.pop("unmapped_subset")
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate SCA-Specialist predictions")
    parser.add_argument(
        "--predictions",
        default=os.environ.get(
            "PREDICTIONS_FILE",
            str(_PROJECT_DIR / "output" / "infer_qwen3_grounded" / "predictions.jsonl"),
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.environ.get(
            "OUTPUT_DIR",
            str(_PROJECT_DIR / "output" / "eval"),
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.predictions).exists():
        logger.error("Predictions file not found: %s", args.predictions)
        sys.exit(1)

    logger.info("Loading predictions from %s", args.predictions)
    records = load_predictions(args.predictions)
    logger.info("Loaded %d records", len(records))

    if not records:
        logger.error("No records found.")
        sys.exit(1)

    overall, all_metrics = evaluate(records)
    overall["predictions_file"] = args.predictions

    # eval_overall.json
    overall_path = out_dir / "eval_overall.json"
    overall_path.write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved %s", overall_path)

    # eval_by_split.csv
    by_split = build_by_group_table(all_metrics, "split_tag")
    write_csv(by_split, out_dir / "eval_by_split.csv")

    # eval_by_tissue.csv
    by_tissue = build_by_group_table(all_metrics, "tissue")
    write_csv(by_tissue, out_dir / "eval_by_tissue.csv")

    # eval_by_label.csv
    by_label = build_by_group_table(all_metrics, "cell_type")
    write_csv(by_label, out_dir / "eval_by_label.csv")

    # eval_by_hardness.csv
    by_hard = build_by_group_table(all_metrics, "is_hard")
    write_csv(by_hard, out_dir / "eval_by_hardness.csv")

    logger.info("All evaluation files saved to %s", out_dir)

    # Print summary to stdout
    print("\n===== Evaluation Summary =====")
    print(f"N total:           {overall.get('n')}")
    print(f"Parse success:     {overall.get('parse_success_rate')}")
    print(f"Exact match:       {overall.get('exact_cell_type_match_rate')}")
    print(f"Combined match:    {overall.get('cell_type_match_rate_combined')}")
    print(f"CL ID exact:       {overall.get('cl_id_exact_match_rate')}")
    print(f"Top-k retrieval:   {overall.get('topk_retrieval_hit_rate')}")
    print(f"ECE:               {overall.get('calibration_ece', 'N/A')}")
    print(f"Brier score:       {overall.get('calibration_brier_score', 'N/A')}")
    dec = overall.get("decision_distribution", {})
    print(f"Decision dist:     {dec}")
    rare = overall.get("rare_label_subset", {})
    print(f"Rare-label exact:  {rare.get('exact_cell_type_match_rate', 'N/A')} (n={rare.get('n', 0)})")
    unm = overall.get("unmapped_subset", {})
    print(f"Unmapped exact:    {unm.get('exact_cell_type_match_rate', 'N/A')} (n={unm.get('n', 0)})")
    print("==============================\n")


if __name__ == "__main__":
    main()
