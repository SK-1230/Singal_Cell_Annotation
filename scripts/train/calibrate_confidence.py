#!/usr/bin/env python3
"""
calibrate_confidence.py — Train a lightweight confidence calibrator (Phase 3).

Input:
    Predictions JSONL from infer_qwen3_grounded.py (val or test split)
    that contains both model predictions and gold labels.

Output:
    output/calibration/confidence_calibrator.joblib
    output/calibration/calibration_metrics.json

Usage:
    python scripts/train/calibrate_confidence.py \\
        --predictions output/infer_qwen3_grounded/predictions.jsonl \\
        --output_dir output/calibration

Environment overrides:
    PREDICTIONS_FILE, OUTPUT_DIR, METHOD (logistic|isotonic)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


def load_predictions(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(
    records: List[Dict[str, Any]],
) -> Tuple["np.ndarray", "np.ndarray", List[Dict[str, Any]]]:
    """
    Build feature matrix X and label vector y from predictions.

    y[i] = 1 if cell_type matches gold, 0 otherwise.
    Skips records without gold labels.
    """
    import numpy as np
    from sca.model.calibration import build_calibration_features

    X_rows: List[List[float]] = []
    y_vals: List[int] = []
    used: List[Dict[str, Any]] = []

    for rec in records:
        gold = (rec.get("_gold_cell_type") or "").strip().lower()
        pred = (rec.get("cell_type") or "").strip().lower()
        if not gold:
            continue  # Skip records without gold

        is_correct = int(gold == pred or gold in pred or pred in gold)
        features = build_calibration_features(rec)
        X_rows.append(features)
        y_vals.append(is_correct)
        used.append(rec)

    if not X_rows:
        raise ValueError("No records with gold labels found in predictions file.")

    return np.array(X_rows, dtype=float), np.array(y_vals, dtype=int), used


def compute_calibration_metrics(
    y_true: "np.ndarray",
    y_proba: "np.ndarray",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ECE, Brier score, accuracy, and confusion breakdown."""
    import numpy as np

    # Brier score
    brier = float(np.mean((y_proba - y_true) ** 2))

    # Accuracy at 0.5 threshold
    y_pred_binary = (y_proba >= 0.5).astype(int)
    accuracy = float(np.mean(y_pred_binary == y_true))

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bin_stats.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "avg_prob": None, "avg_correct": None})
            continue
        avg_prob = float(y_proba[mask].mean())
        avg_correct = float(y_true[mask].mean())
        ece += (n_bin / n_total) * abs(avg_prob - avg_correct)
        bin_stats.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": n_bin,
            "avg_prob": round(avg_prob, 4),
            "avg_correct": round(avg_correct, 4),
        })

    return {
        "n_samples": n_total,
        "accuracy_at_0.5": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "positive_rate": round(float(y_true.mean()), 4),
        "calibration_bins": bin_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SCA confidence calibrator")
    parser.add_argument(
        "--predictions",
        default=os.environ.get(
            "PREDICTIONS_FILE",
            str(_PROJECT_DIR / "output" / "infer_qwen3_grounded" / "predictions.jsonl"),
        ),
        help="Predictions JSONL from infer_qwen3_grounded.py",
    )
    parser.add_argument(
        "--output_dir",
        default=os.environ.get(
            "OUTPUT_DIR",
            str(_PROJECT_DIR / "output" / "calibration"),
        ),
        help="Directory for calibrator and metrics",
    )
    parser.add_argument(
        "--method",
        default=os.environ.get("METHOD", "logistic"),
        choices=["logistic", "isotonic"],
        help="Calibration method",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for hold-out evaluation",
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

    import numpy as np
    X, y, used_records = build_dataset(records)
    logger.info("Dataset: %d samples, %d features, %.1f%% correct",
                len(y), X.shape[1], 100 * y.mean())

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    from sca.model.calibration import ConfidenceCalibrator

    cal = ConfidenceCalibrator(method=args.method)
    logger.info("Fitting calibrator (method=%s, n_train=%d) ...", args.method, len(y_train))
    cal.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_proba_test = cal.predict_proba(X_test)[:, 1]
    metrics = compute_calibration_metrics(y_test, y_proba_test)
    metrics["method"] = args.method
    metrics["predictions_file"] = args.predictions

    logger.info(
        "Calibration metrics — ECE: %.4f | Brier: %.4f | Accuracy@0.5: %.4f",
        metrics["ece"], metrics["brier_score"], metrics["accuracy_at_0.5"],
    )

    # Save calibrator
    cal_path = out_dir / "confidence_calibrator.joblib"
    cal.save(str(cal_path))

    # Save metrics
    metrics_path = out_dir / "calibration_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Calibration metrics saved to %s", metrics_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
