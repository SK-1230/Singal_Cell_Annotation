"""
Confidence calibration for SCA-Specialist (Phase 3).

Provides a lightweight sklearn-based calibrator that takes model output
features and predicts a calibrated confidence score in [0, 1].

Key functions:
    build_calibration_features(merged_output)  → list[float]
    ConfidenceCalibrator                        — wraps sklearn LR/isotonic
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_CONFIDENCE_LABEL_MAP = {"high": 1.0, "medium": 0.5, "low": 0.0}
_EVIDENCE_LEVEL_MAP = {"strong": 1.0, "moderate": 0.67, "weak": 0.33, "conflicting": 0.0}


def build_calibration_features(merged: Dict[str, Any]) -> List[float]:
    """
    Extract a fixed-length feature vector from a merged grounded output dict.

    Features (8 dimensions):
        0: confidence_label  (encoded: high=1.0, medium=0.5, low=0.0)
        1: confidence_score  (model raw, [0,1])
        2: retrieval_support_score ([0,1])
        3: evidence_support_level (encoded: strong=1.0 … conflicting=0.0)
        4: n_cells (log-normalized, clipped at 10000)
        5: ontology_mapped   (1 if cell_ontology_id is set, else 0)
        6: contradictory_marker_count (clipped at 10, /10)
        7: novelty_flag (1 if True else 0)
    """
    cl_label = (merged.get("confidence_label") or "low").strip().lower()
    cl_encoded = _CONFIDENCE_LABEL_MAP.get(cl_label, 0.0)

    conf_score = float(merged.get("confidence_score") or 0.0)
    retrieval_score = float(merged.get("retrieval_support_score") or 0.0)

    ev_level = (merged.get("evidence_support_level") or "weak").strip().lower()
    ev_encoded = _EVIDENCE_LEVEL_MAP.get(ev_level, 0.33)

    n_cells = float(merged.get("n_cells") or 0)
    n_cells_log = float(np.log1p(min(n_cells, 10000)) / np.log1p(10000))

    cl_id = merged.get("cell_ontology_id") or ""
    ontology_mapped = 1.0 if cl_id and cl_id.startswith("CL:") else 0.0

    contra = merged.get("contradictory_markers") or []
    contra_count = min(len(contra), 10) / 10.0

    novelty = 1.0 if merged.get("novelty_flag") else 0.0

    return [
        cl_encoded,
        conf_score,
        retrieval_score,
        ev_encoded,
        n_cells_log,
        ontology_mapped,
        contra_count,
        novelty,
    ]


# ---------------------------------------------------------------------------
# Calibrator class
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """
    Lightweight confidence calibrator for SCA-Specialist.

    Wraps a sklearn binary classifier (logistic regression or isotonic)
    trained on (features, is_correct) pairs from a validation set.

    The calibrated probability P(is_correct) is used as the final
    confidence score.
    """

    def __init__(self, method: str = "logistic"):
        """
        Parameters
        ----------
        method : 'logistic' | 'isotonic'
        """
        if method not in ("logistic", "isotonic"):
            raise ValueError(f"method must be 'logistic' or 'isotonic', got {method!r}")
        self.method = method
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConfidenceCalibrator":
        """
        Fit the calibrator.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix from build_calibration_features
        y : (n_samples,) binary labels — 1 if prediction was correct, 0 otherwise
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        if self.method == "logistic":
            self._model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            self._model.fit(X, y)
        else:
            # Isotonic regression on a single score (confidence_score, index 1)
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(X[:, 1], y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return (n_samples, 2) probability array where col 1 = P(correct).
        """
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted yet — call fit() first.")
        if self.method == "logistic":
            return self._model.predict_proba(X)
        else:
            # Isotonic: only uses the confidence_score feature
            probs = self._model.predict(X[:, 1])
            return np.column_stack([1 - probs, probs])

    def calibrate_score(self, merged: Dict[str, Any]) -> float:
        """Convenience: extract features from merged dict and return P(correct)."""
        features = build_calibration_features(merged)
        X = np.array([features])
        prob = self.predict_proba(X)[0][1]
        return round(float(prob), 4)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)
        logger.info("Calibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ConfidenceCalibrator":
        import joblib
        obj = joblib.load(path)
        logger.info("Calibrator loaded from %s", path)
        return obj
