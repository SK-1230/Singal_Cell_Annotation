"""
Output parser for SCA-Specialist model responses.

Handles:
- JSON extraction from raw model output (with or without <think> blocks)
- Schema validation against annotation_output_v2
- Graceful fallback for malformed outputs
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Required keys for annotation_output_v2
REQUIRED_KEYS_V2 = {
    "cell_type", "cell_ontology_id", "parent_cell_type",
    "supporting_markers", "contradictory_markers",
    "confidence_label", "confidence_score",
    "need_manual_review", "decision", "novelty_flag",
    "evidence_support_level", "rationale",
}

VALID_CONFIDENCE_LABELS = {"high", "medium", "low"}
VALID_DECISIONS = {"accept", "review", "unresolved", "novel_candidate"}
VALID_EVIDENCE_LEVELS = {"strong", "moderate", "weak", "conflicting"}


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> block from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract the first JSON object from raw text.
    Tries: direct parse → find first '{' brace match → regex fallback.
    """
    text = text.strip()

    # Try direct parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Find first '{' and try to match braces
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break

    # Regex fallback: grab everything between outermost braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group())
            return match.group()
        except json.JSONDecodeError:
            pass

    return None


def parse_annotation_output_v2(
    raw_output: str,
    strict: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse raw model output into an annotation_output_v2 dict.

    Returns (parsed_dict, error_message).
    If parsing fails, parsed_dict is None and error_message explains why.

    Parameters
    ----------
    raw_output : raw string from model
    strict     : if True, reject outputs with missing required keys
    """
    # Strip think block
    cleaned = strip_think_block(raw_output)

    json_str = extract_json_from_text(cleaned)
    if json_str is None:
        return None, "no_json_found"

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"

    if not isinstance(obj, dict):
        return None, "not_a_dict"

    # Normalize and fill missing optional fields with defaults
    obj = _normalize_output_v2(obj)

    if strict:
        missing = REQUIRED_KEYS_V2 - set(obj.keys())
        if missing:
            return None, f"missing_keys: {sorted(missing)}"

    return obj, ""


def _normalize_output_v2(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and fill defaults for annotation_output_v2 fields.
    Handles common model quirks (wrong enum values, missing keys, etc.).
    """
    # Normalize confidence_label
    cl = str(obj.get("confidence_label", "")).lower().strip()
    if cl not in VALID_CONFIDENCE_LABELS:
        # Try to map common variants
        if "high" in cl:
            cl = "high"
        elif "low" in cl:
            cl = "low"
        else:
            cl = "medium"
    obj["confidence_label"] = cl

    # Normalize decision
    dec = str(obj.get("decision", "")).lower().strip().replace(" ", "_")
    if dec not in VALID_DECISIONS:
        # Default based on confidence
        dec = "review" if cl in ("low", "medium") else "accept"
    obj["decision"] = dec

    # Normalize evidence_support_level
    esl = str(obj.get("evidence_support_level", "")).lower().strip()
    if esl not in VALID_EVIDENCE_LEVELS:
        esl = "moderate"
    obj["evidence_support_level"] = esl

    # Ensure list fields are lists
    for key in ("supporting_markers", "contradictory_markers"):
        if not isinstance(obj.get(key), list):
            val = obj.get(key, "")
            if isinstance(val, str) and val:
                obj[key] = [v.strip() for v in val.split(",") if v.strip()]
            else:
                obj[key] = []

    # Ensure bool fields
    for key in ("need_manual_review", "novelty_flag"):
        val = obj.get(key, False)
        if isinstance(val, str):
            obj[key] = val.lower() in ("true", "yes", "1")
        else:
            obj[key] = bool(val)

    # Ensure confidence_score is float or None
    cs = obj.get("confidence_score")
    if cs is not None:
        try:
            cs = round(float(cs), 4)
            cs = max(0.0, min(1.0, cs))
        except (TypeError, ValueError):
            cs = None
    obj["confidence_score"] = cs

    # Defaults for optional fields
    obj.setdefault("cell_ontology_id", None)
    obj.setdefault("parent_cell_type", None)
    obj.setdefault("contradictory_markers", [])
    obj.setdefault("rationale", "")

    return obj


def parse_annotation_output_v1(raw_output: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse legacy v1 assistant output format:
        cell_type, supporting_markers, confidence, need_manual_review, rationale
    """
    cleaned = strip_think_block(raw_output)
    json_str = extract_json_from_text(cleaned)
    if json_str is None:
        return None, "no_json_found"
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"
    if not isinstance(obj, dict):
        return None, "not_a_dict"
    return obj, ""
