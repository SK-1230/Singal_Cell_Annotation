import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# =========================
# 可改参数
# =========================
BASE_MODEL_PATH = "/data/projects/shuke/code/singal_cell_annotation/my_models/Qwen/Qwen3-4B"
ADAPTER_PATH = "/data/projects/shuke/code/singal_cell_annotation/output/qwen3_4b_sc_sft_hf_trl_v2_20260331_210804"
TEST_FILE = "/data/projects/shuke/code/singal_cell_annotation/data/splits/test_messages_no_think_v2.jsonl"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/data/projects/shuke/code/singal_cell_annotation/output/infer_qwen_swift_batch_v2_test_{timestamp}"

MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 1.0


# =========================
# 数据加载
# =========================
def load_examples(test_file: str) -> List[Dict[str, Any]]:
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                examples.append(rec)
            except Exception as e:
                print(f"[Skip] Invalid JSON at line {line_idx}: {e}")

    if not examples:
        raise ValueError(f"No valid examples found in: {test_file}")

    return examples


def get_gold_assistant_content(messages: List[Dict[str, str]]) -> str:
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not assistants:
        return ""
    return assistants[-1].get("content", "")


def get_infer_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [m for m in messages if m.get("role") != "assistant"]


def safe_apply_chat_template(tokenizer, infer_messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            infer_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            infer_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# =========================
# JSON 抽取与解析
# =========================
def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()

    # 先尝试整体解析
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 去掉 </think> 前缀干扰
    text = re.sub(r"^\s*</think>\s*", "", text, flags=re.IGNORECASE).strip()

    # 提取第一个 {...}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None
    return None


def parse_prediction(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    json_block = extract_json_block(raw_text)

    if json_block is None:
        return {
            "parse_ok": False,
            "parsed": None,
            "parse_error": "No valid JSON object found in model output."
        }

    try:
        parsed = json.loads(json_block)
        return {
            "parse_ok": True,
            "parsed": parsed,
            "parse_error": ""
        }
    except Exception as e:
        return {
            "parse_ok": False,
            "parsed": None,
            "parse_error": repr(e)
        }


def parse_gold_json(gold_text: str) -> Dict[str, Any]:
    gold_text = gold_text.strip()
    json_block = extract_json_block(gold_text)

    if json_block is None:
        return {
            "gold_parse_ok": False,
            "gold_parsed": None,
            "gold_parse_error": "No valid JSON object found in gold assistant content."
        }

    try:
        parsed = json.loads(json_block)
        return {
            "gold_parse_ok": True,
            "gold_parsed": parsed,
            "gold_parse_error": ""
        }
    except Exception as e:
        return {
            "gold_parse_ok": False,
            "gold_parsed": None,
            "gold_parse_error": repr(e)
        }


# =========================
# 文本规范化与标签标准化
# =========================
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace("/", " ")
    s = s.replace(",", " ")
    s = re.sub(r"\bhuman\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_label(x: Any) -> List[str]:
    s = normalize_text(x)
    if not s:
        return []
    return s.split()


def normalize_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = normalize_text(x)
    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False
    return None


def canonicalize_cell_type(label: Any) -> str:
    s = normalize_text(label)
    if not s:
        return ""

    # 常见缩写与表达统一
    replacements = {
        "nk": "natural killer",
        "cd4 t cell": "cd4 positive alpha beta t cell",
        "cd8 t cell": "cd8 positive alpha beta t cell",
        "treg": "regulatory t cell",
        "b cell precursor": "precursor b cell",
        "erythroid cell": "erythrocyte",
    }

    # 简单安全替换
    for k, v in replacements.items():
        s = re.sub(rf"\b{k}\b", v, s)

    # 清理一些冗余描述
    s = s.replace("cell, terminally differentiated", "terminally differentiated cell")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_lineage(label: Any) -> str:
    s = canonicalize_cell_type(label)

    # 顺序尽量从特殊到一般
    if not s:
        return "unknown"

    if "gamma delta t" in s:
        return "gamma_delta_t"
    if "regulatory t" in s:
        return "t_cell"
    if "natural killer" in s:
        return "nk"
    if "t cell" in s or "alpha beta t" in s:
        return "t_cell"
    if "b cell" in s or "plasma" in s:
        return "b_cell"
    if "monocyte" in s:
        return "monocyte"
    if "dendritic" in s:
        return "dendritic"
    if "macrophage" in s:
        return "macrophage"
    if "erythrocyte" in s or "erythroid" in s:
        return "erythroid"
    if "megakaryocyte" in s or "platelet" in s:
        return "megakaryocyte_platelet"
    if "hematopoietic precursor" in s or "precursor" in s or "pro b" in s:
        return "precursor"
    if "neutrophil" in s:
        return "neutrophil"
    return "other"


def extract_subtype_flags(label: Any) -> Dict[str, bool]:
    s = canonicalize_cell_type(label)
    return {
        "cd4": "cd4" in s,
        "cd8": "cd8" in s,
        "memory": "memory" in s,
        "naive": "naive" in s or "thymus derived" in s,
        "activated": "activated" in s,
        "effector": "effector" in s,
        "terminal": "terminally differentiated" in s,
        "gamma_delta": "gamma delta" in s,
        "nk_bright": "cd56 bright" in s,
        "nk_dim": "cd56 dim" in s or "cd16 positive" in s,
        "cd16_negative": "cd16 negative" in s,
    }


def label_specificity_score(label: Any) -> int:
    """
    粗略估计标签细粒度：
    修饰词越多，分数越高，表示越“细”。
    """
    s = canonicalize_cell_type(label)
    if not s:
        return 0

    score = 0
    keywords = [
        "cd4", "cd8", "memory", "naive", "activated", "effector",
        "terminally differentiated", "gamma delta", "alpha beta",
        "cd16 positive", "cd16 negative", "cd56 bright", "cd56 dim",
        "classical", "non classical", "conventional"
    ]
    for kw in keywords:
        if kw in s:
            score += 1

    # token 更长通常更具体，但只给轻微加成
    score += min(len(tokenize_label(s)) // 4, 2)
    return score


def compare_granularity(pred_label: Any, gold_label: Any) -> str:
    pred_score = label_specificity_score(pred_label)
    gold_score = label_specificity_score(gold_label)

    if pred_score == gold_score:
        return "same"
    if pred_score < gold_score:
        return "coarser_than_gold"
    return "finer_than_gold"


def same_major_lineage(pred_label: Any, gold_label: Any) -> bool:
    return infer_lineage(pred_label) == infer_lineage(gold_label)


def subtype_conflict(pred_label: Any, gold_label: Any) -> bool:
    """
    同一大类内的重要冲突：
    例如 CD4 vs CD8, naive vs memory, NK bright vs NK dim
    """
    p = extract_subtype_flags(pred_label)
    g = extract_subtype_flags(gold_label)

    # 强冲突
    strong_pairs = [
        ("cd4", "cd8"),
        ("naive", "memory"),
        ("gamma_delta", "cd4"),
        ("gamma_delta", "cd8"),
        ("nk_bright", "nk_dim"),
    ]

    for a, b in strong_pairs:
        if (p[a] and g[b]) or (p[b] and g[a]):
            return True

    return False


def compare_cell_type(pred_label: Any, gold_label: Any) -> Dict[str, Any]:
    pred_raw = "" if pred_label is None else str(pred_label)
    gold_raw = "" if gold_label is None else str(gold_label)

    pred_norm = canonicalize_cell_type(pred_raw)
    gold_norm = canonicalize_cell_type(gold_raw)

    if not pred_norm or not gold_norm:
        return {
            "exact_match": None,
            "normalized_exact_match": None,
            "same_lineage": None,
            "granularity_relation": None,
            "match_level": "unavailable",
            "severe_error": None,
        }

    exact_match = normalize_text(pred_raw) == normalize_text(gold_raw)
    normalized_exact_match = pred_norm == gold_norm
    lineage_match = same_major_lineage(pred_norm, gold_norm)
    granularity_relation = compare_granularity(pred_norm, gold_norm)

    pred_tokens = set(tokenize_label(pred_norm))
    gold_tokens = set(tokenize_label(gold_norm))
    token_overlap = len(pred_tokens & gold_tokens) / max(1, len(pred_tokens | gold_tokens))

    if exact_match:
        match_level = "exact"
        severe_error = False
    elif normalized_exact_match:
        match_level = "normalized_exact"
        severe_error = False
    elif lineage_match:
        if subtype_conflict(pred_norm, gold_norm):
            match_level = "same_lineage_conflict"
        elif token_overlap >= 0.5:
            match_level = "same_lineage_close"
        else:
            match_level = "same_lineage_broad"
        severe_error = False
    else:
        match_level = "cross_lineage_error"
        severe_error = True

    return {
        "exact_match": exact_match,
        "normalized_exact_match": normalized_exact_match,
        "same_lineage": lineage_match,
        "granularity_relation": granularity_relation,
        "match_level": match_level,
        "severe_error": severe_error,
        "pred_lineage": infer_lineage(pred_norm),
        "gold_lineage": infer_lineage(gold_norm),
        "pred_canonical": pred_norm,
        "gold_canonical": gold_norm,
        "token_overlap": round(token_overlap, 4),
    }


def compare_list_overlap(pred_list: Any, gold_list: Any) -> Dict[str, Any]:
    pred_items = {normalize_text(x) for x in (pred_list or []) if normalize_text(x)}
    gold_items = {normalize_text(x) for x in (gold_list or []) if normalize_text(x)}

    if not pred_items and not gold_items:
        return {"jaccard": None, "pred_count": 0, "gold_count": 0, "shared_count": 0}

    inter = pred_items & gold_items
    union = pred_items | gold_items
    jaccard = len(inter) / max(1, len(union))

    return {
        "jaccard": round(jaccard, 4),
        "pred_count": len(pred_items),
        "gold_count": len(gold_items),
        "shared_count": len(inter),
    }


def compare_prediction_with_gold(pred: Optional[Dict[str, Any]], gold: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if pred is None or gold is None:
        return {
            "cell_type_exact_match": None,
            "cell_type_normalized_exact_match": None,
            "cell_type_same_lineage": None,
            "cell_type_granularity_relation": None,
            "cell_type_match_level": "unavailable",
            "cell_type_severe_error": None,

            "confidence_label_match": None,
            "decision_match": None,
            "need_manual_review_match": None,
            "novelty_flag_match": None,
            "evidence_support_level_match": None,

            "supporting_markers_jaccard": None,
        }

    cell_cmp = compare_cell_type(pred.get("cell_type"), gold.get("cell_type"))
    marker_cmp = compare_list_overlap(
        pred.get("supporting_markers", []),
        gold.get("supporting_markers", [])
    )

    pred_conf_label = normalize_text(pred.get("confidence_label"))
    gold_conf_label = normalize_text(gold.get("confidence_label"))

    pred_decision = normalize_text(pred.get("decision"))
    gold_decision = normalize_text(gold.get("decision"))

    pred_need_review = normalize_bool(pred.get("need_manual_review"))
    gold_need_review = normalize_bool(gold.get("need_manual_review"))

    pred_novel = normalize_bool(pred.get("novelty_flag"))
    gold_novel = normalize_bool(gold.get("novelty_flag"))

    pred_evidence = normalize_text(pred.get("evidence_support_level"))
    gold_evidence = normalize_text(gold.get("evidence_support_level"))

    return {
        "cell_type_exact_match": cell_cmp["exact_match"],
        "cell_type_normalized_exact_match": cell_cmp["normalized_exact_match"],
        "cell_type_same_lineage": cell_cmp["same_lineage"],
        "cell_type_granularity_relation": cell_cmp["granularity_relation"],
        "cell_type_match_level": cell_cmp["match_level"],
        "cell_type_severe_error": cell_cmp["severe_error"],
        "pred_lineage": cell_cmp["pred_lineage"],
        "gold_lineage": cell_cmp["gold_lineage"],
        "pred_canonical_cell_type": cell_cmp["pred_canonical"],
        "gold_canonical_cell_type": cell_cmp["gold_canonical"],
        "cell_type_token_overlap": cell_cmp["token_overlap"],

        "confidence_label_match": (pred_conf_label == gold_conf_label) if pred_conf_label or gold_conf_label else None,
        "decision_match": (pred_decision == gold_decision) if pred_decision or gold_decision else None,
        "need_manual_review_match": (pred_need_review == gold_need_review) if pred_need_review is not None and gold_need_review is not None else None,
        "novelty_flag_match": (pred_novel == gold_novel) if pred_novel is not None and gold_novel is not None else None,
        "evidence_support_level_match": (pred_evidence == gold_evidence) if pred_evidence or gold_evidence else None,

        "supporting_markers_jaccard": marker_cmp["jaccard"],
        "supporting_markers_shared_count": marker_cmp["shared_count"],
    }


# =========================
# 保存
# =========================
def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    flat_rows = []
    for row in rows:
        flat_rows.append({
            "sample_id": row.get("sample_id"),
            "parse_ok": row.get("parse_ok"),
            "gold_parse_ok": row.get("gold_parse_ok"),

            "pred_cell_type": row.get("pred_cell_type"),
            "gold_cell_type": row.get("gold_cell_type"),
            "pred_canonical_cell_type": row.get("pred_canonical_cell_type"),
            "gold_canonical_cell_type": row.get("gold_canonical_cell_type"),
            "pred_lineage": row.get("pred_lineage"),
            "gold_lineage": row.get("gold_lineage"),

            "cell_type_exact_match": row.get("cell_type_exact_match"),
            "cell_type_normalized_exact_match": row.get("cell_type_normalized_exact_match"),
            "cell_type_same_lineage": row.get("cell_type_same_lineage"),
            "cell_type_granularity_relation": row.get("cell_type_granularity_relation"),
            "cell_type_match_level": row.get("cell_type_match_level"),
            "cell_type_severe_error": row.get("cell_type_severe_error"),
            "cell_type_token_overlap": row.get("cell_type_token_overlap"),

            "pred_confidence_label": row.get("pred_confidence_label"),
            "gold_confidence_label": row.get("gold_confidence_label"),
            "confidence_label_match": row.get("confidence_label_match"),

            "pred_confidence_score": row.get("pred_confidence_score"),
            "gold_confidence_score": row.get("gold_confidence_score"),

            "pred_need_manual_review": row.get("pred_need_manual_review"),
            "gold_need_manual_review": row.get("gold_need_manual_review"),
            "need_manual_review_match": row.get("need_manual_review_match"),

            "pred_decision": row.get("pred_decision"),
            "gold_decision": row.get("gold_decision"),
            "decision_match": row.get("decision_match"),

            "pred_novelty_flag": row.get("pred_novelty_flag"),
            "gold_novelty_flag": row.get("gold_novelty_flag"),
            "novelty_flag_match": row.get("novelty_flag_match"),

            "pred_evidence_support_level": row.get("pred_evidence_support_level"),
            "gold_evidence_support_level": row.get("gold_evidence_support_level"),
            "evidence_support_level_match": row.get("evidence_support_level_match"),

            "supporting_markers_jaccard": row.get("supporting_markers_jaccard"),
            "pred_supporting_markers": "|".join(row.get("pred_supporting_markers", [])),
            "gold_supporting_markers": "|".join(row.get("gold_supporting_markers", [])),

            "pred_rationale": row.get("pred_rationale"),
            "gold_rationale": row.get("gold_rationale"),
            "raw_model_output": row.get("raw_model_output"),
        })

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _acc(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None
    return round(sum(1 for r in valid if r.get(key) is True) / len(valid), 6)


def save_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    total = len(rows)
    parse_ok = sum(1 for r in rows if r.get("parse_ok") is True)
    gold_parse_ok = sum(1 for r in rows if r.get("gold_parse_ok") is True)

    valid_rows = [r for r in rows if r.get("cell_type_match_level") != "unavailable"]

    match_level_counts = {}
    for r in valid_rows:
        lvl = r.get("cell_type_match_level")
        match_level_counts[lvl] = match_level_counts.get(lvl, 0) + 1

    severe_error_rows = [r for r in valid_rows if r.get("cell_type_severe_error") is not None]
    severe_error_rate = None
    if severe_error_rows:
        severe_error_rate = round(
            sum(1 for r in severe_error_rows if r.get("cell_type_severe_error") is True) / len(severe_error_rows),
            6
        )

    same_lineage_but_not_exact_rows = [
        r for r in valid_rows
        if r.get("cell_type_same_lineage") is True and r.get("cell_type_normalized_exact_match") is False
    ]

    marker_jaccards = [
        r.get("supporting_markers_jaccard")
        for r in rows
        if r.get("supporting_markers_jaccard") is not None
    ]

    # 高风险错误：跨谱系严重错误 + 高置信 + accept
    high_risk_errors = [
        r for r in valid_rows
        if r.get("cell_type_severe_error") is True
        and normalize_text(r.get("pred_confidence_label")) == "high"
        and normalize_text(r.get("pred_decision")) == "accept"
    ]

    summary = {
        "total_examples": total,
        "parse_ok_examples": parse_ok,
        "gold_parse_ok_examples": gold_parse_ok,
        "parse_ok_rate": round(parse_ok / total, 6) if total else None,
        "gold_parse_ok_rate": round(gold_parse_ok / total, 6) if total else None,

        "cell_type_exact_accuracy": _acc(valid_rows, "cell_type_exact_match"),
        "cell_type_normalized_exact_accuracy": _acc(valid_rows, "cell_type_normalized_exact_match"),
        "cell_type_same_lineage_rate": _acc(valid_rows, "cell_type_same_lineage"),
        "cell_type_severe_error_rate": severe_error_rate,
        "same_lineage_but_not_exact_count": len(same_lineage_but_not_exact_rows),

        "confidence_label_match_accuracy": _acc(rows, "confidence_label_match"),
        "need_manual_review_match_accuracy": _acc(rows, "need_manual_review_match"),
        "decision_match_accuracy": _acc(rows, "decision_match"),
        "novelty_flag_match_accuracy": _acc(rows, "novelty_flag_match"),
        "evidence_support_level_match_accuracy": _acc(rows, "evidence_support_level_match"),

        "mean_supporting_markers_jaccard": _mean(marker_jaccards),
        "high_risk_error_count": len(high_risk_errors),

        "match_level_counts": match_level_counts,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================
# 主流程
# =========================
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = output_dir / "predictions.jsonl"
    result_csv = output_dir / "predictions.csv"
    result_summary = output_dir / "summary.json"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    examples = load_examples(TEST_FILE)
    print(f"Loaded {len(examples)} test examples.")

    all_rows = []

    for idx, rec in enumerate(tqdm(examples, desc="Infer", unit="sample")):
        messages = rec.get("messages", [])
        if not messages:
            print(f"[Skip] sample {idx}: empty messages")
            continue

        gold_text = get_gold_assistant_content(messages)
        infer_messages = get_infer_messages(messages)
        prompt_text = safe_apply_chat_template(tokenizer, infer_messages)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        pred_info = parse_prediction(generated)
        gold_info = parse_gold_json(gold_text)

        pred_parsed = pred_info["parsed"]
        gold_parsed = gold_info["gold_parsed"]
        cmp_info = compare_prediction_with_gold(pred_parsed, gold_parsed)

        row = {
            "sample_id": idx,
            "prompt_text": prompt_text,
            "gold_text": gold_text,
            "raw_model_output": generated,

            "parse_ok": pred_info["parse_ok"],
            "parse_error": pred_info["parse_error"],
            "gold_parse_ok": gold_info["gold_parse_ok"],
            "gold_parse_error": gold_info["gold_parse_error"],

            "pred_json": pred_parsed,
            "gold_json": gold_parsed,

            "pred_cell_type": pred_parsed.get("cell_type") if pred_parsed else None,
            "gold_cell_type": gold_parsed.get("cell_type") if gold_parsed else None,

            "pred_supporting_markers": pred_parsed.get("supporting_markers", []) if pred_parsed else [],
            "gold_supporting_markers": gold_parsed.get("supporting_markers", []) if gold_parsed else [],

            "pred_confidence_label": pred_parsed.get("confidence_label") if pred_parsed else None,
            "gold_confidence_label": gold_parsed.get("confidence_label") if gold_parsed else None,

            "pred_confidence_score": pred_parsed.get("confidence_score") if pred_parsed else None,
            "gold_confidence_score": gold_parsed.get("confidence_score") if gold_parsed else None,

            "pred_need_manual_review": pred_parsed.get("need_manual_review") if pred_parsed else None,
            "gold_need_manual_review": gold_parsed.get("need_manual_review") if gold_parsed else None,

            "pred_decision": pred_parsed.get("decision") if pred_parsed else None,
            "gold_decision": gold_parsed.get("decision") if gold_parsed else None,

            "pred_novelty_flag": pred_parsed.get("novelty_flag") if pred_parsed else None,
            "gold_novelty_flag": gold_parsed.get("novelty_flag") if gold_parsed else None,

            "pred_evidence_support_level": pred_parsed.get("evidence_support_level") if pred_parsed else None,
            "gold_evidence_support_level": gold_parsed.get("evidence_support_level") if gold_parsed else None,

            "pred_rationale": pred_parsed.get("rationale") if pred_parsed else None,
            "gold_rationale": gold_parsed.get("rationale") if gold_parsed else None,

            **cmp_info,
        }

        all_rows.append(row)

    save_jsonl(all_rows, result_jsonl)
    save_csv(all_rows, result_csv)
    save_summary(all_rows, result_summary)

    print("\n===== Done =====")
    print(f"Saved JSONL:   {result_jsonl}")
    print(f"Saved CSV:     {result_csv}")
    print(f"Saved Summary: {result_summary}")


if __name__ == "__main__":
    main()


# 运行方式：
# cd /data/projects/shuke/code/singal_cell_annotation
# python -u scripts/infer/infer_qwen3_swift_batch_V2.py 2>&1 | tee data/meta/infer_qwen3_swift_batch_V2.log