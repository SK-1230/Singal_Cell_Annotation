import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ===== 可改参数 =====
BASE_MODEL_PATH = "/data/projects/shuke/code/singal_cell_annotation/my_models/Qwen/Qwen3-4B"
ADAPTER_PATH = "/data/projects/shuke/code/singal_cell_annotation/output/qwen3_4b_sc_sft_swift_v2/v0-20260326-181619/checkpoint-175"
TEST_FILE = "/data/projects/shuke/code/singal_cell_annotation/data/splits/test_messages_no_think.jsonl"
OUTPUT_DIR = "/data/projects/shuke/code/singal_cell_annotation/output/infer_qwen3_swift_batch"
MAX_NEW_TOKENS = 256


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


def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()

    # 优先尝试直接整体解析
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 提取第一个 {...} 块
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


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def compare_prediction_with_gold(pred: Optional[Dict[str, Any]], gold: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if pred is None or gold is None:
        return {
            "cell_type_match": None,
            "confidence_match": None,
            "need_manual_review_match": None,
        }

    pred_cell_type = normalize_text(pred.get("cell_type"))
    gold_cell_type = normalize_text(gold.get("cell_type"))

    pred_conf = normalize_text(pred.get("confidence"))
    gold_conf = normalize_text(gold.get("confidence"))

    pred_review = pred.get("need_manual_review")
    gold_review = gold.get("need_manual_review")

    return {
        "cell_type_match": pred_cell_type == gold_cell_type,
        "confidence_match": pred_conf == gold_conf,
        "need_manual_review_match": pred_review == gold_review,
    }


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
            "cell_type_match": row.get("cell_type_match"),
            "pred_confidence": row.get("pred_confidence"),
            "gold_confidence": row.get("gold_confidence"),
            "confidence_match": row.get("confidence_match"),
            "pred_need_manual_review": row.get("pred_need_manual_review"),
            "gold_need_manual_review": row.get("gold_need_manual_review"),
            "need_manual_review_match": row.get("need_manual_review_match"),
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


def save_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    total = len(rows)
    parse_ok = sum(1 for r in rows if r.get("parse_ok") is True)
    gold_parse_ok = sum(1 for r in rows if r.get("gold_parse_ok") is True)

    valid_compare_rows = [r for r in rows if r.get("cell_type_match") is not None]
    cell_type_acc = (
        sum(1 for r in valid_compare_rows if r.get("cell_type_match") is True) / len(valid_compare_rows)
        if valid_compare_rows else None
    )

    conf_compare_rows = [r for r in rows if r.get("confidence_match") is not None]
    conf_acc = (
        sum(1 for r in conf_compare_rows if r.get("confidence_match") is True) / len(conf_compare_rows)
        if conf_compare_rows else None
    )

    review_compare_rows = [r for r in rows if r.get("need_manual_review_match") is not None]
    review_acc = (
        sum(1 for r in review_compare_rows if r.get("need_manual_review_match") is True) / len(review_compare_rows)
        if review_compare_rows else None
    )

    summary = {
        "total_examples": total,
        "parse_ok_examples": parse_ok,
        "gold_parse_ok_examples": gold_parse_ok,
        "cell_type_match_accuracy": cell_type_acc,
        "confidence_match_accuracy": conf_acc,
        "need_manual_review_match_accuracy": review_acc,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


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
        dtype=torch.bfloat16,   # 用 dtype，避免 torch_dtype 的弃用提醒
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
                do_sample=False,
                temperature=1.0,
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

            "pred_confidence": pred_parsed.get("confidence") if pred_parsed else None,
            "gold_confidence": gold_parsed.get("confidence") if gold_parsed else None,

            "pred_need_manual_review": pred_parsed.get("need_manual_review") if pred_parsed else None,
            "gold_need_manual_review": gold_parsed.get("need_manual_review") if gold_parsed else None,

            "pred_rationale": pred_parsed.get("rationale") if pred_parsed else None,
            "gold_rationale": gold_parsed.get("rationale") if gold_parsed else None,

            "cell_type_match": cmp_info["cell_type_match"],
            "confidence_match": cmp_info["confidence_match"],
            "need_manual_review_match": cmp_info["need_manual_review_match"],
        }
        all_rows.append(row)

    save_jsonl(all_rows, result_jsonl)
    save_csv(all_rows, result_csv)
    save_summary(all_rows, result_summary)

    print("\n===== Done =====")
    print(f"Saved JSONL: {result_jsonl}")
    print(f"Saved CSV:   {result_csv}")
    print(f"Saved Summary: {result_summary}")


if __name__ == "__main__":
    main()


# 运行方式：
# cd /data/projects/shuke/code/singal_cell_annotation
# python -u scripts/infer/infer_qwen3_swift_batch.py 2>&1 | tee data/meta/infer_qwen3_swift_batch.log