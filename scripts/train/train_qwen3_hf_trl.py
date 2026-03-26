import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


# =========================
# 可改参数
# =========================
BASE_MODEL_PATH = "/data/projects/shuke/code/singal_cell_annotation/my_models/Qwen/Qwen3-4B"
TRAIN_FILE = "/data/projects/shuke/code/singal_cell_annotation/data/splits/train_messages_no_think.jsonl"
VAL_FILE = "/data/projects/shuke/code/singal_cell_annotation/data/splits/val_messages_no_think.jsonl"
OUTPUT_DIR = "/data/projects/shuke/code/singal_cell_annotation/output/qwen3_4b_sc_sft_hf_trl_v1"

SEED = 42
MAX_LENGTH = 1024
NUM_TRAIN_EPOCHS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 1
EVAL_STEPS = 5
SAVE_STEPS = 5
SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 2

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

USE_BF16 = True
USE_FP16 = False
GRADIENT_CHECKPOINTING = True
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True
ENABLE_THINKING = False


@dataclass
class ExampleStats:
    total: int = 0
    kept: int = 0
    skipped_empty: int = 0
    skipped_no_assistant: int = 0
    skipped_bad_format: int = 0


def check_file(path: str, name: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} line {line_no}: {e}") from e
    return records


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    chunks.append(str(text))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content)


def convert_messages_to_prompt_completion(
    records: List[Dict[str, Any]],
    tokenizer,
    source_name: str,
) -> Dataset:
    stats = ExampleStats(total=len(records))
    rows: List[Dict[str, str]] = []

    for idx, rec in enumerate(records):
        messages = rec.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            stats.skipped_bad_format += 1
            continue

        normalized_messages = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = _normalize_message_content(m.get("content"))
            if not role:
                continue
            normalized_messages.append({"role": role, "content": content})

        if not normalized_messages:
            stats.skipped_bad_format += 1
            continue

        last_assistant_idx = None
        for i in range(len(normalized_messages) - 1, -1, -1):
            if normalized_messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            stats.skipped_no_assistant += 1
            continue

        prompt_messages = normalized_messages[:last_assistant_idx]
        assistant_message = normalized_messages[last_assistant_idx]
        completion = assistant_message["content"].strip()

        if not prompt_messages or not completion:
            stats.skipped_empty += 1
            continue

        try:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt = prompt.strip()
        if not prompt:
            stats.skipped_empty += 1
            continue

        rows.append({
            "prompt": prompt,
            "completion": completion,
            "source": source_name,
            "example_id": str(idx),
        })
        stats.kept += 1

    print(f"\n===== {source_name} dataset stats =====")
    print(f"total={stats.total}")
    print(f"kept={stats.kept}")
    print(f"skipped_empty={stats.skipped_empty}")
    print(f"skipped_no_assistant={stats.skipped_no_assistant}")
    print(f"skipped_bad_format={stats.skipped_bad_format}")
    print("===================================\n")

    if not rows:
        raise ValueError(f"No valid samples were produced from {source_name}")

    return Dataset.from_list(rows)


def preview_dataset(dataset: Dataset, n: int = 2) -> None:
    print("===== Dataset preview =====")
    for i in range(min(n, len(dataset))):
        row = dataset[i]
        prompt_preview = row["prompt"][:400].replace("\n", "\\n")
        completion_preview = row["completion"][:200].replace("\n", "\\n")
        print(f"[sample {i}] prompt: {prompt_preview}")
        print(f"[sample {i}] completion: {completion_preview}")
        print("---")
    print("===========================\n")


def main() -> None:
    check_file(TRAIN_FILE, "TRAIN_FILE")
    check_file(VAL_FILE, "VAL_FILE")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("/data/projects/shuke/code/singal_cell_annotation/data/meta", exist_ok=True)

    set_seed(SEED)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        local_files_only=LOCAL_FILES_ONLY,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_records = load_jsonl(TRAIN_FILE)
    val_records = load_jsonl(VAL_FILE)

    train_dataset = convert_messages_to_prompt_completion(train_records, tokenizer, "train")
    val_dataset = convert_messages_to_prompt_completion(val_records, tokenizer, "val")
    preview_dataset(train_dataset)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        trust_remote_code=TRUST_REMOTE_CODE,
        local_files_only=LOCAL_FILES_ONLY,
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        bf16=USE_BF16,
        fp16=USE_FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        max_length=MAX_LENGTH,
        packing=False,
        report_to=[],
        seed=SEED,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    train_result = trainer.train()

    print("Saving final adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Training finished.")
    print(f"Final output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
