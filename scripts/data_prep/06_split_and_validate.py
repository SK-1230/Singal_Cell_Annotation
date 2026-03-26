from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

import config as cfg

from modelscope import snapshot_download


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_full_records(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# def split_dataset_ids(
#     records: List[dict],
#     train_ratio: float = 0.7,
#     val_ratio: float = 0.15,
#     test_ratio: float = 0.15,
#     seed: int = 42,
# ) -> Tuple[set, set, set]:
#     assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

#     rng = random.Random(seed)
#     all_ids = sorted({rec["dataset_id"] for rec in records})

#     # 当前阶段：dataset 很少时采用更实际的策略
#     if len(all_ids) == 1:
#         logging.warning("Only 1 dataset detected; assigning all to train.")
#         return set(all_ids), set(), set()

#     if len(all_ids) == 2:
#         ids = all_ids[:]
#         rng.shuffle(ids)
#         logging.warning("Only 2 datasets detected; assigning 1 to train and 1 to test, val will be empty.")
#         return {ids[0]}, set(), {ids[1]}

#     # 数据集足够多时，再按 tissue 分层
#     tissue_to_ids = defaultdict(set)
#     for rec in records:
#         tissue_to_ids[rec.get("tissue_general", "unknown")].add(rec["dataset_id"])

#     train_ids, val_ids, test_ids = set(), set(), set()

#     for tissue, ids in tissue_to_ids.items():
#         ids = sorted(ids)
#         rng.shuffle(ids)

#         n = len(ids)
#         if n < 3:
#             logging.warning(
#                 "Tissue=%s has only %d dataset(s); assigning all to train for this tissue group.",
#                 tissue, n
#             )
#             train_ids.update(ids)
#             continue

#         n_test = max(1, round(n * test_ratio))
#         n_val = max(1, round(n * val_ratio))
#         n_train = n - n_test - n_val

#         if n_train < 1:
#             n_train = 1
#             if n_val > 1:
#                 n_val -= 1
#             else:
#                 n_test -= 1

#         train_ids.update(ids[:n_train])
#         val_ids.update(ids[n_train:n_train + n_val])
#         test_ids.update(ids[n_train + n_val:])

#     # 兜底：若某些 dataset 没被分到任何集合，补到 train
#     assigned = train_ids | val_ids | test_ids
#     missing = set(all_ids) - assigned
#     if missing:
#         logging.warning("Found %d unassigned dataset(s); moving them to train: %s", len(missing), sorted(missing))
#         train_ids.update(missing)

#     return train_ids, val_ids, test_ids
def split_dataset_ids(
    records: List[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[set, set, set]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    rng = random.Random(seed)
    all_ids = sorted({rec["dataset_id"] for rec in records})
    n = len(all_ids)

    if n == 0:
        return set(), set(), set()

    if n == 1:
        logging.warning("Only 1 dataset detected; assigning all to train.")
        return set(all_ids), set(), set()

    if n == 2:
        ids = all_ids[:]
        rng.shuffle(ids)
        logging.warning("Only 2 datasets detected; assigning 1 to train and 1 to test, val will be empty.")
        return {ids[0]}, set(), {ids[1]}

    # 对于 3 个及以上 dataset，直接按 dataset_id 全局划分
    ids = all_ids[:]
    rng.shuffle(ids)

    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_test - n_val

    # 兜底，确保 train 至少有 1 个
    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_no_overlap(train_ids: set, val_ids: set, test_ids: set) -> None:
    assert train_ids.isdisjoint(val_ids), "train/val overlap detected"
    assert train_ids.isdisjoint(test_ids), "train/test overlap detected"
    assert val_ids.isdisjoint(test_ids), "val/test overlap detected"


def validate_messages_schema(records: List[dict], key: str = "messages") -> None:
    for i, rec in enumerate(records):
        msgs = rec[key]
        assert isinstance(msgs, list), f"{i}: {key} is not a list"
        assert len(msgs) >= 2, f"{i}: too few messages"
        for msg in msgs:
            assert "role" in msg and "content" in msg, f"{i}: malformed message"


def validate_token_lengths(records: List[dict], model_name_or_path: str, key: str = "messages") -> None:
    if not records:
        logging.warning("Skip token length validation for %s: no records.", key)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    lengths = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec[key],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(input_ids))

    lengths = sorted(lengths)
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))]
    p99 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.99))]
    logging.info("Token length stats for %s | p50=%d p95=%d p99=%d max=%d", key, p50, p95, p99, max(lengths))


def save_summary(
    split_dir: Path,
    train_ids: set,
    val_ids: set,
    test_ids: set,
    train_msg: List[dict],
    val_msg: List[dict],
    test_msg: List[dict],
) -> None:
    summary_path = split_dir / "06_split_and_validate_run_summary.txt"
    lines = [
        "=== 06_split_and_validate run summary ===",
        "",
        f"Train examples: {len(train_msg)}",
        f"Val examples: {len(val_msg)}",
        f"Test examples: {len(test_msg)}",
        "",
        f"Train datasets: {len(train_ids)} -> {sorted(train_ids)}",
        f"Val datasets: {len(val_ids)} -> {sorted(val_ids)}",
        f"Test datasets: {len(test_ids)} -> {sorted(test_ids)}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved run summary to %s", summary_path)


def main(check_tokens: bool = False, model_name: str = "Qwen/Qwen3-8B") -> None:
    split_dir = cfg.SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)

    records = load_full_records(Path(cfg.SFT_RECORDS_FULL_JSONL))
    logging.info("Loaded %d full SFT records", len(records))

    train_ids, val_ids, test_ids = split_dataset_ids(records, seed=cfg.RANDOM_SEED)
    validate_no_overlap(train_ids, val_ids, test_ids)

    train_full, val_full, test_full = [], [], []
    train_msg, val_msg, test_msg = [], [], []
    train_no, val_no, test_no = [], [], []

    for rec in records:
        did = rec["dataset_id"]

        full_out = {
            "dataset_id": rec["dataset_id"],
            "dataset_title": rec["dataset_title"],
            "organism": rec["organism"],
            "tissue_general": rec["tissue_general"],
            "tissue": rec["tissue"],
            "disease": rec["disease"],
            "cell_type_clean": rec["cell_type_clean"],
            "cell_type_gold_major": rec["cell_type_gold_major"],
            "n_cells": rec["n_cells"],
            "marker_genes": rec["marker_genes"],
            "confidence": rec.get("confidence", "unknown"),
            "messages": rec["messages"],
            "messages_no_think": rec["messages_no_think"],
        }

        msg_out = {"messages": rec["messages"]}
        no_out = {"messages": rec["messages_no_think"]}

        if did in train_ids:
            train_full.append(full_out)
            train_msg.append(msg_out)
            train_no.append(no_out)
        elif did in val_ids:
            val_full.append(full_out)
            val_msg.append(msg_out)
            val_no.append(no_out)
        elif did in test_ids:
            test_full.append(full_out)
            test_msg.append(msg_out)
            test_no.append(no_out)

    validate_messages_schema(train_msg, key="messages")
    validate_messages_schema(val_msg, key="messages")
    validate_messages_schema(test_msg, key="messages")
    validate_messages_schema(train_no, key="messages")
    validate_messages_schema(val_no, key="messages")
    validate_messages_schema(test_no, key="messages")

    write_jsonl(split_dir / "train_full.jsonl", train_full)
    write_jsonl(split_dir / "val_full.jsonl", val_full)
    write_jsonl(split_dir / "test_full.jsonl", test_full)

    write_jsonl(split_dir / "train_messages.jsonl", train_msg)
    write_jsonl(split_dir / "val_messages.jsonl", val_msg)
    write_jsonl(split_dir / "test_messages.jsonl", test_msg)

    write_jsonl(split_dir / "train_messages_no_think.jsonl", train_no)
    write_jsonl(split_dir / "val_messages_no_think.jsonl", val_no)
    write_jsonl(split_dir / "test_messages_no_think.jsonl", test_no)

    logging.info("Split done.")
    logging.info("Train=%d | Val=%d | Test=%d", len(train_msg), len(val_msg), len(test_msg))
    logging.info("Train datasets=%d | Val datasets=%d | Test datasets=%d", len(train_ids), len(val_ids), len(test_ids))

    if check_tokens:
        local_model_path = snapshot_download("Qwen/Qwen3-8B", cache_dir="./my_models")
        print(local_model_path)

        validate_token_lengths(train_msg, model_name_or_path=local_model_path, key="messages")
        validate_token_lengths(train_no, model_name_or_path=local_model_path, key="messages")

    save_summary(split_dir, train_ids, val_ids, test_ids, train_msg, val_msg, test_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-tokens", action="store_true")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    main(check_tokens=args.check_tokens, model_name=args.model_name)
    
    # 做法 2：改成走 ModelScope，本地加载 tokenizer
    # python -u 06_split_and_validate.py --check-tokens --model-name ./my_models/Qwen3-8B 2>&1 | tee data/meta/06_split_and_validate.log

    # 做法 1：先不做 token 校验
    # python -u 06_split_and_validate.py 2>&1 | tee data/meta/06_split_and_validate.log