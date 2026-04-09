from __future__ import annotations

"""
06_split_and_validate_v2.py

相较于原始 06_split_and_validate.py，本脚本做了以下改进：

1. 继续保持 dataset-level split，避免同一 dataset 同时出现在 train 和 test，防止数据泄漏。
2. 增加 dataset profile（数据集级画像）统计：
   - main_tissue_general（主组织）
   - main_disease（主疾病状态）
   - n_records（该 dataset 产生的 marker/SFT 样本数）
   - n_cell_types（该 dataset 中不同 cell type 数）
3. 将原来的“全局随机 dataset 切分”改成“按主 tissue_general 分层切分”，使 train/val/test 的组织分布更平衡。
4. 对小数据集数量场景做特殊处理：
   - 1 个 dataset：全部 train
   - 2 个 dataset：1 train + 1 test，val 为空
   - 3~small_dataset_upper_bound 个 dataset：优先保证 train/test，必要时从 train records 中再切 pseudo-val
   - 更大规模 dataset：正常 train/val/test 分层切分
5. 新增 pseudo-val：
   - 当独立 val dataset 不足或为空时，可从 train records 中抽取少量样本作为 val_pseudo
   - 用于训练过程中的基本监控，但它不如独立 dataset-level val 严格
6. 新增 hard test 导出：
   - 从 test_full 中额外筛出更难的样本，生成 test_hard
   - 当前规则由 config 控制，例如：
     * confidence != high
     * n_cells 较小
     * cell type 稀有
7. 额外导出 dataset_profiles.csv，方便你检查每个 dataset 的分布与切分结果。
8. 所有和 v2 划分策略相关的关键超参数，均从 data_prep_config.py 读取，便于后续调参。

说明：
- 本脚本仍兼容你当前的数据格式：
  输入：cfg.SFT_RECORDS_FULL_JSONL
  输出：train/val/test 的 full / messages / messages_no_think
- 如果你的 dataset 仍然很少，那么 split 的波动会比较大，这是正常现象。
"""

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

from transformers import AutoTokenizer

import data_prep_config as cfg

try:
    from sca.data.split_grouping import resolve_group_key, get_unique_groups
    _GROUPING_AVAILABLE = True
except ImportError:
    _GROUPING_AVAILABLE = False

from sca.data.split_builder import (
    build_test_rare_subset,
    build_test_unmapped_subset,
    build_benchmark_manifest,
    write_benchmark_manifest,
    extract_no_think_messages,
    build_v2_dataset_profiles,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =========================
# 基础读取与写出函数
# =========================

def load_full_records(path: Path) -> List[dict]:
    """
    读取 05_make_sft_jsonl.py 生成的 full records。
    每行一个 JSON，包含 dataset_id / tissue_general / cell_type_clean / messages 等字段。
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, rows: List[dict]) -> None:
    """
    将列表写成 jsonl 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    写 CSV 文件。这里使用标准库 csv，避免额外依赖。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 一些小工具函数
# =========================

def dominant_value(values: List[str], default: str = "unknown") -> str:
    """
    取一个列表中出现频率最高的值。
    用于给 dataset 指定主 tissue / 主 disease。
    """
    clean = [str(v) for v in values if str(v).strip()]
    if not clean:
        return default
    return Counter(clean).most_common(1)[0][0]


def validate_no_overlap(train_ids: set, val_ids: set, test_ids: set) -> None:
    """
    检查 train/val/test 的 dataset_id 是否有交叉。
    """
    assert train_ids.isdisjoint(val_ids), "train/val overlap detected"
    assert train_ids.isdisjoint(test_ids), "train/test overlap detected"
    assert val_ids.isdisjoint(test_ids), "val/test overlap detected"


def validate_messages_schema(records: List[dict], key: str = "messages") -> None:
    """
    检查 messages 格式是否合法。
    """
    for i, rec in enumerate(records):
        msgs = rec[key]
        assert isinstance(msgs, list), f"{i}: {key} is not a list"
        assert len(msgs) >= 2, f"{i}: too few messages"
        for msg in msgs:
            assert "role" in msg and "content" in msg, f"{i}: malformed message"


def validate_token_lengths(records: List[dict], model_name_or_path: str, key: str = "messages") -> None:
    """
    可选的 token 长度校验。
    主要用于了解样本长度分布，避免 max_length 设置不合理。
    """
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
    logging.info(
        "Token length stats for %s | p50=%d p95=%d p99=%d max=%d",
        key, p50, p95, p99, max(lengths)
    )


# =========================
# dataset profile 构建
# =========================

def build_dataset_profiles(records: List[dict]) -> List[dict]:
    """
    将 record 级样本聚合成 dataset 级画像。

    每个 dataset profile 包含：
    - dataset_id
    - dataset_title
    - main_tissue_general
    - main_disease
    - n_records
    - n_cell_types
    """
    dataset_to_records = defaultdict(list)
    for rec in records:
        dataset_to_records[rec["dataset_id"]].append(rec)

    profiles = []
    for dataset_id, recs in dataset_to_records.items():
        tissues = [rec.get("tissue_general", "unknown") for rec in recs]
        diseases = [rec.get("disease", "unknown") for rec in recs]
        cell_types = [rec.get("cell_type_clean", "unknown") for rec in recs]

        profile = {
            "dataset_id": dataset_id,
            "dataset_title": recs[0].get("dataset_title", ""),
            "main_tissue_general": dominant_value(tissues, default="unknown"),
            "main_disease": dominant_value(diseases, default="unknown"),
            "n_records": len(recs),
            "n_cell_types": len(set(cell_types)),
        }
        profiles.append(profile)

    profiles = sorted(
        profiles,
        key=lambda x: (x["main_tissue_general"], -x["n_records"], -x["n_cell_types"], x["dataset_id"])
    )
    return profiles


# =========================
# dataset-level 切分逻辑
# =========================

def simple_random_split_dataset_ids(
    profiles: List[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    """
    简单的 dataset_id 全局随机切分。
    仅在 config 中关闭主 tissue 分层时使用。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    rng = random.Random(seed)
    ids = [p["dataset_id"] for p in profiles]
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return set(ids), set(), set()
    if n == 2:
        return {ids[0]}, set(), {ids[1]}

    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_test - n_val

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


def stratified_split_dataset_ids_by_main_tissue(
    profiles: List[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    """
    依据 dataset profile 的 main_tissue_general 做 dataset-level 分层切分。

    注意：
    - 这里分层的单位是 dataset，不是 record。
    - 这样可以同时兼顾：
      1) 避免 dataset 泄漏
      2) 让不同组织在 train/val/test 中尽量更平衡
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    rng = random.Random(seed)
    all_ids = [p["dataset_id"] for p in profiles]
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

    # 小数据集模式：优先保证 train/test，val 不强求严格独立
    if 3 <= n <= cfg.SPLIT_V2_SMALL_DATASET_UPPER_BOUND:
        by_tissue = defaultdict(list)
        for p in profiles:
            by_tissue[p["main_tissue_general"]].append(p["dataset_id"])

        train_ids, val_ids, test_ids = set(), set(), set()

        for tissue, ids in by_tissue.items():
            ids = ids[:]
            rng.shuffle(ids)

            if len(ids) == 1:
                train_ids.add(ids[0])
            elif len(ids) == 2:
                train_ids.add(ids[0])
                test_ids.add(ids[1])
            else:
                # 至少给 train 和 test 各 1 个
                train_ids.add(ids[0])
                test_ids.add(ids[1])

                # 若还有更多 dataset，再考虑分一点给 val 或补充 train
                if len(ids) >= 4:
                    val_ids.add(ids[2])
                    for x in ids[3:]:
                        train_ids.add(x)
                else:
                    for x in ids[2:]:
                        train_ids.add(x)

        # 兜底：如果 test 为空，则从 train 中挪一个
        if not test_ids and train_ids:
            moved = sorted(train_ids)[-1]
            train_ids.remove(moved)
            test_ids.add(moved)

        # 兜底：确保 train 非空
        if not train_ids and test_ids:
            moved = sorted(test_ids)[0]
            test_ids.remove(moved)
            train_ids.add(moved)

        return train_ids, val_ids, test_ids

    # 常规分层模式
    by_tissue = defaultdict(list)
    for p in profiles:
        by_tissue[p["main_tissue_general"]].append(p["dataset_id"])

    train_ids, val_ids, test_ids = set(), set(), set()

    for tissue, ids in by_tissue.items():
        ids = ids[:]
        rng.shuffle(ids)

        m = len(ids)
        n_test = max(1, round(m * test_ratio))
        n_val = max(1, round(m * val_ratio))
        n_train = m - n_test - n_val

        if n_train < 1:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_test = max(1, n_test - 1)

        tissue_train = ids[:n_train]
        tissue_val = ids[n_train:n_train + n_val]
        tissue_test = ids[n_train + n_val:]

        train_ids.update(tissue_train)
        val_ids.update(tissue_val)
        test_ids.update(tissue_test)

    all_set = set(all_ids)

    if not train_ids and all_set:
        train_ids.add(sorted(all_set)[0])

    if not test_ids and len(all_set - train_ids) > 0:
        test_ids.add(sorted(all_set - train_ids)[0])

    remaining = all_set - train_ids - test_ids
    if not val_ids and len(remaining) > 0:
        val_ids.add(sorted(remaining)[0])

    # 兜底去重
    val_ids -= train_ids
    test_ids -= train_ids
    test_ids -= val_ids

    return train_ids, val_ids, test_ids


# =========================
# pseudo-val 构造
# =========================

def maybe_build_pseudo_val_from_train(
    train_full: List[dict],
    seed: int,
    pseudo_val_ratio: float,
    min_pseudo_val_examples: int,
    max_pseudo_val_examples: int,
    min_train_examples_to_enable: int,
) -> Tuple[List[dict], List[dict], bool]:
    """
    当独立 val dataset 不足时，可从 train_full 中抽取少量 record 形成 pseudo-val。

    注意：
    - 这不是严格的 dataset-level validation，因为它仍来自 train datasets
    - 但在 dataset 很少时，它至少可以帮助你观察训练过程中的损失变化
    """
    if not train_full:
        return train_full, [], False

    n = len(train_full)
    n_pseudo = max(min_pseudo_val_examples, round(n * pseudo_val_ratio))
    n_pseudo = min(n_pseudo, max_pseudo_val_examples)

    # 如果 train 本身太小，不再强行切 pseudo-val
    if n <= max(min_train_examples_to_enable, n_pseudo + 2):
        return train_full, [], False

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    pseudo_idx = set(indices[:n_pseudo])

    new_train = []
    pseudo_val = []
    for i, rec in enumerate(train_full):
        if i in pseudo_idx:
            pseudo_val.append(rec)
        else:
            new_train.append(rec)

    return new_train, pseudo_val, True


# =========================
# hard test 构造
# =========================

def build_test_hard_subset(test_full: List[dict]) -> List[dict]:
    """
    从 test_full 中筛出更难的样本。

    规则由 config 控制：
    1) confidence != high
    2) n_cells 小于阈值
    3) 在 test 中出现次数较少的稀有 cell type
    """
    if not test_full:
        return []

    cell_type_counter = Counter(rec.get("cell_type_clean", "unknown") for rec in test_full)

    hard_rows = []
    for rec in test_full:
        conf = str(rec.get("confidence", "unknown")).strip().lower()
        n_cells = int(rec.get("n_cells", 0))
        cell_type = rec.get("cell_type_clean", "unknown")

        is_low_conf = (
            cfg.SPLIT_V2_HARD_TEST_USE_NON_HIGH_CONFIDENCE and conf != "high"
        )
        is_small_group = n_cells < cfg.SPLIT_V2_HARD_TEST_MIN_N_CELLS_THRESHOLD
        is_rare_cell_type = (
            cell_type_counter[cell_type] <= cfg.SPLIT_V2_HARD_TEST_RARE_CELLTYPE_MAX_COUNT
        )

        if is_low_conf or is_small_group or is_rare_cell_type:
            hard_rows.append(rec)

    return hard_rows


# =========================
# 输出包装
# =========================

def convert_full_to_msg_and_no(full_records: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    将 full records 转成：
    - messages
    - messages_no_think
    """
    msg = [{"messages": rec["messages"]} for rec in full_records]
    no = [{"messages": rec["messages_no_think"]} for rec in full_records]
    return msg, no


def save_summary(
    split_dir: Path,
    train_ids: set,
    val_ids: set,
    test_ids: set,
    train_full: List[dict],
    val_full: List[dict],
    test_full: List[dict],
    val_mode: str,
    test_hard_full: List[dict],
    dataset_profiles_path: Path,
) -> None:
    """
    保存更详细的运行摘要。
    """
    summary_path = split_dir / "06_split_and_validate_v2_run_summary.txt"
    lines = [
        "=== 06_split_and_validate_v2 run summary ===",
        "",
        "[Split sizes]",
        f"Train examples: {len(train_full)}",
        f"Val examples: {len(val_full)}",
        f"Test examples: {len(test_full)}",
        f"Test hard examples: {len(test_hard_full)}",
        "",
        "[Dataset allocation]",
        f"Train datasets ({len(train_ids)}): {sorted(train_ids)}",
        f"Val datasets ({len(val_ids)}): {sorted(val_ids)}",
        f"Test datasets ({len(test_ids)}): {sorted(test_ids)}",
        "",
        "[Validation mode]",
        f"Val mode: {val_mode}",
        "",
        "[Artifacts]",
        f"Dataset profiles: {dataset_profiles_path}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved run summary to %s", summary_path)


# =========================
# Group-level split support (Phase-A)
# =========================

def assign_groups_to_records(records: List[dict]) -> List[dict]:
    """
    Assign a group_id to each record based on collection_doi → collection_name → dataset_id.
    This group_id is used for group-level split to prevent leakage.
    """
    primary_key = getattr(cfg, "SPLIT_GROUP_KEY", "collection_doi")
    fallback_keys = getattr(cfg, "SPLIT_GROUP_FALLBACK_KEYS", ["collection_name", "dataset_id"])

    if _GROUPING_AVAILABLE:
        for rec in records:
            rec["_group_id"] = resolve_group_key(rec, primary_key, fallback_keys)
    else:
        # fallback: use dataset_id
        for rec in records:
            rec["_group_id"] = str(rec.get("dataset_id", "unknown"))
    return records


# =========================
# 主流程
# =========================

def main(
    check_tokens: bool = False,
    model_name: str = cfg.SPLIT_TOKEN_CHECK_MODEL,
) -> None:
    split_dir = cfg.SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 full records
    records = load_full_records(Path(cfg.SFT_RECORDS_FULL_JSONL))
    logging.info("Loaded %d full SFT records", len(records))

    if not records:
        raise ValueError(f"No records found in {cfg.SFT_RECORDS_FULL_JSONL}")

    # Phase-A: assign group IDs for study-level split
    records = assign_groups_to_records(records)
    group_ids = sorted(set(r["_group_id"] for r in records))
    logging.info("Group-level split: found %d unique groups (vs %d records)", len(group_ids), len(records))

    # 2) 构建 dataset profiles
    dataset_profiles = build_dataset_profiles(records)
    dataset_profiles_path = split_dir / "dataset_profiles.csv"
    write_csv(dataset_profiles_path, dataset_profiles)
    logging.info("Saved dataset profiles to %s", dataset_profiles_path)
    logging.info("Total datasets=%d", len(dataset_profiles))

    # 3) 选择切分方式（基于 group_id 而非 dataset_id）
    # Build group-level profiles for split
    group_profiles = []
    group_tissue_map: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        group_tissue_map[rec["_group_id"]].append(rec.get("tissue_general", "unknown"))
    for gid in group_ids:
        tissues = group_tissue_map[gid]
        group_profiles.append({
            "dataset_id": gid,  # reuse dataset_id field for compatibility
            "main_tissue_general": dominant_value(tissues, default="unknown"),
            "n_records": len(tissues),
            "n_cell_types": 0,
        })

    if cfg.SPLIT_V2_USE_MAIN_TISSUE_STRATIFY:
        train_group_ids, val_group_ids, test_group_ids = stratified_split_dataset_ids_by_main_tissue(
            profiles=group_profiles,
            train_ratio=cfg.SPLIT_TRAIN_RATIO,
            val_ratio=cfg.SPLIT_VAL_RATIO,
            test_ratio=cfg.SPLIT_TEST_RATIO,
            seed=cfg.RANDOM_SEED,
        )
        logging.info("Using main_tissue_general stratified group split.")
    else:
        train_group_ids, val_group_ids, test_group_ids = simple_random_split_dataset_ids(
            profiles=group_profiles,
            train_ratio=cfg.SPLIT_TRAIN_RATIO,
            val_ratio=cfg.SPLIT_VAL_RATIO,
            test_ratio=cfg.SPLIT_TEST_RATIO,
            seed=cfg.RANDOM_SEED,
        )
        logging.info("Using simple random group split.")

    validate_no_overlap(train_group_ids, val_group_ids, test_group_ids)

    # Also compute dataset_id-level sets for downstream v2 compatibility
    train_ids = set(rec["dataset_id"] for rec in records if rec["_group_id"] in train_group_ids)
    val_ids = set(rec["dataset_id"] for rec in records if rec["_group_id"] in val_group_ids)
    test_ids = set(rec["dataset_id"] for rec in records if rec["_group_id"] in test_group_ids)

    logging.info(
        "Group split: train_groups=%d | val_groups=%d | test_groups=%d",
        len(train_group_ids), len(val_group_ids), len(test_group_ids),
    )
    logging.info(
        "Dataset split (derived): train_datasets=%d | val_datasets=%d | test_datasets=%d",
        len(train_ids), len(val_ids), len(test_ids),
    )

    # 4) 将 full records 按 _group_id 分发到 train/val/test
    train_full, val_full, test_full = [], [], []

    for rec in records:
        did = rec["dataset_id"]
        gid = rec["_group_id"]

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

        if gid in train_group_ids:
            train_full.append(full_out)
        elif gid in val_group_ids:
            val_full.append(full_out)
        elif gid in test_group_ids:
            test_full.append(full_out)

    # 5) 若 val 为空，则按配置决定是否构造 pseudo-val
    if len(val_full) == 0:
        if cfg.SPLIT_V2_ENABLE_PSEUDO_VAL:
            new_train_full, pseudo_val_full, used_pseudo = maybe_build_pseudo_val_from_train(
                train_full=train_full,
                seed=cfg.RANDOM_SEED,
                pseudo_val_ratio=cfg.SPLIT_V2_PSEUDO_VAL_RATIO,
                min_pseudo_val_examples=cfg.SPLIT_V2_MIN_PSEUDO_VAL_EXAMPLES,
                max_pseudo_val_examples=cfg.SPLIT_V2_MAX_PSEUDO_VAL_EXAMPLES,
                min_train_examples_to_enable=cfg.SPLIT_V2_MIN_TRAIN_EXAMPLES_TO_ENABLE_PSEUDO_VAL,
            )
        else:
            new_train_full, pseudo_val_full, used_pseudo = train_full, [], False

        if used_pseudo:
            logging.warning("No independent val dataset. Built pseudo-val from train records.")
            train_full = new_train_full
            val_full = pseudo_val_full
            val_mode = "pseudo_val_from_train_records"
        else:
            val_mode = "empty_val"
    else:
        val_mode = "independent_dataset_val"

    # 6) 转成 messages / messages_no_think
    train_msg, train_no = convert_full_to_msg_and_no(train_full)
    val_msg, val_no = convert_full_to_msg_and_no(val_full)
    test_msg, test_no = convert_full_to_msg_and_no(test_full)

    # 7) 按配置构造 hard test
    if cfg.SPLIT_V2_EXPORT_HARD_TEST:
        test_hard_full = build_test_hard_subset(test_full)
    else:
        test_hard_full = []

    test_hard_msg, test_hard_no = convert_full_to_msg_and_no(test_hard_full)

    # 8) schema 检查
    validate_messages_schema(train_msg, key="messages")
    if val_msg:
        validate_messages_schema(val_msg, key="messages")
    if test_msg:
        validate_messages_schema(test_msg, key="messages")

    validate_messages_schema(train_no, key="messages")
    if val_no:
        validate_messages_schema(val_no, key="messages")
    if test_no:
        validate_messages_schema(test_no, key="messages")

    # 9) 写出 split 文件
    write_jsonl(split_dir / "train_full.jsonl", train_full)
    write_jsonl(split_dir / "val_full.jsonl", val_full)
    write_jsonl(split_dir / "test_full.jsonl", test_full)

    write_jsonl(split_dir / "train_messages.jsonl", train_msg)
    write_jsonl(split_dir / "val_messages.jsonl", val_msg)
    write_jsonl(split_dir / "test_messages.jsonl", test_msg)

    write_jsonl(split_dir / "train_messages_no_think.jsonl", train_no)
    write_jsonl(split_dir / "val_messages_no_think.jsonl", val_no)
    write_jsonl(split_dir / "test_messages_no_think.jsonl", test_no)

    # 新增 hard test 导出
    write_jsonl(split_dir / "test_hard_full.jsonl", test_hard_full)
    write_jsonl(split_dir / "test_hard_messages.jsonl", test_hard_msg)
    write_jsonl(split_dir / "test_hard_messages_no_think.jsonl", test_hard_no)

    logging.info("Split done.")
    logging.info(
        "Train=%d | Val=%d | Test=%d | TestHard=%d",
        len(train_msg), len(val_msg), len(test_msg), len(test_hard_msg)
    )
    logging.info(
        "Train datasets=%d | Val datasets=%d | Test datasets=%d",
        len(train_ids), len(val_ids), len(test_ids)
    )
    logging.info("Val mode=%s", val_mode)

    # 10) 可选 token 长度检查
    if check_tokens:
        validate_token_lengths(train_msg, model_name_or_path=model_name, key="messages")
        validate_token_lengths(train_no, model_name_or_path=model_name, key="messages")

        if val_msg:
            validate_token_lengths(val_msg, model_name_or_path=model_name, key="messages")
            validate_token_lengths(val_no, model_name_or_path=model_name, key="messages")

        if test_msg:
            validate_token_lengths(test_msg, model_name_or_path=model_name, key="messages")
            validate_token_lengths(test_no, model_name_or_path=model_name, key="messages")

    # 11) Phase 2 — v2 benchmark splits
    v2_records_path = Path(cfg.SFT_RECORDS_FULL_V2_JSONL)
    if not v2_records_path.exists():
        logging.warning(
            "v2 records not found at %s; skipping v2 benchmark split.", v2_records_path
        )
    else:
        v2_records = load_full_records(v2_records_path)
        logging.info("Loaded %d v2 SFT records", len(v2_records))

        # Partition v2 records using the same dataset IDs as v1 split
        train_v2: List[dict] = []
        val_v2: List[dict] = []
        test_v2: List[dict] = []
        for rec in v2_records:
            did = rec.get("dataset_id", "")
            if did in train_ids:
                train_v2.append(rec)
            elif did in val_ids:
                val_v2.append(rec)
            elif did in test_ids:
                test_v2.append(rec)

        logging.info(
            "v2 split: train=%d | val=%d | test=%d",
            len(train_v2), len(val_v2), len(test_v2),
        )

        # Write v2 full splits
        write_jsonl(split_dir / "train_full_v2.jsonl", train_v2)
        write_jsonl(split_dir / "val_full_v2.jsonl", val_v2)
        write_jsonl(split_dir / "test_full_v2.jsonl", test_v2)

        # Write v2 message-only splits
        train_v2_msg, train_v2_no = convert_full_to_msg_and_no(train_v2)
        val_v2_msg, val_v2_no = convert_full_to_msg_and_no(val_v2)
        test_v2_msg, test_v2_no = convert_full_to_msg_and_no(test_v2)

        write_jsonl(split_dir / "train_messages_v2.jsonl", train_v2_msg)
        write_jsonl(split_dir / "val_messages_v2.jsonl", val_v2_msg)
        write_jsonl(split_dir / "test_messages_v2.jsonl", test_v2_msg)

        write_jsonl(split_dir / "train_messages_no_think_v2.jsonl", train_v2_no)
        write_jsonl(split_dir / "val_messages_no_think_v2.jsonl", val_v2_no)
        write_jsonl(split_dir / "test_messages_no_think_v2.jsonl", test_v2_no)

        # Benchmark subsets: globally-rare and ontology-unmapped
        all_v2_records = train_v2 + val_v2 + test_v2
        test_rare_v2 = build_test_rare_subset(
            test_v2,
            global_records=all_v2_records,
            rare_max_global_count=cfg.SPLIT_V2_RARE_MAX_GLOBAL_COUNT,
        )
        test_unmapped_v2 = build_test_unmapped_subset(test_v2)

        write_jsonl(split_dir / "test_rare_full.jsonl", test_rare_v2)
        write_jsonl(split_dir / "test_unmapped_full.jsonl", test_unmapped_v2)

        write_jsonl(
            split_dir / "test_rare_messages_no_think.jsonl",
            extract_no_think_messages(test_rare_v2),
        )
        write_jsonl(
            split_dir / "test_unmapped_messages_no_think.jsonl",
            extract_no_think_messages(test_unmapped_v2),
        )

        logging.info(
            "Benchmark splits: test_rare=%d | test_unmapped=%d",
            len(test_rare_v2), len(test_unmapped_v2),
        )

        # Benchmark manifest CSV
        def _n_datasets(recs: List[dict]) -> int:
            return len({r.get("dataset_id", "") for r in recs})

        split_sizes_v2 = {
            "train_v2": len(train_v2),
            "val_v2": len(val_v2),
            "test_v2": len(test_v2),
            "test_rare": len(test_rare_v2),
            "test_unmapped": len(test_unmapped_v2),
        }
        split_dataset_counts_v2 = {
            "train_v2": _n_datasets(train_v2),
            "val_v2": _n_datasets(val_v2),
            "test_v2": _n_datasets(test_v2),
            "test_rare": _n_datasets(test_rare_v2),
            "test_unmapped": _n_datasets(test_unmapped_v2),
        }

        manifest_rows = build_benchmark_manifest(split_sizes_v2, split_dataset_counts_v2)
        write_benchmark_manifest(manifest_rows, split_dir / "benchmark_manifest.csv")

        # v2 dataset profiles (extended with ontology stats)
        v2_profiles = build_v2_dataset_profiles(all_v2_records)
        write_csv(split_dir / "dataset_profiles_v2.csv", v2_profiles)
        logging.info("Saved v2 dataset profiles: %d datasets", len(v2_profiles))

    # 12) Phase 3 — v3 splits (ontology-aligned, canonical labels, no LLM distillation)
    v3_records_path = Path(cfg.SFT_RECORDS_FULL_V3_JSONL)
    if not v3_records_path.exists():
        logging.warning(
            "v3 records not found at %s; skipping v3 split.", v3_records_path
        )
    else:
        v3_records = load_full_records(v3_records_path)
        logging.info("Loaded %d v3 SFT records", len(v3_records))

        # Partition v3 records using the same dataset IDs as v1 split
        train_v3: List[dict] = []
        val_v3: List[dict] = []
        test_v3: List[dict] = []
        for rec in v3_records:
            did = rec.get("dataset_id", "")
            if did in train_ids:
                train_v3.append(rec)
            elif did in val_ids:
                val_v3.append(rec)
            elif did in test_ids:
                test_v3.append(rec)

        logging.info(
            "v3 split: train=%d | val=%d | test=%d",
            len(train_v3), len(val_v3), len(test_v3),
        )

        train_v3_msg, train_v3_no = convert_full_to_msg_and_no(train_v3)
        val_v3_msg, val_v3_no = convert_full_to_msg_and_no(val_v3)
        test_v3_msg, test_v3_no = convert_full_to_msg_and_no(test_v3)

        write_jsonl(split_dir / "train_full_v3.jsonl", train_v3)
        write_jsonl(split_dir / "val_full_v3.jsonl", val_v3)
        write_jsonl(split_dir / "test_full_v3.jsonl", test_v3)

        write_jsonl(split_dir / "train_messages_no_think_v3.jsonl", train_v3_no)
        write_jsonl(split_dir / "val_messages_no_think_v3.jsonl", val_v3_no)
        write_jsonl(split_dir / "test_messages_no_think_v3.jsonl", test_v3_no)

        # Count valid CL IDs in train for quick sanity check
        n_with_cl = sum(
            1 for r in train_v3
            if (r.get("cell_ontology_id") or "").startswith("CL:")
        )
        logging.info(
            "v3 train ontology_id coverage: %d/%d (%.1f%%)",
            n_with_cl, len(train_v3),
            n_with_cl / len(train_v3) * 100 if train_v3 else 0,
        )

    # 13) 保存 summary
    save_summary(
        split_dir=split_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_full=train_full,
        val_full=val_full,
        test_full=test_full,
        val_mode=val_mode,
        test_hard_full=test_hard_full,
        dataset_profiles_path=dataset_profiles_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-tokens", action="store_true")
    parser.add_argument("--model-name", type=str, default=cfg.SPLIT_TOKEN_CHECK_MODEL)
    args = parser.parse_args()

    main(check_tokens=args.check_tokens, model_name=args.model_name)

    # 示例运行：
    # cd /data/projects/shuke/code/singal_cell_annotation
    # nohup python -u scripts/data_prep/06_split_and_validate_v2.py 2>&1 | tee data/meta/06_split_and_validate_v2.log

    # nohup bash -lc 'export PYTHONPATH="$PWD/src:$PYTHONPATH"; python -u scripts/data_prep/06_split_and_validate_v2.py >> data/meta/06_split_and_validate_v2.log 2>&1' >/dev/null 2>&1 &
    # grep -E "Generated|Building markers|Make markers:" data/meta/06_split_and_validate_v2.log | tail -10