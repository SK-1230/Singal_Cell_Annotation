from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Must be set before scanpy/numba is imported.
# scanpy's wilcoxon rankdata uses numba.prange; with 72+ threads and large arrays
# (n_threads ≈ n_chunk_columns), numba triggers SIGBUS. Force override to a safe value.
os.environ["NUMBA_NUM_THREADS"] = "4"

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

import data_prep_config as cfg

# Phase-A: use target label for grouping if available
_TARGET_GROUP_COL = getattr(cfg, "MARKER_TARGET_GROUP_COL", None)
if _TARGET_GROUP_COL is None:
    _TARGET_GROUP_COL = "cell_type_target_label" if getattr(cfg, "TARGET_LABEL_MODE", "clean") == "ontology_label" else cfg.MARKER_GROUP_COL


# Phase 1 v2: evidence-aware marker extraction
try:
    from sca.data.marker_extraction import (
        extract_positive_markers,
        extract_negative_markers,
        compute_marker_quality_score,
        safe_mean_topk,
    )
    from sca.data.label_normalization import normalize_and_map, init_alias_table
    _V2_AVAILABLE = True
except ImportError:
    _V2_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# 可选增强配置（若 data_prep_config.py 里没有这些字段，则使用默认值）
# =========================
MARKER_RERUN_ZERO_EXAMPLE_SUCCESS = getattr(cfg, "MARKER_RERUN_ZERO_EXAMPLE_SUCCESS", False)
MARKER_RECOVER_FROM_JSONL_IF_MANIFEST_MISSING = getattr(cfg, "MARKER_RECOVER_FROM_JSONL_IF_MANIFEST_MISSING", True)
MARKER_REMOVE_ORPHAN_JSONL_RECORDS = getattr(cfg, "MARKER_REMOVE_ORPHAN_JSONL_RECORDS", False)

# Phase 1 v2 配置
MARKER_DE_METHOD = getattr(cfg, "MARKER_DE_METHOD", "wilcoxon")
TOP_K_NEGATIVE_MARKERS = getattr(cfg, "TOP_K_NEGATIVE_MARKERS", 5)
MARKER_LOW_CELLS_THRESHOLD = getattr(cfg, "MARKER_LOW_CELLS_THRESHOLD", 100)
MARKER_LOW_QUALITY_SCORE_THRESHOLD = getattr(cfg, "MARKER_LOW_QUALITY_SCORE_THRESHOLD", 0.3)
MARKER_RARE_LABEL_FRACTION_THRESHOLD = getattr(cfg, "MARKER_RARE_LABEL_FRACTION_THRESHOLD", 0.02)


def is_bad_marker_gene(gene: str) -> bool:
    gene = str(gene).upper()
    if gene in cfg.UNINFORMATIVE_GENES:
        return True
    return any(gene.startswith(prefix) for prefix in cfg.BAD_GENE_PREFIXES)


def dominant_value(series: pd.Series, default: str = "unknown") -> str:
    series = series.dropna().astype(str)
    if len(series) == 0:
        return default
    return series.value_counts().index[0]


def downsample_per_label(adata: ad.AnnData, label_col: str) -> ad.AnnData:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    keep_obs_names: List[str] = []

    for label in sorted(adata.obs[label_col].dropna().astype(str).unique()):
        sub_obs_names = adata.obs_names[adata.obs[label_col].astype(str) == label]
        sub_obs_names = np.array(sub_obs_names, dtype=object)

        if len(sub_obs_names) > cfg.MAX_CELLS_PER_LABEL_FOR_DE:
            sub_obs_names = rng.choice(
                sub_obs_names,
                size=cfg.MAX_CELLS_PER_LABEL_FOR_DE,
                replace=False
            )

        keep_obs_names.extend(sub_obs_names.tolist())

    keep_obs_names = pd.Index(keep_obs_names).unique()
    return adata[keep_obs_names].copy()


def safe_mean_topk(series: pd.Series, k: int = 5) -> float | None:
    s = series.head(k).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return None
    val = s.mean()
    return None if pd.isna(val) else float(val)


def extract_markers_for_dataset(adata: ad.AnnData) -> List[dict]:
    records: List[dict] = []

    if cfg.MARKER_GROUP_COL not in adata.obs.columns:
        return records

    if adata.obs[cfg.MARKER_GROUP_COL].nunique() < 2:
        return records

    adata_work = adata.copy()

    # 只保留有效标签
    adata_work = adata_work[adata_work.obs[cfg.MARKER_GROUP_COL].notna()].copy()
    if adata_work.n_obs == 0:
        return records

    # 每类最多抽样一定数量细胞，防止超大类压制计算
    adata_work = downsample_per_label(adata_work, cfg.MARKER_GROUP_COL)
    if adata_work.n_obs == 0:
        return records

    # 使用 counts 层
    if "counts" in adata_work.layers:
        adata_work.X = adata_work.layers["counts"].copy()

    # marker 阶段再做一次轻量基因过滤
    sc.pp.filter_genes(adata_work, min_cells=cfg.MARKER_MIN_GENE_CELLS)
    if adata_work.n_vars == 0:
        return records

    # 归一化 + log1p
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)

    # 显式设 category，groupby 更稳
    adata_work.obs[cfg.MARKER_GROUP_COL] = adata_work.obs[cfg.MARKER_GROUP_COL].astype("category")

    # 差异分析：每个标签 vs 其余 (v1 默认 t-test 保持不变)
    sc.tl.rank_genes_groups(
        adata_work,
        groupby=cfg.MARKER_GROUP_COL,
        method="t-test",
        corr_method="benjamini-hochberg",
        n_genes=cfg.MAX_CANDIDATE_MARKERS,
        use_raw=False,
    )

    dataset_meta = adata.uns.get("source_meta", {})
    dataset_id = dataset_meta.get("dataset_id", "unknown_dataset")
    dataset_title = dataset_meta.get("dataset_title", "unknown_title")

    categories = list(adata_work.obs[cfg.MARKER_GROUP_COL].cat.categories)

    for label in categories:
        try:
            df = sc.get.rank_genes_groups_df(adata_work, group=label)
        except Exception:
            continue

        df = df[df["names"].notna()].copy()
        if df.empty:
            continue

        df["names"] = df["names"].astype(str)
        df = df[~df["names"].map(is_bad_marker_gene)]

        if "logfoldchanges" in df.columns:
            df = df[df["logfoldchanges"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf) > 0]

        top_df = df.head(cfg.TOP_K_MARKERS).copy()
        if len(top_df) < cfg.MARKER_MIN_TOP_GENES:
            continue

        label_cells = adata.obs[adata.obs[cfg.MARKER_GROUP_COL].astype(str) == str(label)]

        record = {
            "dataset_id": dataset_id,
            "dataset_title": dataset_title,
            "organism": cfg.ORGANISM,
            "tissue_general": dominant_value(label_cells["tissue_general"]) if "tissue_general" in label_cells.columns else "unknown",
            "tissue": dominant_value(label_cells["tissue"]) if "tissue" in label_cells.columns else "unknown",
            "disease": dominant_value(label_cells["disease"]) if "disease" in label_cells.columns else "unknown",
            "cell_type_clean": str(label),
            "cell_type_gold_major": dominant_value(label_cells["cell_type_gold"]) if "cell_type_gold" in label_cells.columns else str(label),
            "n_cells": int(label_cells.shape[0]),
            "marker_genes": top_df["names"].tolist(),
            "marker_logfoldchanges": [
                float(x) if pd.notna(x) and np.isfinite(x) else None
                for x in top_df["logfoldchanges"].tolist()
            ] if "logfoldchanges" in top_df.columns else [],
            "marker_pvals_adj": [
                float(x) if pd.notna(x) and np.isfinite(x) else None
                for x in top_df["pvals_adj"].tolist()
            ] if "pvals_adj" in top_df.columns else [],
            "avg_logfc_top5": safe_mean_topk(top_df["logfoldchanges"], 5) if "logfoldchanges" in top_df.columns else None,
            "avg_padj_top5": safe_mean_topk(top_df["pvals_adj"], 5) if "pvals_adj" in top_df.columns else None,
        }
        records.append(record)

    return records


def _make_record_id(dataset_id: str, label: str) -> str:
    raw = f"{dataset_id}|{label}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def extract_markers_v2_for_dataset(adata: ad.AnnData) -> List[dict]:
    """
    Phase 1 v2 marker extraction:
    - Uses wilcoxon DE method
    - Extracts positive + negative markers with full stats
    - Adds ontology mapping (if available)
    - Computes marker_quality_score and hardness_flags
    """
    if not _V2_AVAILABLE:
        logging.warning("sca.data.marker_extraction not available — falling back to v1")
        return []

    records: List[dict] = []

    # Phase-A: determine effective group column
    group_col = _TARGET_GROUP_COL if _TARGET_GROUP_COL in adata.obs.columns else cfg.MARKER_GROUP_COL

    if group_col not in adata.obs.columns:
        return records

    if adata.obs[group_col].nunique() < 2:
        return records

    adata_work = adata.copy()
    adata_work = adata_work[adata_work.obs[group_col].notna()].copy()
    if adata_work.n_obs == 0:
        return records

    adata_work = downsample_per_label(adata_work, group_col)
    if adata_work.n_obs == 0:
        return records

    if "counts" in adata_work.layers:
        adata_work.X = adata_work.layers["counts"].copy()

    sc.pp.filter_genes(adata_work, min_cells=cfg.MARKER_MIN_GENE_CELLS)
    if adata_work.n_vars == 0:
        return records

    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)

    adata_work.obs[group_col] = adata_work.obs[group_col].astype("category")

    # Phase 1 v2: use wilcoxon
    try:
        sc.tl.rank_genes_groups(
            adata_work,
            groupby=group_col,
            method=MARKER_DE_METHOD,
            corr_method="benjamini-hochberg",
            n_genes=cfg.MAX_CANDIDATE_MARKERS,
            use_raw=False,
        )
    except Exception as e:
        logging.warning("wilcoxon DE failed (%r), trying t-test fallback", e)
        sc.tl.rank_genes_groups(
            adata_work,
            groupby=group_col,
            method="t-test",
            corr_method="benjamini-hochberg",
            n_genes=cfg.MAX_CANDIDATE_MARKERS,
            use_raw=False,
        )

    dataset_meta = adata.uns.get("source_meta", {})
    dataset_id = dataset_meta.get("dataset_id", "unknown_dataset")
    dataset_title = dataset_meta.get("dataset_title", "unknown_title")

    # Initialize ontology if available
    if _V2_AVAILABLE and hasattr(cfg, "LABEL_ALIASES_TSV"):
        try:
            init_alias_table(cfg.LABEL_ALIASES_TSV)
        except Exception:
            pass

    total_cells = adata_work.n_obs
    categories = list(adata_work.obs[group_col].cat.categories)
    label_counts_in_ds = adata_work.obs[group_col].value_counts()

    for label in categories:
        try:
            de_df = sc.get.rank_genes_groups_df(adata_work, group=label)
        except Exception:
            continue

        de_df = de_df[de_df["names"].notna()].copy()
        if de_df.empty:
            continue

        # Build label mask in the working adata
        label_mask = (adata_work.obs[group_col].astype(str) == str(label)).values

        # Extract positive and negative markers
        positive_markers = extract_positive_markers(
            de_df=de_df,
            adata=adata_work,
            label_mask=label_mask,
            top_k=cfg.TOP_K_MARKERS,
            uninformative_genes=cfg.UNINFORMATIVE_GENES,
            bad_prefixes=cfg.BAD_GENE_PREFIXES,
        )

        if len(positive_markers) < cfg.MARKER_MIN_TOP_GENES:
            continue

        negative_markers = extract_negative_markers(
            de_df=de_df,
            adata=adata_work,
            label_mask=label_mask,
            top_k=TOP_K_NEGATIVE_MARKERS,
            uninformative_genes=cfg.UNINFORMATIVE_GENES,
            bad_prefixes=cfg.BAD_GENE_PREFIXES,
        )

        # Get cell-level info from original adata
        label_cells = adata.obs[adata.obs[group_col].astype(str) == str(label)]
        n_cells = int(label_cells.shape[0])

        # Ontology mapping
        ontology_info: Dict[str, Any] = {}

        if True:
            # Check if obs already has ontology cols from 03
            if "cell_ontology_id" in label_cells.columns:
                ont_id = label_cells["cell_ontology_id"].dropna()
                ontology_info["cell_ontology_id"] = str(ont_id.iloc[0]) if len(ont_id) > 0 else None
            if "cell_ontology_label" in label_cells.columns:
                ont_lbl = label_cells["cell_ontology_label"].dropna()
                ontology_info["cell_ontology_label"] = str(ont_lbl.iloc[0]) if len(ont_lbl) > 0 else None
            if "cell_ontology_parent_label" in label_cells.columns:
                parent = label_cells["cell_ontology_parent_label"].dropna()
                ontology_info["cell_ontology_parent_label"] = str(parent.iloc[0]) if len(parent) > 0 else None

            if not ontology_info and _V2_AVAILABLE:
                mapped = normalize_and_map(str(label))
                ontology_info = {
                    "cell_ontology_id": mapped.get("cell_ontology_id"),
                    "cell_ontology_label": mapped.get("cell_ontology_label"),
                    "cell_ontology_parent_label": mapped.get("cell_ontology_parent_label"),
                }

        # Marker quality score
        quality_score = compute_marker_quality_score(
            positive_markers, n_cells, MARKER_LOW_CELLS_THRESHOLD
        )

        # avg stats
        lfc_series = pd.Series([m.get("logfoldchange") for m in positive_markers]).dropna()
        padj_series = pd.Series([m.get("pvals_adj") for m in positive_markers]).dropna()
        avg_logfc_top5 = safe_mean_topk(lfc_series, 5)
        avg_padj_top5 = safe_mean_topk(padj_series, 5)

        # Hardness flags
        label_count_in_ds = int(label_counts_in_ds.get(label, 0))
        label_fraction = label_count_in_ds / total_cells if total_cells > 0 else 0.0

        hardness_flags = {
            "low_cells": n_cells < MARKER_LOW_CELLS_THRESHOLD,
            "low_marker_quality": quality_score < MARKER_LOW_QUALITY_SCORE_THRESHOLD,
            "ontology_unmapped": not bool(ontology_info.get("cell_ontology_id")),
            "rare_label_in_dataset": label_fraction < MARKER_RARE_LABEL_FRACTION_THRESHOLD,
        }

        record: Dict[str, Any] = {
            "record_id": _make_record_id(dataset_id, str(label)),
            "dataset_id": dataset_id,
            "dataset_title": dataset_title,
            "organism": cfg.ORGANISM,
            "tissue_general": dominant_value(label_cells["tissue_general"]) if "tissue_general" in label_cells.columns else "unknown",
            "tissue": dominant_value(label_cells["tissue"]) if "tissue" in label_cells.columns else "unknown",
            "disease": dominant_value(label_cells["disease"]) if "disease" in label_cells.columns else "unknown",
            "cell_type_clean": str(label),
            "cell_ontology_id": ontology_info.get("cell_ontology_id"),
            "cell_ontology_label": ontology_info.get("cell_ontology_label"),
            "cell_ontology_parent_label": ontology_info.get("cell_ontology_parent_label"),
            "n_cells": n_cells,
            "de_method": MARKER_DE_METHOD,
            "positive_markers": positive_markers,
            "negative_markers": negative_markers,
            "avg_logfc_top5": avg_logfc_top5,
            "avg_padj_top5": avg_padj_top5,
            "marker_quality_score": quality_score,
            "hardness_flags": hardness_flags,
            # Phase-A: target label fields
            "cell_type_source_clean": str(label),
            "cell_type_target_label": str(label_cells["cell_type_target_label"].dropna().iloc[0]) if "cell_type_target_label" in label_cells.columns and len(label_cells["cell_type_target_label"].dropna()) > 0 else None,
            "cell_type_target_id": str(label_cells["cell_type_target_id"].dropna().iloc[0]) if "cell_type_target_id" in label_cells.columns and len(label_cells["cell_type_target_id"].dropna()) > 0 else None,
            # Phase-A: study/collection info
            "collection_name": dataset_meta.get("collection_name"),
            "collection_doi": dataset_meta.get("collection_doi"),
        }

        records.append(record)

    return records


def save_jsonl(records: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_manifest(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    if not df.empty and "file_name" in df.columns:
        priority = {
            "success": 5,
            "success_zero_examples": 5,
            "exists_jsonl_recovered": 4,
            "failed": 3,
        }
        df["_priority"] = df["status"].map(lambda x: priority.get(str(x), 0))
        df = (
            df.sort_values(by=["file_name", "_priority"])
            .drop_duplicates(subset=["file_name"], keep="last")
            .drop(columns=["_priority"])
            .reset_index(drop=True)
        )
    df.to_csv(path, index=False)


def save_run_summary(rows: List[Dict[str, Any]], summary_path: Path, jsonl_path: Path) -> None:
    df = pd.DataFrame(rows)
    total_files = len(df)
    success = int(df["status"].isin(["success", "success_zero_examples", "exists_jsonl_recovered"]).sum()) if total_files else 0
    failed = int((df["status"] == "failed").sum()) if total_files else 0
    total_examples = int(df["n_examples"].fillna(0).sum()) if total_files else 0

    lines = [
        "=== 04_make_marker_examples run summary ===",
        "",
        f"Total files: {total_files}",
        f"Success-like: {success}",
        f"Failed: {failed}",
        f"Total marker examples: {total_examples}",
        "",
        f"JSONL: {jsonl_path}",
    ]

    if failed > 0:
        lines.append("")
        lines.append("Failed files:")
        for _, row in df[df["status"] == "failed"].iterrows():
            lines.append(f"- {row.get('file_name', '')} | error={row.get('error', '')}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        logging.info("No existing marker manifest found, starting fresh.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        if "file_name" not in df.columns:
            logging.warning("Existing marker manifest missing 'file_name', ignoring old manifest.")
            return pd.DataFrame()
        df = df.drop_duplicates(subset=["file_name"], keep="last").reset_index(drop=True)
        logging.info("Loaded existing marker manifest: %s | rows=%d", path, len(df))
        return df
    except Exception as e:
        logging.warning("Failed to read existing marker manifest, ignoring it. error=%r", e)
        return pd.DataFrame()


def load_existing_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        logging.info("No existing marker jsonl found, starting fresh.")
        return []

    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                logging.warning("Skip invalid old JSONL line %d: %r", line_no, e)

    logging.info("Loaded existing marker jsonl: %s | records=%d", path, len(records))
    return records


def build_dataset_id_to_records(records: List[dict]) -> Dict[str, List[dict]]:
    mp: Dict[str, List[dict]] = {}
    for rec in records:
        dataset_id = str(rec.get("dataset_id", "unknown_dataset"))
        mp.setdefault(dataset_id, []).append(rec)
    return mp


def build_file_name_to_manifest_row(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty:
        return {}
    return {
        str(row["file_name"]): row.to_dict()
        for _, row in df.iterrows()
    }


def get_dataset_id_from_h5ad(h5ad_path: Path) -> str:
    return h5ad_path.stem


def remove_dataset_records(records: List[dict], dataset_id: str) -> List[dict]:
    return [r for r in records if str(r.get("dataset_id", "")) != str(dataset_id)]


def make_manifest_row(
    file_name: str,
    dataset_id: str,
    dataset_title: str,
    status: str,
    n_examples: int,
    elapsed_sec: float,
    error: str,
) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "dataset_id": dataset_id,
        "dataset_title": dataset_title,
        "status": status,
        "n_examples": n_examples,
        "elapsed_sec": round(elapsed_sec, 2),
        "error": error,
    }


def should_skip_file(
    h5ad_path: Path,
    existing_manifest_row: Dict[str, Any] | None,
    dataset_records_map: Dict[str, List[dict]],
) -> bool:
    if existing_manifest_row is None:
        return False

    status = str(existing_manifest_row.get("status", "")).strip().lower()
    dataset_id = str(existing_manifest_row.get("dataset_id", h5ad_path.stem))
    n_examples = int(existing_manifest_row.get("n_examples", 0) or 0)

    if status in {"success", "exists_jsonl_recovered"}:
        if n_examples > 0:
            return dataset_id in dataset_records_map and len(dataset_records_map[dataset_id]) > 0
        return not MARKER_RERUN_ZERO_EXAMPLE_SUCCESS

    if status == "success_zero_examples":
        return not MARKER_RERUN_ZERO_EXAMPLE_SUCCESS

    return False


def recover_rows_from_jsonl_only(
    input_files: List[Path],
    existing_manifest_map: Dict[str, Dict[str, Any]],
    dataset_records_map: Dict[str, List[dict]],
) -> Dict[str, Dict[str, Any]]:
    """
    若 manifest 丢失/不完整，但 jsonl 中已有某 dataset 的记录，则尝试恢复为 success-like 状态。
    """
    recovered: Dict[str, Dict[str, Any]] = {}

    if not MARKER_RECOVER_FROM_JSONL_IF_MANIFEST_MISSING:
        return recovered

    for h5ad_path in input_files:
        if h5ad_path.name in existing_manifest_map:
            continue

        dataset_id = get_dataset_id_from_h5ad(h5ad_path)
        old_records = dataset_records_map.get(dataset_id, [])

        if len(old_records) > 0:
            dataset_title = str(old_records[0].get("dataset_title", ""))
            recovered[h5ad_path.name] = make_manifest_row(
                file_name=h5ad_path.name,
                dataset_id=dataset_id,
                dataset_title=dataset_title,
                status="exists_jsonl_recovered",
                n_examples=len(old_records),
                elapsed_sec=0.0,
                error="",
            )
            logging.info(
                "Recovered manifest row from existing jsonl: %s | dataset_id=%s | n_examples=%d",
                h5ad_path.name, dataset_id, len(old_records)
            )

    return recovered


def remove_orphan_jsonl_records(
    all_records: List[dict],
    valid_dataset_ids: set[str],
) -> List[dict]:
    """
    可选：移除 JSONL 中那些不再对应当前 clean_h5ad 输入文件的旧记录。
    """
    if not MARKER_REMOVE_ORPHAN_JSONL_RECORDS:
        return all_records

    new_records = [r for r in all_records if str(r.get("dataset_id", "")) in valid_dataset_ids]
    removed = len(all_records) - len(new_records)
    if removed > 0:
        logging.warning("Removed %d orphan records from marker jsonl.", removed)
    return new_records


def save_v2_manifest_summary(
    manifest_rows: List[Dict[str, Any]],
    all_v2_records: List[dict],
    manifest_path: Path,
    summary_path: Path,
    out_jsonl: Path,
) -> None:
    """Save manifest and run summary for v2 output."""
    save_manifest(manifest_rows, manifest_path)
    df = pd.DataFrame(manifest_rows)
    total_files = len(df)
    success = int(df["status"].isin(["success", "success_zero_examples", "exists_jsonl_recovered"]).sum()) if total_files else 0
    failed = int((df["status"] == "failed").sum()) if total_files else 0
    total_examples = len(all_v2_records)

    # Compute manifest-level summary stats
    n_zero_example_labels = sum(1 for r in manifest_rows if int(r.get("n_examples", 0) or 0) == 0)
    quality_scores = [r.get("marker_quality_score") for r in all_v2_records if r.get("marker_quality_score") is not None]
    mean_quality = round(float(np.mean(quality_scores)), 4) if quality_scores else None
    n_ontology_mapped = sum(1 for r in all_v2_records if r.get("cell_ontology_id"))

    lines = [
        "=== 04_make_marker_examples v2 run summary ===",
        "",
        f"Total files: {total_files}",
        f"Success-like: {success}",
        f"Failed: {failed}",
        f"Total marker examples (v2): {total_examples}",
        "",
        f"n_zero_example_labels: {n_zero_example_labels}",
        f"mean_marker_quality_score: {mean_quality}",
        f"n_ontology_mapped_examples: {n_ontology_mapped}",
        f"DE method: {MARKER_DE_METHOD}",
        "",
        f"V2 JSONL: {out_jsonl}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_files = sorted(cfg.CLEAN_H5AD_DIR.glob("*.h5ad"))
    if not input_files:
        raise FileNotFoundError(f"No cleaned h5ad files found in {cfg.CLEAN_H5AD_DIR}")

    out_jsonl = Path(cfg.MARKER_EXAMPLES_JSONL)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1 v2 output paths
    out_v2_jsonl = Path(cfg.MARKER_EXAMPLES_V2_JSONL)
    out_v2_manifest = out_v2_jsonl.with_name("marker_examples_v2_manifest.csv")
    out_v2_summary = out_v2_jsonl.with_name("04_make_marker_examples_v2_run_summary.txt")

    manifest_path = out_jsonl.with_name("marker_examples_manifest.csv")
    summary_path = out_jsonl.with_name("04_make_marker_examples_run_summary.txt")

    logging.info("Start building marker examples.")
    logging.info("Input dir=%s", cfg.CLEAN_H5AD_DIR)
    logging.info("Output jsonl=%s", out_jsonl)
    logging.info("Found %d cleaned h5ad files", len(input_files))
    logging.info("Marker group column=%s", cfg.MARKER_GROUP_COL)
    logging.info("MARKER_RERUN_ZERO_EXAMPLE_SUCCESS=%s", MARKER_RERUN_ZERO_EXAMPLE_SUCCESS)
    logging.info("MARKER_RECOVER_FROM_JSONL_IF_MANIFEST_MISSING=%s", MARKER_RECOVER_FROM_JSONL_IF_MANIFEST_MISSING)
    logging.info("MARKER_REMOVE_ORPHAN_JSONL_RECORDS=%s", MARKER_REMOVE_ORPHAN_JSONL_RECORDS)
    logging.info("Phase 1 v2: DE method=%s | v2_available=%s", MARKER_DE_METHOD, _V2_AVAILABLE)

    # ========= 读取旧进度 =========
    existing_manifest_df = load_existing_manifest(manifest_path)
    existing_manifest_map = build_file_name_to_manifest_row(existing_manifest_df)

    all_records: List[dict] = load_existing_jsonl(out_jsonl)
    valid_dataset_ids = {p.stem for p in input_files}
    all_records = remove_orphan_jsonl_records(all_records, valid_dataset_ids)
    dataset_records_map = build_dataset_id_to_records(all_records)

    # ========= Phase 1 v2: 读取旧 v2 进度 =========
    all_v2_records: List[dict] = load_existing_jsonl(out_v2_jsonl)
    all_v2_records = remove_orphan_jsonl_records(all_v2_records, valid_dataset_ids)

    existing_v2_manifest_df = load_existing_manifest(out_v2_manifest)
    existing_v2_manifest_map = build_file_name_to_manifest_row(existing_v2_manifest_df)
    v2_manifest_rows_map: Dict[str, Dict[str, Any]] = dict(existing_v2_manifest_map)

    # ========= 若 manifest 缺失但 jsonl 里有数据，则自动恢复 =========
    recovered_rows = recover_rows_from_jsonl_only(input_files, existing_manifest_map, dataset_records_map)

    manifest_rows_map: Dict[str, Dict[str, Any]] = dict(existing_manifest_map)
    manifest_rows_map.update(recovered_rows)

    for h5ad_path in tqdm(input_files, desc="Make markers", unit="file"):
        existing_row = manifest_rows_map.get(h5ad_path.name)
        dataset_id = get_dataset_id_from_h5ad(h5ad_path)

        # 临时跳过超大且会导致崩溃的文件
        if h5ad_path.name == "19053a82-9c89-4fb8-bd19-d7b1800b0b7b.h5ad":
            logging.warning("Temporarily skip large file: %s", h5ad_path.name)
            continue

        if should_skip_file(h5ad_path, existing_row, dataset_records_map):
            logging.info(
                "Resume skip %s | previous status=%s | previous_examples=%s",
                h5ad_path.name,
                existing_row.get("status", "") if existing_row else "",
                existing_row.get("n_examples", 0) if existing_row else 0,
            )
            continue

        t0 = time.time()

        try:
            logging.info("Building markers for %s", h5ad_path.name)
            adata = ad.read_h5ad(h5ad_path)
            logging.info("[%s] read_h5ad ok shape=%s", h5ad_path.name, adata.shape)

            source_meta = adata.uns.get("source_meta", {}) if isinstance(adata.uns, dict) else {}
            source_dataset_id = str(source_meta.get("dataset_id", dataset_id))
            source_dataset_title = str(source_meta.get("dataset_title", h5ad_path.stem))

            # 若本次准备重跑，先删掉该 dataset 的旧 marker 记录，避免重复
            all_records = remove_dataset_records(all_records, source_dataset_id)
            all_v2_records = remove_dataset_records(all_v2_records, source_dataset_id)

            # v1: original extraction (t-test)
            records = extract_markers_for_dataset(adata)
            all_records.extend(records)
            dataset_records_map = build_dataset_id_to_records(all_records)

            # v2: evidence-aware extraction (wilcoxon + full stats)
            if _V2_AVAILABLE:
                try:
                    records_v2 = extract_markers_v2_for_dataset(adata)
                    all_v2_records.extend(records_v2)
                    logging.info(
                        "Generated %d v2 marker examples for %s",
                        len(records_v2), h5ad_path.name
                    )
                    v2_manifest_rows_map[h5ad_path.name] = make_manifest_row(
                        file_name=h5ad_path.name,
                        dataset_id=source_dataset_id,
                        dataset_title=source_dataset_title,
                        status="success" if records_v2 else "success_zero_examples",
                        n_examples=len(records_v2),
                        elapsed_sec=time.time() - t0,
                        error="",
                    )
                    # Add quality stats to v2 manifest row
                    if records_v2:
                        qs = [r.get("marker_quality_score") for r in records_v2 if r.get("marker_quality_score") is not None]
                        v2_manifest_rows_map[h5ad_path.name]["marker_quality_score"] = round(float(np.mean(qs)), 4) if qs else None
                except Exception as ev2:
                    logging.warning("v2 extraction failed for %s: %r", h5ad_path.name, ev2)
                    if h5ad_path.name not in v2_manifest_rows_map:
                        v2_manifest_rows_map[h5ad_path.name] = make_manifest_row(
                            file_name=h5ad_path.name,
                            dataset_id=source_dataset_id,
                            dataset_title=source_dataset_title,
                            status="failed",
                            n_examples=0,
                            elapsed_sec=time.time() - t0,
                            error=repr(ev2),
                        )

            status = "success" if len(records) > 0 else "success_zero_examples"

            manifest_rows_map[h5ad_path.name] = make_manifest_row(
                file_name=h5ad_path.name,
                dataset_id=source_dataset_id,
                dataset_title=source_dataset_title,
                status=status,
                n_examples=len(records),
                elapsed_sec=time.time() - t0,
                error="",
            )

            logging.info(
                "Generated %d marker examples for %s | status=%s",
                len(records), h5ad_path.name, status
            )

            del adata
            gc.collect()

        except Exception as e:
            logging.exception("Failed building markers for %s", h5ad_path.name)

            manifest_rows_map[h5ad_path.name] = make_manifest_row(
                file_name=h5ad_path.name,
                dataset_id=dataset_id,
                dataset_title="",
                status="failed",
                n_examples=0,
                elapsed_sec=time.time() - t0,
                error=repr(e),
            )

        if cfg.MARKER_WRITE_JSONL_EVERY_FILE:
            save_jsonl(all_records, out_jsonl)
            save_manifest(list(manifest_rows_map.values()), manifest_path)
            if _V2_AVAILABLE:
                save_jsonl(all_v2_records, out_v2_jsonl)
                save_manifest(list(v2_manifest_rows_map.values()), out_v2_manifest)

    if not cfg.MARKER_WRITE_JSONL_EVERY_FILE:
        save_jsonl(all_records, out_jsonl)
        save_manifest(list(manifest_rows_map.values()), manifest_path)
        if _V2_AVAILABLE:
            save_jsonl(all_v2_records, out_v2_jsonl)
            save_manifest(list(v2_manifest_rows_map.values()), out_v2_manifest)

    final_manifest_rows = list(manifest_rows_map.values())
    save_run_summary(final_manifest_rows, summary_path, out_jsonl)
    logging.info("Saved %d marker examples to %s", len(all_records), out_jsonl)

    if _V2_AVAILABLE:
        save_v2_manifest_summary(
            list(v2_manifest_rows_map.values()),
            all_v2_records,
            out_v2_manifest,
            out_v2_summary,
            out_v2_jsonl,
        )
        logging.info("Saved %d v2 marker examples to %s", len(all_v2_records), out_v2_jsonl)


if __name__ == "__main__":
    main()

# nohup python -u scripts/data_prep/04_make_marker_examples.py 2>&1 | tee data/meta/04_make_marker_examples.log

#  nohup bash -lc 'export PYTHONPATH="$PWD/src:$PYTHONPATH"; python -u scripts/data_prep/04_make_marker_examples.py >> data/meta/04_make_marker_examples.log 2>&1' >/dev/null 2>&1 &