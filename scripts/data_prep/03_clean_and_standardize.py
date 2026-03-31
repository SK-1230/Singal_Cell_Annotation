from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

import data_prep_config as cfg


# Phase 1: ontology-aware label normalization
try:
    from sca.data.label_normalization import normalize_and_map, init_alias_table
    _ONTOLOGY_AVAILABLE = True
except ImportError:
    _ONTOLOGY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def ensure_dirs() -> None:
    cfg.CLEAN_H5AD_DIR.mkdir(parents=True, exist_ok=True)
    Path(cfg.CLEAN_MANIFEST_CSV).parent.mkdir(parents=True, exist_ok=True)


def normalize_text_keep_na(x: Any) -> str | None:
    """
    保留缺失值，不要把 NaN 变成字符串 'nan'
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def canonicalize_label(x: Any) -> str | None:
    """
    对 cell_type 标签做更适合训练的标准化：
    - 保留缺失值
    - 压缩空格
    - 小写
    """
    s = normalize_text_keep_na(x)
    if s is None:
        return None
    s = s.lower()

    remap = {
        "t cells": "t cell",
        "b cells": "b cell",
        "nk cells": "nk cell",
        "myeloid cells": "myeloid cell",
    }
    return remap.get(s, s)


def is_ambiguous_label(label: str | None) -> bool:
    if label is None:
        return True
    low = label.lower().strip()
    if not low:
        return True
    return any(k in low for k in cfg.AMBIGUOUS_LABEL_KEYWORDS)


def normalize_series_with_default(s: pd.Series, default: str) -> pd.Series:
    s = s.astype("object").map(normalize_text_keep_na)
    s = s.where(pd.notna(s), default)
    return s


def make_empty_record(h5ad_path: Path) -> Dict[str, Any]:
    return {
        "dataset_id": h5ad_path.stem,
        "file_name": h5ad_path.name,
        "status": None,
        "n_obs_before": None,
        "n_vars_before": None,
        "n_obs_after": None,
        "n_vars_after": None,
        "unique_cell_types_after": None,
        "tissues": "",
        # Phase 1: ontology stats
        "n_mapped_ontology_labels": None,
        "n_unmapped_ontology_labels": None,
        "n_unique_cl_ids": None,
        "elapsed_sec": None,
        "error": "",
    }


def get_clean_output_paths(h5ad_path: Path) -> Dict[str, Path]:
    stem = h5ad_path.stem
    return {
        "clean_h5ad": cfg.CLEAN_H5AD_DIR / h5ad_path.name,
        "label_counts": cfg.CLEAN_H5AD_DIR / f"{stem}.label_counts.csv",
        "label_counts_gold": cfg.CLEAN_H5AD_DIR / f"{stem}.label_counts_gold.csv",
        # Phase 1 新增 sidecar 输出
        "cell_type_mapping": cfg.CLEAN_H5AD_DIR / f"{stem}.cell_type_mapping.csv",
        "dataset_profile": cfg.CLEAN_H5AD_DIR / f"{stem}.dataset_profile.json",
    }


def outputs_status(h5ad_path: Path) -> Dict[str, bool]:
    paths = get_clean_output_paths(h5ad_path)
    return {
        "clean_h5ad": paths["clean_h5ad"].exists(),
        "label_counts": paths["label_counts"].exists(),
        "label_counts_gold": paths["label_counts_gold"].exists(),
        "cell_type_mapping": paths["cell_type_mapping"].exists(),
        "dataset_profile": paths["dataset_profile"].exists(),
    }


def all_outputs_exist(h5ad_path: Path) -> bool:
    st = outputs_status(h5ad_path)
    return (
        st["clean_h5ad"]
        and st["label_counts"]
        and st["label_counts_gold"]
        and st["cell_type_mapping"]
        and st["dataset_profile"]
    )


def save_manifest(rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)

    preferred_status_order = {
        "cleaned": 5,
        "exists_cleaned_complete": 5,
        "exists_cleaned_repaired": 5,
        "skipped": 4,
        "failed": 3,
    }

    if not df.empty and "file_name" in df.columns and "status" in df.columns:
        df["_rank"] = df["status"].map(lambda x: preferred_status_order.get(str(x), 0))
        df = (
            df.sort_values(by=["file_name", "_rank"])
            .drop_duplicates(subset=["file_name"], keep="last")
            .drop(columns=["_rank"])
            .reset_index(drop=True)
        )

    df.to_csv(cfg.CLEAN_MANIFEST_CSV, index=False)
    logging.info("Saved clean manifest to %s | rows=%d", cfg.CLEAN_MANIFEST_CSV, len(df))


def save_run_summary(rows: List[Dict[str, Any]]) -> None:
    summary_path = Path(cfg.CLEAN_MANIFEST_CSV).with_name("03_clean_and_standardize_run_summary.txt")
    df = pd.DataFrame(rows)

    total = len(df)
    cleaned_like = {
        "cleaned",
        "exists_cleaned_complete",
        "exists_cleaned_repaired",
    }
    success = int(df["status"].isin(cleaned_like).sum()) if total else 0
    skipped = int((df["status"] == "skipped").sum()) if total else 0
    failed = int((df["status"] == "failed").sum()) if total else 0

    lines = [
        "=== 03_clean_and_standardize run summary ===",
        "",
        f"Total files seen: {total}",
        f"Success-like: {success}",
        f"Skipped: {skipped}",
        f"Failed: {failed}",
        "",
        f"Manifest: {cfg.CLEAN_MANIFEST_CSV}",
        f"Output dir: {cfg.CLEAN_H5AD_DIR}",
        "",
    ]

    if failed > 0:
        lines.append("Failed files:")
        for _, row in df[df["status"] == "failed"].iterrows():
            lines.append(f"- {row.get('file_name', '')} | error={row.get('error', '')}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved run summary to %s", summary_path)


def load_existing_manifest() -> pd.DataFrame:
    manifest_path = Path(cfg.CLEAN_MANIFEST_CSV)
    if not manifest_path.exists():
        logging.info("No existing clean manifest found, starting fresh.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(manifest_path)
        if "file_name" not in df.columns:
            logging.warning("Existing manifest missing 'file_name' column, ignoring old manifest.")
            return pd.DataFrame()

        df = df.drop_duplicates(subset=["file_name"], keep="last").reset_index(drop=True)
        logging.info("Loaded existing clean manifest: %s | rows=%d", manifest_path, len(df))
        return df
    except Exception as e:
        logging.warning("Failed to read existing manifest, will start fresh. error=%r", e)
        return pd.DataFrame()


def apply_ontology_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Add ontology-aware columns to obs DataFrame:
      cell_type_input_raw, cell_type_gold, cell_type_clean (already present),
      cell_ontology_id, cell_ontology_label, cell_ontology_parent_label,
      cell_type_status, cell_type_level
    """
    if not _ONTOLOGY_AVAILABLE:
        logging.warning("sca.data.label_normalization not available — skipping ontology columns")
        for col in ["cell_type_input_raw", "cell_ontology_id", "cell_ontology_label",
                    "cell_ontology_parent_label", "cell_type_status", "cell_type_level"]:
            if col not in obs.columns:
                obs[col] = None
        return obs

    # Initialize alias table
    init_alias_table(cfg.LABEL_ALIASES_TSV)

    # cell_type_input_raw = original raw value before any normalization
    if "cell_type_input_raw" not in obs.columns:
        # Use cell_type_gold as proxy for raw input
        obs["cell_type_input_raw"] = obs.get("cell_type_gold", obs.get("cell_type_clean", pd.Series([None] * len(obs), index=obs.index)))

    # Map each unique clean label once for efficiency
    unique_labels = obs["cell_type_clean"].dropna().astype(str).unique()
    mapping_cache: Dict[str, Dict[str, Any]] = {}
    for lbl in unique_labels:
        mapping_cache[lbl] = normalize_and_map(lbl, tsv_path=cfg.LABEL_ALIASES_TSV)

    def get_field(label_val, field: str):
        if pd.isna(label_val):
            return None
        return mapping_cache.get(str(label_val), {}).get(field)

    obs["cell_ontology_id"] = obs["cell_type_clean"].map(lambda x: get_field(x, "cell_ontology_id"))
    obs["cell_ontology_label"] = obs["cell_type_clean"].map(lambda x: get_field(x, "cell_ontology_label"))
    obs["cell_ontology_parent_label"] = obs["cell_type_clean"].map(lambda x: get_field(x, "cell_ontology_parent_label"))
    obs["cell_type_status"] = obs["cell_type_clean"].map(lambda x: get_field(x, "cell_type_status"))
    obs["cell_type_level"] = obs["cell_type_clean"].map(lambda x: get_field(x, "cell_type_level"))

    return obs


def write_cell_type_mapping(adata: ad.AnnData, stem: str) -> None:
    """Write per-unique-label cell_type_mapping.csv sidecar."""
    obs = adata.obs.copy()
    group_cols = [
        "cell_type_input_raw", "cell_type_gold", "cell_type_clean",
        "cell_ontology_id", "cell_ontology_label", "cell_ontology_parent_label",
        "cell_type_status",
    ]
    # Only include cols that exist
    group_cols = [c for c in group_cols if c in obs.columns]

    if "cell_type_clean" not in group_cols:
        return

    mapping_df = (
        obs.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_cells")
    )
    out_path = cfg.CLEAN_H5AD_DIR / f"{stem}.cell_type_mapping.csv"
    mapping_df.to_csv(out_path, index=False)


def write_dataset_profile(adata: ad.AnnData, stem: str, dataset_id: str) -> None:
    """Write dataset_profile.json sidecar with dataset-level statistics."""
    obs = adata.obs

    source_meta = adata.uns.get("source_meta", {}) if isinstance(adata.uns, dict) else {}
    dataset_title = str(source_meta.get("dataset_title", stem))
    organism = str(source_meta.get("organism", getattr(cfg, "ORGANISM", "unknown")))

    n_mapped = int(obs["cell_ontology_id"].notna().sum()) if "cell_ontology_id" in obs.columns else 0
    n_unmapped = int(adata.n_obs) - n_mapped
    n_unique_cl_ids = int(obs["cell_ontology_id"].dropna().nunique()) if "cell_ontology_id" in obs.columns else 0
    n_unique_ct = int(obs["cell_type_clean"].astype(str).nunique()) if "cell_type_clean" in obs.columns else 0
    mapped_ratio = round(n_mapped / adata.n_obs, 4) if adata.n_obs > 0 else 0.0

    dominant_tissue_general = "unknown"
    if "tissue_general" in obs.columns:
        series = obs["tissue_general"].dropna().astype(str)
        if len(series) > 0:
            dominant_tissue_general = series.value_counts().index[0]

    dominant_disease = "unknown"
    if "disease" in obs.columns:
        series = obs["disease"].dropna().astype(str)
        if len(series) > 0:
            dominant_disease = series.value_counts().index[0]

    profile = {
        "dataset_id": dataset_id,
        "dataset_title": dataset_title,
        "organism": organism,
        "dominant_tissue_general": dominant_tissue_general,
        "dominant_disease": dominant_disease,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_unique_cell_types": n_unique_ct,
        "n_unique_cl_ids": n_unique_cl_ids,
        "mapped_ratio": mapped_ratio,
    }

    out_path = cfg.CLEAN_H5AD_DIR / f"{stem}.dataset_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def write_label_count_files_from_clean_adata(adata: ad.AnnData, stem: str) -> None:
    if "cell_type_clean" not in adata.obs.columns:
        raise ValueError(f"{stem}: cleaned h5ad missing obs['cell_type_clean']")
    if "cell_type_gold" not in adata.obs.columns:
        raise ValueError(f"{stem}: cleaned h5ad missing obs['cell_type_gold']")

    label_counts = (
        adata.obs["cell_type_clean"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_type_clean")
        .reset_index(name="cell_count")
    )
    label_count_path = cfg.CLEAN_H5AD_DIR / f"{stem}.label_counts.csv"
    label_counts.to_csv(label_count_path, index=False)

    label_counts_gold = (
        adata.obs["cell_type_gold"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_type_gold")
        .reset_index(name="cell_count")
    )
    label_count_gold_path = cfg.CLEAN_H5AD_DIR / f"{stem}.label_counts_gold.csv"
    label_counts_gold.to_csv(label_count_gold_path, index=False)


def repair_outputs_from_clean_h5ad(h5ad_path: Path) -> Dict[str, Any]:
    """
    当 clean 后的 h5ad 已存在，但标签统计文件不完整时，
    直接从 clean h5ad 补生成 label_counts 文件，而不是整套重跑。
    """
    t0 = time.time()
    record = make_empty_record(h5ad_path)
    paths = get_clean_output_paths(h5ad_path)

    if not paths["clean_h5ad"].exists():
        raise FileNotFoundError(f"clean h5ad not found: {paths['clean_h5ad']}")

    logging.info("Repairing side outputs from existing cleaned file: %s", paths["clean_h5ad"].name)
    adata = ad.read_h5ad(paths["clean_h5ad"])

    record["n_obs_after"] = int(adata.n_obs)
    record["n_vars_after"] = int(adata.n_vars)
    if "cell_type_clean" in adata.obs.columns:
        record["unique_cell_types_after"] = int(adata.obs["cell_type_clean"].astype(str).nunique())
    if "tissue_general" in adata.obs.columns:
        record["tissues"] = ";".join(sorted(set(adata.obs["tissue_general"].astype(str))))

    source_meta = adata.uns.get("source_meta", {}) if isinstance(adata.uns, dict) else {}
    if isinstance(source_meta, dict):
        record["dataset_id"] = str(source_meta.get("dataset_id", h5ad_path.stem))

    # Phase 1: add ontology columns if missing
    if "cell_ontology_id" not in adata.obs.columns:
        adata.obs = apply_ontology_columns(adata.obs)
        if cfg.CLEAN_WRITE_COMPRESSION:
            adata.write_h5ad(paths["clean_h5ad"], compression=cfg.CLEAN_WRITE_COMPRESSION)
        else:
            adata.write_h5ad(paths["clean_h5ad"])

    # Phase 1: ontology stats for manifest
    if "cell_ontology_id" in adata.obs.columns:
        n_mapped = int(adata.obs["cell_ontology_id"].notna().sum())
        record["n_mapped_ontology_labels"] = n_mapped
        record["n_unmapped_ontology_labels"] = int(adata.n_obs) - n_mapped
        record["n_unique_cl_ids"] = int(adata.obs["cell_ontology_id"].dropna().nunique())

    write_label_count_files_from_clean_adata(adata, h5ad_path.stem)

    # Phase 1: write sidecar outputs if missing
    if not paths["cell_type_mapping"].exists():
        write_cell_type_mapping(adata, h5ad_path.stem)
    if not paths["dataset_profile"].exists():
        write_dataset_profile(adata, h5ad_path.stem, record["dataset_id"])

    record["status"] = "exists_cleaned_repaired"
    record["elapsed_sec"] = round(time.time() - t0, 2)

    del adata
    gc.collect()
    return record


def should_skip_by_manifest_and_outputs(existing_row: pd.Series | None, h5ad_path: Path) -> bool:
    """
    只有在 manifest 表示该文件已成功/已跳过，且当前输出状态与之匹配时，才真正跳过。
    这样避免 manifest 说 cleaned，但 side outputs 缺失时被错误跳过。
    """
    if existing_row is None:
        return False

    status = str(existing_row.get("status", "")).strip()

    if status == "skipped":
        return True

    if status in {"cleaned", "exists_cleaned_complete", "exists_cleaned_repaired"}:
        return all_outputs_exist(h5ad_path)

    return False


def clean_one_file(h5ad_path: Path) -> Dict[str, Any]:
    t0 = time.time()
    record: Dict[str, Any] = make_empty_record(h5ad_path)

    logging.info("Cleaning %s", h5ad_path.name)

    # 文件大小预检：超过阈值直接跳过，避免读取超大文件卡死
    if cfg.CLEAN_MAX_RAW_FILE_BYTES is not None:
        file_bytes = h5ad_path.stat().st_size
        if file_bytes > cfg.CLEAN_MAX_RAW_FILE_BYTES:
            size_gb = file_bytes / (1024 ** 3)
            limit_gb = cfg.CLEAN_MAX_RAW_FILE_BYTES / (1024 ** 3)
            logging.warning(
                "Skip %s: file too large (%.2f GB > %.2f GB limit)",
                h5ad_path.name, size_gb, limit_gb,
            )
            record["status"] = "skipped"
            record["error"] = f"file_too_large: {size_gb:.2f} GB"
            record["elapsed_sec"] = round(time.time() - t0, 2)
            return record

    adata = ad.read_h5ad(h5ad_path)

    record["n_obs_before"] = int(adata.n_obs)
    record["n_vars_before"] = int(adata.n_vars)

    source_meta = adata.uns.get("source_meta", {}) if isinstance(adata.uns, dict) else {}
    if isinstance(source_meta, dict):
        record["dataset_id"] = str(source_meta.get("dataset_id", h5ad_path.stem))

    if adata.n_obs == 0 or adata.n_vars == 0:
        record["status"] = "skipped"
        record["error"] = "empty input anndata"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        del adata
        gc.collect()
        return record

    # =========================
    # 标准化基因名
    # =========================
    if "feature_name" in adata.var.columns:
        feature_names = adata.var["feature_name"].astype("object")
        feature_names = feature_names.where(feature_names.notna(), adata.var_names.astype(str))
        adata.var_names = feature_names.astype(str)

    adata.var_names_make_unique()
    adata.var.index.name = "gene_symbol"

    if "feature_name" in adata.var.columns:
        adata.var["feature_name_original"] = adata.var["feature_name"].astype(str)
        adata.var = adata.var.drop(columns=["feature_name"])

    # =========================
    # 标准化 obs 字段
    # =========================
    if "cell_type" not in adata.obs.columns:
        record["status"] = "skipped"
        record["error"] = "missing cell_type column"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        logging.warning("Skip %s: missing cell_type", h5ad_path.name)
        del adata
        gc.collect()
        return record

    obs = adata.obs.copy()

    obs["cell_type_gold"] = obs["cell_type"].map(normalize_text_keep_na)
    obs["cell_type_clean"] = obs["cell_type"].map(canonicalize_label)

    if "tissue_general" in obs.columns:
        obs["tissue_general"] = normalize_series_with_default(obs["tissue_general"], "unknown")
    else:
        obs["tissue_general"] = "unknown"

    if "tissue" in obs.columns:
        obs["tissue"] = obs["tissue"].astype("object").map(normalize_text_keep_na)
        obs["tissue"] = obs["tissue"].where(pd.notna(obs["tissue"]), obs["tissue_general"])
    else:
        obs["tissue"] = obs["tissue_general"]

    if "disease" in obs.columns:
        obs["disease"] = normalize_series_with_default(obs["disease"], "unknown")
    else:
        obs["disease"] = "unknown"

    # =========================
    # 基础过滤
    # =========================
    keep = np.ones(adata.n_obs, dtype=bool)

    if "is_primary_data" in obs.columns:
        primary_mask = obs["is_primary_data"]
        if primary_mask.dtype != bool:
            primary_mask = primary_mask.astype(str).str.lower().isin(["true", "1", "yes"])
        keep &= primary_mask.values

    keep &= obs["cell_type_gold"].notna().values
    keep &= obs["cell_type_clean"].notna().values
    keep &= ~obs["cell_type_clean"].map(is_ambiguous_label).values

    adata = adata[keep].copy()
    obs = obs.loc[adata.obs_names].copy()
    adata.obs = obs

    if adata.n_obs == 0:
        record["status"] = "skipped"
        record["error"] = "no cells left after basic filtering"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        logging.warning("Skip %s: no cells left after basic filtering", h5ad_path.name)
        del adata
        gc.collect()
        return record

    # =========================
    # 保存当前输入矩阵备份
    # =========================
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # =========================
    # 过滤低频基因
    # =========================
    sc.pp.filter_genes(adata, min_cells=cfg.MIN_GENES_DETECTED_IN_CELLS)

    if adata.n_vars == 0:
        record["status"] = "skipped"
        record["error"] = "no genes left after gene filtering"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        logging.warning("Skip %s: no genes left after gene filtering", h5ad_path.name)
        del adata
        gc.collect()
        return record

    # =========================
    # 过滤样本数过少的标签
    # =========================
    counts = adata.obs["cell_type_clean"].value_counts()
    valid_labels = counts[counts >= cfg.MIN_CELLS_PER_LABEL].index
    adata = adata[adata.obs["cell_type_clean"].isin(valid_labels)].copy()

    if adata.n_obs == 0:
        record["status"] = "skipped"
        record["error"] = "no cells left after label-frequency filtering"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        logging.warning("Skip %s: no cells left after label-frequency filtering", h5ad_path.name)
        del adata
        gc.collect()
        return record

    # Phase 1: add cell_type_input_raw before compressing categories
    obs["cell_type_input_raw"] = obs["cell_type"].map(normalize_text_keep_na)
    adata.obs = obs

    # Phase 1: apply ontology columns
    adata.obs = apply_ontology_columns(adata.obs)

    # Phase 1: collect ontology stats before compressing to category
    n_mapped = int(adata.obs["cell_ontology_id"].notna().sum()) if "cell_ontology_id" in adata.obs.columns else 0
    n_unmapped = int(adata.n_obs) - n_mapped
    n_unique_cl_ids = int(adata.obs["cell_ontology_id"].dropna().nunique()) if "cell_ontology_id" in adata.obs.columns else 0

    # 类别列压缩
    adata.obs["cell_type_clean"] = adata.obs["cell_type_clean"].astype("category")
    adata.obs["cell_type_gold"] = adata.obs["cell_type_gold"].astype("category")
    adata.obs["tissue_general"] = adata.obs["tissue_general"].astype("category")
    adata.obs["tissue"] = adata.obs["tissue"].astype("category")
    adata.obs["disease"] = adata.obs["disease"].astype("category")

    # 先写主 clean h5ad
    out_path = cfg.CLEAN_H5AD_DIR / h5ad_path.name
    if cfg.CLEAN_WRITE_COMPRESSION:
        adata.write_h5ad(out_path, compression=cfg.CLEAN_WRITE_COMPRESSION)
    else:
        adata.write_h5ad(out_path)

    # 再写 side outputs
    write_label_count_files_from_clean_adata(adata, h5ad_path.stem)

    # Phase 1: write sidecar outputs
    write_cell_type_mapping(adata, h5ad_path.stem)
    write_dataset_profile(adata, h5ad_path.stem, record["dataset_id"])

    record["status"] = "cleaned"
    record["n_obs_after"] = int(adata.n_obs)
    record["n_vars_after"] = int(adata.n_vars)
    record["unique_cell_types_after"] = int(adata.obs["cell_type_clean"].astype(str).nunique())
    record["tissues"] = ";".join(sorted(set(adata.obs["tissue_general"].astype(str))))
    # Phase 1 ontology stats
    record["n_mapped_ontology_labels"] = n_mapped
    record["n_unmapped_ontology_labels"] = n_unmapped
    record["n_unique_cl_ids"] = n_unique_cl_ids
    record["elapsed_sec"] = round(time.time() - t0, 2)

    logging.info(
        "Saved cleaned %s | cells %d -> %d | genes %d -> %d | labels=%d",
        out_path.name,
        record["n_obs_before"],
        record["n_obs_after"],
        record["n_vars_before"],
        record["n_vars_after"],
        record["unique_cell_types_after"],
    )

    del adata
    gc.collect()

    return record


def main() -> None:
    overall_t0 = time.time()
    ensure_dirs()

    h5ad_files = sorted(cfg.RAW_H5AD_DIR.glob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No .h5ad files found in {cfg.RAW_H5AD_DIR}")

    logging.info("Start cleaning exported h5ad files.")
    logging.info("Input dir=%s", cfg.RAW_H5AD_DIR)
    logging.info("Output dir=%s", cfg.CLEAN_H5AD_DIR)
    logging.info("Found %d h5ad files", len(h5ad_files))
    logging.info("MIN_GENES_DETECTED_IN_CELLS=%s", cfg.MIN_GENES_DETECTED_IN_CELLS)
    logging.info("MIN_CELLS_PER_LABEL=%s", cfg.MIN_CELLS_PER_LABEL)

    existing_df = load_existing_manifest()
    existing_map: Dict[str, Dict[str, Any]] = {}
    if not existing_df.empty:
        existing_map = {
            str(row["file_name"]): row.to_dict()
            for _, row in existing_df.iterrows()
        }

    rows_map: Dict[str, Dict[str, Any]] = dict(existing_map)

    for h5ad_path in tqdm(h5ad_files, desc="Clean h5ad", unit="file"):
        existing_row = existing_map.get(h5ad_path.name)
        out_st = outputs_status(h5ad_path)

        # 情况1：manifest 和当前输出都完整，直接跳过
        if existing_row is not None and should_skip_by_manifest_and_outputs(pd.Series(existing_row), h5ad_path):
            logging.info("Resume skip %s | previous status=%s", h5ad_path.name, existing_row.get("status", ""))
            rows_map[h5ad_path.name] = existing_row
            continue

        # 情况2：主 clean h5ad 存在，side outputs 也完整，但 manifest 没记好
        if out_st["clean_h5ad"] and out_st["label_counts"] and out_st["label_counts_gold"]:
            logging.info("Detected complete existing cleaned outputs: %s", h5ad_path.name)
            result = make_empty_record(h5ad_path)
            result["status"] = "exists_cleaned_complete"
            result["elapsed_sec"] = 0.0
            rows_map[h5ad_path.name] = result
            if cfg.CLEAN_SAVE_MANIFEST_EVERY_FILE:
                save_manifest(list(rows_map.values()))
            continue

        # 情况3：主 clean h5ad 已存在，但 side outputs 缺失，自动补
        if out_st["clean_h5ad"] and (not out_st["label_counts"] or not out_st["label_counts_gold"]):
            try:
                result = repair_outputs_from_clean_h5ad(h5ad_path)
            except Exception as e:
                logging.exception("Failed repairing side outputs for %s", h5ad_path.name)
                result = make_empty_record(h5ad_path)
                result["status"] = "failed"
                result["error"] = f"repair_failed: {repr(e)}"

            rows_map[h5ad_path.name] = result
            if cfg.CLEAN_SAVE_MANIFEST_EVERY_FILE:
                save_manifest(list(rows_map.values()))
            continue

        # 情况4：没有 clean h5ad，正常走完整清洗
        try:
            result = clean_one_file(h5ad_path)
        except Exception as e:
            logging.exception("Failed cleaning %s", h5ad_path.name)
            result = make_empty_record(h5ad_path)
            result["status"] = "failed"
            result["error"] = repr(e)

        rows_map[h5ad_path.name] = result

        if cfg.CLEAN_SAVE_MANIFEST_EVERY_FILE:
            save_manifest(list(rows_map.values()))

    final_rows = list(rows_map.values())

    if not cfg.CLEAN_SAVE_MANIFEST_EVERY_FILE:
        save_manifest(final_rows)

    save_run_summary(final_rows)
    logging.info("All done. Total elapsed = %.2fs", time.time() - overall_t0)

    df = pd.DataFrame(final_rows)
    success_like = int(
        df["status"].isin({"cleaned", "exists_cleaned_complete", "exists_cleaned_repaired"}).sum()
    ) if not df.empty else 0
    skipped = int((df["status"] == "skipped").sum()) if not df.empty else 0
    failed = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nClean summary:")
    print(f"- success_like: {success_like}")
    print(f"- skipped:      {skipped}")
    print(f"- failed:       {failed}")
    print(f"- manifest:     {cfg.CLEAN_MANIFEST_CSV}")
    print(f"- out dir:      {cfg.CLEAN_H5AD_DIR}")


if __name__ == "__main__":
    main()

# nohup python -u scripts/data_prep/03_clean_and_standardize.py 2>&1 | tee data/meta/03_clean_and_standardize.log
# ps -ef | grep scripts/data_prep/03_clean_and_standardize.py

'''
raw到clean的过程中，实际执行的时候，程序卡在了文件b252b015，查看发现该文件大小为25GB。此外，selected集合中，还有一个raw格式文件f7c1c579的大小为27GB
所以该函数是不是应该添加条件，比如看到raw文件为10GB以上，就略过。还是说在01py的地方判断，如果dataset特别特别大，就不加入到selected集合。但是似乎在01阶段，
无法通过文件大小判断？这是我的猜测，是不是01阶段获取selected集合的时候，只能看到数据集的其它信息。
'''