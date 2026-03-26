from __future__ import annotations

import gc
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

import config as cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =========================
# 可调参数
# =========================
WRITE_COMPRESSION = None   # 可改为 "gzip"
SAVE_MANIFEST_EVERY_FILE = True


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

    # 这里可以逐步扩展你的标签归一化规则
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


def save_manifest(rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(cfg.CLEAN_MANIFEST_CSV, index=False)
    logging.info("Saved clean manifest to %s | rows=%d", cfg.CLEAN_MANIFEST_CSV, len(rows))


def save_run_summary(rows: List[Dict[str, Any]]) -> None:
    summary_path = Path(cfg.CLEAN_MANIFEST_CSV).with_name("03_clean_and_standardize_run_summary.txt")
    df = pd.DataFrame(rows)

    total = len(df)
    success = int((df["status"] == "cleaned").sum()) if total else 0
    skipped = int((df["status"] == "skipped").sum()) if total else 0
    failed = int((df["status"] == "failed").sum()) if total else 0

    lines = [
        "=== 03_clean_and_standardize run summary ===",
        "",
        f"Total files seen: {total}",
        f"Cleaned: {success}",
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



def normalize_series_with_default(s: pd.Series, default: str) -> pd.Series:
    s = s.astype("object").map(normalize_text_keep_na)
    s = s.where(pd.notna(s), default)
    return s


def clean_one_file(h5ad_path: Path) -> Dict[str, Any]:
    t0 = time.time()

    record: Dict[str, Any] = {
        "dataset_id": h5ad_path.stem,
        "file_name": h5ad_path.name,
        "status": None,
        "n_obs_before": None,
        "n_vars_before": None,
        "n_obs_after": None,
        "n_vars_after": None,
        "unique_cell_types_after": None,
        "tissues": "",
        "elapsed_sec": None,
        "error": "",
    }

    logging.info("Cleaning %s", h5ad_path.name)

    adata = ad.read_h5ad(h5ad_path)

    record["n_obs_before"] = int(adata.n_obs)
    record["n_vars_before"] = int(adata.n_vars)

    # 尽量从 source_meta 恢复 dataset_id
    source_meta = adata.uns.get("source_meta", {}) if isinstance(adata.uns, dict) else {}
    if isinstance(source_meta, dict):
        record["dataset_id"] = str(source_meta.get("dataset_id", h5ad_path.stem))

    if adata.n_obs == 0 or adata.n_vars == 0:
        record["status"] = "skipped"
        record["error"] = "empty input anndata"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        return record

    # =========================
    # 标准化基因名
    # =========================
    if "feature_name" in adata.var.columns:
        feature_names = adata.var["feature_name"].astype("object")
        feature_names = feature_names.where(feature_names.notna(), adata.var_names.astype(str))
        adata.var_names = feature_names.astype(str)

    adata.var_names_make_unique()

    # 避免写 h5ad 时 index.name 与同名列 feature_name 冲突
    adata.var.index.name = "gene_symbol"

    # 如果你想保留原始 feature_name 列，就留着；
    # 如果仍担心冲突，也可以把它改名
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
        # 正常情况下这里就是 bool；保险起见做更稳一点的转换
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
        return record

    # 保存标签分布表
    label_counts = (
        adata.obs["cell_type_clean"]
        .value_counts()
        .rename_axis("cell_type_clean")
        .reset_index(name="cell_count")
    )

    label_count_path = cfg.CLEAN_H5AD_DIR / f"{h5ad_path.stem}.label_counts.csv"
    label_counts.to_csv(label_count_path, index=False)

    # 把 gold label 也保存一份
    label_counts_gold = (
        adata.obs["cell_type_gold"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_type_gold")
        .reset_index(name="cell_count")
    )
    label_count_gold_path = cfg.CLEAN_H5AD_DIR / f"{h5ad_path.stem}.label_counts_gold.csv"
    label_counts_gold.to_csv(label_count_gold_path, index=False)


    # 可选：把 cell_type_clean 设成 category，省点内存
    adata.obs["cell_type_clean"] = adata.obs["cell_type_clean"].astype("category")
    adata.obs["cell_type_gold"] = adata.obs["cell_type_gold"].astype("category")
    adata.obs["tissue_general"] = adata.obs["tissue_general"].astype("category")
    adata.obs["tissue"] = adata.obs["tissue"].astype("category")
    adata.obs["disease"] = adata.obs["disease"].astype("category")

    out_path = cfg.CLEAN_H5AD_DIR / h5ad_path.name
    if WRITE_COMPRESSION:
        adata.write_h5ad(out_path, compression=WRITE_COMPRESSION)
    else:
        adata.write_h5ad(out_path)

    record["status"] = "cleaned"
    record["n_obs_after"] = int(adata.n_obs)
    record["n_vars_after"] = int(adata.n_vars)
    record["unique_cell_types_after"] = int(adata.obs["cell_type_clean"].nunique())
    record["tissues"] = ";".join(sorted(set(adata.obs["tissue_general"].astype(str))))
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

    rows: List[Dict[str, Any]] = []

    for h5ad_path in tqdm(h5ad_files, desc="Clean h5ad", unit="file"):
        try:
            result = clean_one_file(h5ad_path)
        except Exception as e:
            logging.exception("Failed cleaning %s", h5ad_path.name)
            result = {
                "dataset_id": h5ad_path.stem,
                "file_name": h5ad_path.name,
                "status": "failed",
                "n_obs_before": None,
                "n_vars_before": None,
                "n_obs_after": None,
                "n_vars_after": None,
                "unique_cell_types_after": None,
                "tissues": "",
                "elapsed_sec": None,
                "error": repr(e),
            }

        rows.append(result)

        if SAVE_MANIFEST_EVERY_FILE:
            save_manifest(rows)

    if not SAVE_MANIFEST_EVERY_FILE:
        save_manifest(rows)

    save_run_summary(rows)
    logging.info("All done. Total elapsed = %.2fs", time.time() - overall_t0)

    df = pd.DataFrame(rows)
    cleaned = int((df["status"] == "cleaned").sum()) if not df.empty else 0
    skipped = int((df["status"] == "skipped").sum()) if not df.empty else 0
    failed = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nClean summary:")
    print(f"- cleaned: {cleaned}")
    print(f"- skipped: {skipped}")
    print(f"- failed:  {failed}")
    print(f"- manifest: {cfg.CLEAN_MANIFEST_CSV}")
    print(f"- out dir:  {cfg.CLEAN_H5AD_DIR}")


if __name__ == "__main__":
    main()

# python -u 03_clean_and_standardize.py 2>&1 | tee data/meta/03_clean_and_standardize.log