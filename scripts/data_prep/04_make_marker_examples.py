from __future__ import annotations

import gc
import json
import logging
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

# 可调参数
MARKER_GROUP_COL = "cell_type_clean"   # 优先用 clean 标签做 marker
MARKER_MIN_TOP_GENES = 5               # 至少保留多少 marker 才输出
MARKER_MIN_GENE_CELLS = 5              # marker 阶段再次过滤低频基因
WRITE_JSONL_EVERY_FILE = True


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

    if MARKER_GROUP_COL not in adata.obs.columns:
        return records

    if adata.obs[MARKER_GROUP_COL].nunique() < 2:
        return records

    adata_work = adata.copy()

    # 只保留有效标签
    adata_work = adata_work[adata_work.obs[MARKER_GROUP_COL].notna()].copy()
    if adata_work.n_obs == 0:
        return records

    # 每类最多抽样一定数量细胞，防止超大类压制计算
    adata_work = downsample_per_label(adata_work, MARKER_GROUP_COL)
    if adata_work.n_obs == 0:
        return records

    # 使用 counts 层
    if "counts" in adata_work.layers:
        adata_work.X = adata_work.layers["counts"].copy()

    # marker 阶段再做一次轻量基因过滤
    sc.pp.filter_genes(adata_work, min_cells=MARKER_MIN_GENE_CELLS)
    if adata_work.n_vars == 0:
        return records

    # 归一化 + log1p
    sc.pp.normalize_total(adata_work, target_sum=1e4)
    sc.pp.log1p(adata_work)

    # 显式设 category，groupby 更稳
    adata_work.obs[MARKER_GROUP_COL] = adata_work.obs[MARKER_GROUP_COL].astype("category")

    # 差异分析：每个标签 vs 其余
    sc.tl.rank_genes_groups(
        adata_work,
        groupby=MARKER_GROUP_COL,
        method="t-test",
        corr_method="benjamini-hochberg",
        n_genes=cfg.MAX_CANDIDATE_MARKERS,
        use_raw=False,
    )

    dataset_meta = adata.uns.get("source_meta", {})
    dataset_id = dataset_meta.get("dataset_id", "unknown_dataset")
    dataset_title = dataset_meta.get("dataset_title", "unknown_title")

    categories = list(adata_work.obs[MARKER_GROUP_COL].cat.categories)

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

        # 可选：再过滤 logFC <= 0 的基因，保留更像“正 marker”的基因
        if "logfoldchanges" in df.columns:
            df = df[df["logfoldchanges"].replace([np.inf, -np.inf], np.nan).fillna(-np.inf) > 0]

        top_df = df.head(cfg.TOP_K_MARKERS).copy()
        if len(top_df) < MARKER_MIN_TOP_GENES:
            continue

        label_cells = adata.obs[adata.obs[MARKER_GROUP_COL].astype(str) == str(label)]

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


def save_jsonl(records: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_manifest(rows: List[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_run_summary(rows: List[Dict[str, Any]], summary_path: Path, jsonl_path: Path) -> None:
    df = pd.DataFrame(rows)
    total_files = len(df)
    success = int((df["status"] == "success").sum()) if total_files else 0
    failed = int((df["status"] == "failed").sum()) if total_files else 0
    total_examples = int(df["n_examples"].fillna(0).sum()) if total_files else 0

    lines = [
        "=== 04_make_marker_examples run summary ===",
        "",
        f"Total files: {total_files}",
        f"Success: {success}",
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


def main() -> None:
    input_files = sorted(cfg.CLEAN_H5AD_DIR.glob("*.h5ad"))
    if not input_files:
        raise FileNotFoundError(f"No cleaned h5ad files found in {cfg.CLEAN_H5AD_DIR}")

    out_jsonl = Path(cfg.MARKER_EXAMPLES_JSONL)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = out_jsonl.with_name("marker_examples_manifest.csv")
    summary_path = out_jsonl.with_name("04_make_marker_examples_run_summary.txt")

    all_records: List[dict] = []
    manifest_rows: List[Dict[str, Any]] = []

    logging.info("Start building marker examples.")
    logging.info("Input dir=%s", cfg.CLEAN_H5AD_DIR)
    logging.info("Output jsonl=%s", out_jsonl)
    logging.info("Found %d cleaned h5ad files", len(input_files))
    logging.info("Marker group column=%s", MARKER_GROUP_COL)

    for h5ad_path in tqdm(input_files, desc="Make markers", unit="file"):
        t0 = time.time()
        try:
            logging.info("Building markers for %s", h5ad_path.name)
            adata = ad.read_h5ad(h5ad_path)

            records = extract_markers_for_dataset(adata)
            all_records.extend(records)

            manifest_rows.append({
                "file_name": h5ad_path.name,
                "dataset_id": adata.uns.get("source_meta", {}).get("dataset_id", h5ad_path.stem),
                "status": "success",
                "n_examples": len(records),
                "elapsed_sec": round(time.time() - t0, 2),
                "error": "",
            })

            logging.info("Generated %d marker examples for %s", len(records), h5ad_path.name)

            del adata
            gc.collect()

        except Exception as e:
            logging.exception("Failed building markers for %s", h5ad_path.name)
            manifest_rows.append({
                "file_name": h5ad_path.name,
                "dataset_id": h5ad_path.stem,
                "status": "failed",
                "n_examples": 0,
                "elapsed_sec": round(time.time() - t0, 2),
                "error": repr(e),
            })

        if WRITE_JSONL_EVERY_FILE:
            save_jsonl(all_records, out_jsonl)
            save_manifest(manifest_rows, manifest_path)

    if not WRITE_JSONL_EVERY_FILE:
        save_jsonl(all_records, out_jsonl)
        save_manifest(manifest_rows, manifest_path)

    save_run_summary(manifest_rows, summary_path, out_jsonl)
    logging.info("Saved %d marker examples to %s", len(all_records), out_jsonl)


if __name__ == "__main__":
    main()

# python -u 04_make_marker_examples.py 2>&1 | tee data/meta/04_make_marker_examples.log