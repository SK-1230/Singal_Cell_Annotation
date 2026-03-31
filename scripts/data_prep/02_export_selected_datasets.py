from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import cellxgene_census
import pandas as pd
from tqdm import tqdm

import data_prep_config as cfg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


OBS_COLUMNS = [
    "dataset_id",
    "assay",
    "assay_ontology_term_id",
    "cell_type",
    "cell_type_ontology_term_id",
    "development_stage",
    "development_stage_ontology_term_id",
    "disease",
    "disease_ontology_term_id",
    "donor_id",
    "is_primary_data",
    "self_reported_ethnicity",
    "self_reported_ethnicity_ontology_term_id",
    "sex",
    "sex_ontology_term_id",
    "suspension_type",
    "tissue",
    "tissue_ontology_term_id",
    "tissue_general",
    "tissue_general_ontology_term_id",
]


def resolve_census_version() -> str:
    return str(cfg.CENSUS_VERSION).strip()


def safe_int(x: Any) -> int | None:
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def ensure_parent_dirs() -> None:
    cfg.RAW_H5AD_DIR.mkdir(parents=True, exist_ok=True)
    Path(cfg.RAW_MANIFEST_CSV).parent.mkdir(parents=True, exist_ok=True)


def build_filter_expr(dataset_id: str) -> str:
    return f"dataset_id == '{dataset_id}' and is_primary_data == True"


def build_source_meta(row: pd.Series) -> Dict[str, Any]:
    return {
        "dataset_id": str(row.get("dataset_id", "")),
        "collection_name": str(row.get("collection_name", "")),
        "collection_doi": str(row.get("collection_doi", "")),
        "dataset_title": str(row.get("dataset_title", "")),
        "dataset_total_cell_count": safe_int(row.get("dataset_total_cell_count")),
        "cell_count": safe_int(row.get("cell_count")),
        "unique_cell_types": safe_int(row.get("unique_cell_types")),
        "tissues": str(row.get("tissues", "")),
        "diseases": str(row.get("diseases", "")),
        "census_version": str(cfg.CENSUS_VERSION),
        "organism": str(cfg.ORGANISM),
    }


def export_one_dataset(
    census,
    row: pd.Series,
) -> Dict[str, Any]:
    dataset_id = str(row["dataset_id"])
    dataset_title = str(row.get("dataset_title", ""))
    out_path = cfg.RAW_H5AD_DIR / f"{dataset_id}.h5ad"
    meta_json_path = cfg.RAW_H5AD_DIR / f"{dataset_id}.source_meta.json"

    record: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_title": dataset_title,
        "output_path": str(out_path),
        "status": None,
        "n_obs": None,
        "n_vars": None,
        "elapsed_sec": None,
        "attempts": 0,
        "error": "",
    }

    if out_path.exists():
        logging.info("Skip existing file: %s", out_path.name)
        record["status"] = "exists"
        return record

    filter_expr = build_filter_expr(dataset_id)
    source_meta = build_source_meta(row)

    last_error = None
    t0 = time.time()

    for attempt in range(1, cfg.EXPORT_MAX_RETRIES + 1):
        record["attempts"] = attempt
        try:
            logging.info(
                "[%s] attempt %d/%d | start export | title=%s",
                dataset_id, attempt, cfg.EXPORT_MAX_RETRIES, dataset_title
            )

            read_t0 = time.time()
            adata = cellxgene_census.get_anndata(
                census=census,
                organism=cfg.ORGANISM,
                obs_value_filter=filter_expr,
                column_names={"obs": OBS_COLUMNS},
            )
            logging.info(
                "[%s] get_anndata finished in %.2fs | cells=%d genes=%d",
                dataset_id,
                time.time() - read_t0,
                adata.n_obs,
                adata.n_vars,
            )

            adata.uns["source_meta"] = source_meta

            if cfg.EXPORT_SAVE_SOURCE_META_JSON:
                meta_json_path.write_text(
                    json.dumps(source_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            write_t0 = time.time()
            if cfg.EXPORT_WRITE_COMPRESSION:
                adata.write_h5ad(out_path, compression=cfg.EXPORT_WRITE_COMPRESSION)
            else:
                adata.write_h5ad(out_path)
            logging.info(
                "[%s] write_h5ad finished in %.2fs -> %s",
                dataset_id,
                time.time() - write_t0,
                out_path.name,
            )

            record["status"] = "exported"
            record["n_obs"] = int(adata.n_obs)
            record["n_vars"] = int(adata.n_vars)
            record["elapsed_sec"] = round(time.time() - t0, 2)

            # 主动释放内存
            del adata
            gc.collect()

            logging.info(
                "[%s] export success | total elapsed=%.2fs",
                dataset_id,
                time.time() - t0,
            )
            return record

        except Exception as e:
            last_error = e
            record["error"] = repr(e)
            logging.exception("[%s] export failed on attempt %d", dataset_id, attempt)

            # 失败后清理可能的半成品
            if out_path.exists():
                try:
                    out_path.unlink()
                    logging.warning("[%s] removed incomplete file: %s", dataset_id, out_path.name)
                except Exception:
                    logging.warning("[%s] failed to remove incomplete file", dataset_id)

            gc.collect()

            if attempt < cfg.EXPORT_MAX_RETRIES:
                logging.info(
                    "[%s] will retry after %ds...",
                    dataset_id,
                    cfg.EXPORT_RETRY_SLEEP_SECONDS,
                )
                time.sleep(cfg.EXPORT_RETRY_SLEEP_SECONDS)

    record["status"] = "failed"
    record["elapsed_sec"] = round(time.time() - t0, 2)
    record["error"] = repr(last_error) if last_error is not None else "unknown error"
    return record


def load_selected_datasets() -> pd.DataFrame:
    if not cfg.SELECTED_DATASETS_CSV.exists():
        raise FileNotFoundError(
            f"{cfg.SELECTED_DATASETS_CSV} not found. Please run 01_list_candidate_datasets.py first."
        )

    selected = pd.read_csv(cfg.SELECTED_DATASETS_CSV)

    if "use" in selected.columns:
        selected = selected[selected["use"] == 1].copy()

    if selected.empty:
        raise ValueError("No selected datasets found. Please mark use=1 in selected_datasets.csv")

    if "dataset_id" not in selected.columns:
        raise ValueError("selected_datasets.csv must contain 'dataset_id' column")

    selected = selected.drop_duplicates(subset=["dataset_id"]).reset_index(drop=True)
    return selected


def save_manifest(results: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(results)
    df.to_csv(cfg.RAW_MANIFEST_CSV, index=False)
    logging.info("Saved manifest to %s | rows=%d", cfg.RAW_MANIFEST_CSV, len(df))


def save_run_summary(results: List[Dict[str, Any]]) -> None:
    summary_path = Path(cfg.RAW_MANIFEST_CSV).with_name("02_export_selected_datasets_run_summary.txt")
    df = pd.DataFrame(results)

    total = len(df)
    exported = int((df["status"] == "exported").sum()) if total else 0
    exists = int((df["status"] == "exists").sum()) if total else 0
    failed = int((df["status"] == "failed").sum()) if total else 0

    lines = [
        "=== 02_export_selected_datasets run summary ===",
        "",
        f"Total selected datasets: {total}",
        f"Exported: {exported}",
        f"Already existed: {exists}",
        f"Failed: {failed}",
        "",
        f"Manifest: {cfg.RAW_MANIFEST_CSV}",
        f"Output dir: {cfg.RAW_H5AD_DIR}",
        "",
    ]

    if failed > 0:
        lines.append("Failed datasets:")
        failed_df = df[df["status"] == "failed"]
        for _, row in failed_df.iterrows():
            lines.append(
                f"- {row.get('dataset_id', '')} | attempts={row.get('attempts', '')} | error={row.get('error', '')}"
            )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved run summary to %s", summary_path)


def main() -> None:
    overall_t0 = time.time()
    ensure_parent_dirs()

    version = resolve_census_version()
    selected = load_selected_datasets()

    logging.info("Start exporting selected datasets from CELLxGENE Census.")
    logging.info("Using census_version=%s", version)
    logging.info("Using organism=%s", cfg.ORGANISM)
    logging.info("Selected dataset count=%d", len(selected))
    logging.info("Output dir=%s", cfg.RAW_H5AD_DIR)

    results: List[Dict[str, Any]] = []

    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="Export datasets",
        unit="dataset",
    ):
        # 每个数据集使用独立的 SOMA 连接，避免共享长连接在大文件下载时 TileDB C++ 层超时
        # 导致进程被直接 abort（Python 的 try/except 无法捕获 C++ std::abort/SIGSEGV）
        dataset_id = str(row["dataset_id"])
        result = None
        for soma_attempt in range(1, cfg.EXPORT_MAX_RETRIES + 1):
            try:
                logging.info("open_soma() attempt %d for dataset_id=%s", soma_attempt, dataset_id)
                with cellxgene_census.open_soma(census_version=version) as census:
                    result = export_one_dataset(census=census, row=row)
                break
            except Exception as e:
                logging.exception("open_soma() failed on attempt %d for %s", soma_attempt, dataset_id)
                if soma_attempt < cfg.EXPORT_MAX_RETRIES:
                    logging.info("will retry open_soma() after %ds...", cfg.EXPORT_RETRY_SLEEP_SECONDS)
                    time.sleep(cfg.EXPORT_RETRY_SLEEP_SECONDS)
                else:
                    result = {
                        "dataset_id": dataset_id,
                        "dataset_title": str(row.get("dataset_title", "")),
                        "output_path": str(cfg.RAW_H5AD_DIR / f"{dataset_id}.h5ad"),
                        "status": "failed",
                        "n_obs": None, "n_vars": None, "elapsed_sec": None,
                        "attempts": soma_attempt,
                        "error": repr(e),
                    }
        results.append(result)

        # 每完成一个就写一次 manifest，防止中途断掉后完全没记录
        save_manifest(results)

    save_run_summary(results)

    total_elapsed = time.time() - overall_t0
    logging.info("All done. Total elapsed = %.2fs", total_elapsed)

    df = pd.DataFrame(results)
    exported = int((df["status"] == "exported").sum()) if not df.empty else 0
    exists = int((df["status"] == "exists").sum()) if not df.empty else 0
    failed = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nExport summary:")
    print(f"- exported: {exported}")
    print(f"- exists:   {exists}")
    print(f"- failed:   {failed}")
    print(f"- manifest: {cfg.RAW_MANIFEST_CSV}")
    print(f"- out dir:  {cfg.RAW_H5AD_DIR}")


if __name__ == "__main__":
    main()


# python -u scripts/data_prep/02_export_selected_datasets.py 2>&1 | tee data/meta/02_export_selected_datasets.log