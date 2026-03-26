from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import cellxgene_census
from tqdm import tqdm

import config as cfg


# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# =========================
# 工具函数
# =========================
def resolve_census_version() -> str:
    """
    尽量避免 stable 每次打印提示。
    如果 cfg.CENSUS_VERSION 已经是具体版本字符串，就直接使用。
    """
    version = str(cfg.CENSUS_VERSION).strip()
    return version


def get_obs_columns() -> List[str]:
    """
    只读取当前步骤真正需要的列，减少内存占用。
    """
    return [
        "dataset_id",
        "cell_type",
        "tissue",
        "tissue_general",
        "disease",
        "suspension_type",
        "is_primary_data",
    ]


def build_filter_expr(tissue_general: str) -> str:
    """
    构造 Census 过滤条件。
    """
    return (
        f"tissue_general == '{tissue_general}' "
        f"and is_primary_data == True "
        f"and suspension_type == 'cell'"
    )


def summarize_one_tissue_obs(obs: pd.DataFrame, tissue_query: str) -> pd.DataFrame:
    """
    将单个 tissue 的细胞级 obs 立刻聚合成 dataset 级统计，
    避免把多个 tissue 的超大 obs 全都堆在内存里。
    """
    if obs.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "cell_count",
                "unique_cell_types",
                "tissues",
                "diseases",
                "tissue_general_query",
            ]
        )

    grouped = (
        obs.groupby("dataset_id")
        .agg(
            cell_count=("dataset_id", "size"),
            unique_cell_types=("cell_type", "nunique"),
            tissues=("tissue_general", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
            diseases=("disease", lambda x: ";".join(sorted(set(map(str, x.dropna()))))[:500]),
        )
        .reset_index()
    )
    grouped["tissue_general_query"] = tissue_query
    return grouped


def merge_partial_stats(partial_stats: List[pd.DataFrame]) -> pd.DataFrame:
    """
    将每个 tissue 的 dataset 级统计再汇总成全局 dataset 级统计。
    同一个 dataset 可能跨多个 tissue，因此这里需要二次聚合。
    """
    if not partial_stats:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "cell_count",
                "unique_cell_types",
                "tissues",
                "diseases",
                "tissue_general_query",
            ]
        )

    combined = pd.concat(partial_stats, ignore_index=True)

    final_stats = (
        combined.groupby("dataset_id")
        .agg(
            cell_count=("cell_count", "sum"),
            unique_cell_types=("unique_cell_types", "max"),
            tissues=("tissues", lambda x: ";".join(sorted(set(
                token
                for item in x.dropna()
                for token in str(item).split(";")
                if token
            )))),
            diseases=("diseases", lambda x: ";".join(sorted(set(
                token
                for item in x.dropna()
                for token in str(item).split(";")
                if token
            )))[:500]),
            tissue_general_query=("tissue_general_query", lambda x: ";".join(sorted(set(map(str, x.dropna()))))),
        )
        .reset_index()
    )

    return final_stats


def fetch_tissue_obs(
    census,
    tissue_general: str,
    save_obs_preview: bool = True,
) -> pd.DataFrame:
    """
    拉取某个 tissue_general 的细胞级 obs。
    """
    filter_expr = build_filter_expr(tissue_general)
    cols = get_obs_columns()

    t0 = time.time()
    logging.info("===== Begin tissue: %s =====", tissue_general)
    logging.info("[%s] filter = %s", tissue_general, filter_expr)

    logging.info("[%s] creating reader...", tissue_general)
    reader = census["census_data"][cfg.ORGANISM_KEY].obs.read(
        value_filter=filter_expr,
        column_names=cols,
    )

    t1 = time.time()
    logging.info("[%s] reader created in %.2fs", tissue_general, t1 - t0)

    logging.info("[%s] running concat() ...", tissue_general)
    table = reader.concat()

    t2 = time.time()
    logging.info("[%s] concat() finished in %.2fs", tissue_general, t2 - t1)

    logging.info("[%s] converting to pandas ...", tissue_general)
    obs = table.to_pandas()

    t3 = time.time()
    logging.info(
        "[%s] to_pandas() finished in %.2fs | rows=%d",
        tissue_general,
        t3 - t2,
        len(obs),
    )

    obs["tissue_general_query"] = tissue_general

    if save_obs_preview:
        preview_path = cfg.META_DIR / f"obs_preview_{tissue_general}.csv"
        logging.info("[%s] saving per-tissue preview to %s", tissue_general, preview_path)
        obs.to_csv(preview_path, index=False)

    logging.info("[%s] total fetch done in %.2fs", tissue_general, time.time() - t0)
    logging.info("===== End tissue: %s =====", tissue_general)
    return obs


def load_datasets_table(census) -> pd.DataFrame:
    """
    读取 Census datasets 元数据表。
    """
    t0 = time.time()
    logging.info("Reading census_info['datasets'] ...")
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    logging.info("Loaded datasets table | rows=%d | elapsed=%.2fs", len(datasets), time.time() - t0)
    return datasets


def filter_candidate_datasets(merged: pd.DataFrame) -> pd.DataFrame:
    """
    按 config 中规则过滤数据集。
    """
    filtered = merged[
        (merged["cell_count"] >= cfg.MIN_DATASET_CELLS)
        & (merged["cell_count"] <= cfg.MAX_DATASET_CELLS)
        & (merged["unique_cell_types"] >= cfg.MIN_UNIQUE_CELL_TYPES)
    ].copy()

    filtered = filtered.sort_values(
        by=["tissues", "unique_cell_types", "cell_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return filtered


def build_preselected_template(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    自动生成一个预选模板，供人工确认。
    """
    preselected_rows = []

    for tissue in tqdm(cfg.TARGET_TISSUES, desc="Build selection template", unit="tissue"):
        subset = filtered[
            filtered["tissues"].str.contains(tissue, na=False)
        ].head(cfg.AUTO_SELECT_TOPK_PER_TISSUE).copy()

        if subset.empty:
            logging.warning("No filtered dataset found for tissue=%s", tissue)
            continue

        subset["use"] = 1
        subset["selected_reason"] = f"auto_preselected_for_{tissue}"
        preselected_rows.append(subset)

    if preselected_rows:
        selected = (
            pd.concat(preselected_rows, ignore_index=True)
            .drop_duplicates(subset=["dataset_id"])
            .reset_index(drop=True)
        )
    else:
        logging.warning("No preselected rows found. Falling back to top 5 filtered datasets.")
        selected = filtered.head(5).copy()
        selected["use"] = 1
        selected["selected_reason"] = "fallback_auto_selection"

    return selected


def save_run_summary(
    output_path: Path,
    tissue_row_stats: List[Dict],
    dataset_stats: pd.DataFrame,
    filtered: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """
    保存一份运行摘要，便于服务器上回看。
    """
    rows_df = pd.DataFrame(tissue_row_stats)

    summary_lines = []
    summary_lines.append("=== 01_list_candidate_datasets run summary ===")
    summary_lines.append("")
    summary_lines.append("Target tissues:")
    for tissue in cfg.TARGET_TISSUES:
        summary_lines.append(f"- {tissue}")

    summary_lines.append("")
    summary_lines.append("Per-tissue fetched rows:")
    if rows_df.empty:
        summary_lines.append("No tissue rows fetched.")
    else:
        for _, row in rows_df.iterrows():
            summary_lines.append(
                f"- {row['tissue']}: rows={row['rows']}, elapsed_sec={row['elapsed_sec']:.2f}"
            )

    summary_lines.append("")
    summary_lines.append(f"Dataset stats rows: {len(dataset_stats)}")
    summary_lines.append(f"Filtered candidate rows: {len(filtered)}")
    summary_lines.append(f"Selected template rows: {len(selected)}")
    summary_lines.append("")
    summary_lines.append(f"candidate_datasets.csv: {cfg.CANDIDATE_DATASETS_CSV}")
    summary_lines.append(f"selected_datasets.csv: {cfg.SELECTED_DATASETS_CSV}")

    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    overall_t0 = time.time()
    logging.info("Start querying candidate datasets from CELLxGENE Census.")

    version = resolve_census_version()
    logging.info("Using census_version=%s", version)
    logging.info("Using organism_key=%s", cfg.ORGANISM_KEY)
    logging.info("Target tissues=%s", cfg.TARGET_TISSUES)

    partial_stats: List[pd.DataFrame] = []
    tissue_row_stats: List[Dict] = []

    with cellxgene_census.open_soma(census_version=version) as census:
        logging.info("open_soma() succeeded")

        for tissue in tqdm(cfg.TARGET_TISSUES, desc="Query tissues", unit="tissue"):
            t0 = time.time()
            obs = fetch_tissue_obs(
                census=census,
                tissue_general=tissue,
                save_obs_preview=True,
            )

            # 保存每个 tissue 的 dataset 级统计
            tissue_dataset_stats = summarize_one_tissue_obs(obs, tissue)
            partial_stats.append(tissue_dataset_stats)

            tissue_row_stats.append(
                {
                    "tissue": tissue,
                    "rows": int(len(obs)),
                    "elapsed_sec": time.time() - t0,
                }
            )

            # 主预览文件保留一个汇总版
            logging.info("[%s] building per-tissue dataset summary...", tissue)
            per_tissue_summary_path = cfg.META_DIR / f"dataset_stats_{tissue}.csv"
            tissue_dataset_stats.to_csv(per_tissue_summary_path, index=False)
            logging.info("[%s] saved per-tissue dataset summary to %s", tissue, per_tissue_summary_path)

            # 主动释放大对象引用
            del obs
            del tissue_dataset_stats

        logging.info("Merging per-tissue dataset statistics...")
        dataset_stats = merge_partial_stats(partial_stats)
        dataset_stats_path = cfg.META_DIR / "dataset_stats_merged.csv"
        dataset_stats.to_csv(dataset_stats_path, index=False)
        logging.info("Saved merged dataset stats to %s", dataset_stats_path)

        # 兼容你原来脚本里 obs_preview.csv 的位置，
        # 这里不再保存全量 obs_all，而改成保存 tissue 级抓取统计摘要，避免超大文件和高内存。
        obs_preview_path = cfg.META_DIR / "obs_preview.csv"
        pd.DataFrame(tissue_row_stats).to_csv(obs_preview_path, index=False)
        logging.info("Saved lightweight obs preview summary to %s", obs_preview_path)

        datasets = load_datasets_table(census)

    logging.info("Merging dataset statistics with Census datasets table...")
    merged = dataset_stats.merge(
        datasets[
            [
                "dataset_id",
                "collection_name",
                "collection_doi",
                "dataset_title",
                "dataset_h5ad_path",
                "dataset_total_cell_count",
            ]
        ],
        on="dataset_id",
        how="left",
    )

    merged_path = cfg.META_DIR / "candidate_datasets_merged_before_filter.csv"
    merged.to_csv(merged_path, index=False)
    logging.info("Saved merged pre-filter table to %s", merged_path)

    logging.info(
        "Applying filters: MIN_DATASET_CELLS=%s, MAX_DATASET_CELLS=%s, MIN_UNIQUE_CELL_TYPES=%s",
        cfg.MIN_DATASET_CELLS,
        cfg.MAX_DATASET_CELLS,
        cfg.MIN_UNIQUE_CELL_TYPES,
    )
    filtered = filter_candidate_datasets(merged)
    filtered.to_csv(cfg.CANDIDATE_DATASETS_CSV, index=False)
    logging.info("Saved candidate datasets to %s | rows=%d", cfg.CANDIDATE_DATASETS_CSV, len(filtered))

    logging.info("Building editable selected_datasets template...")
    selected = build_preselected_template(filtered)
    selected.to_csv(cfg.SELECTED_DATASETS_CSV, index=False)
    logging.info("Saved editable selection template to %s | rows=%d", cfg.SELECTED_DATASETS_CSV, len(selected))

    summary_txt = cfg.META_DIR / "01_list_candidate_datasets_run_summary.txt"
    save_run_summary(
        output_path=summary_txt,
        tissue_row_stats=tissue_row_stats,
        dataset_stats=dataset_stats,
        filtered=filtered,
        selected=selected,
    )
    logging.info("Saved run summary to %s", summary_txt)

    logging.info("All done. Total elapsed = %.2fs", time.time() - overall_t0)
    print("\nNext step:")
    print(f"1) Open: {cfg.CANDIDATE_DATASETS_CSV}")
    print(f"2) Edit: {cfg.SELECTED_DATASETS_CSV}")
    print("   - keep rows you want")
    print("   - set use=1 for datasets to export")
    print("   - delete or set use=0 for datasets you do not want")


if __name__ == "__main__":
    main()

# python -u 02_export_selected_datasets.py 2>&1 | tee data/meta/02_export_selected_datasets.log
# python -u 01_list_candidate_datasets.py