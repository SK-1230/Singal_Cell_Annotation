from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import cellxgene_census
from tqdm import tqdm

import data_prep_config as cfg

# Phase-A: metadata guardrails
try:
    import sys as _sys
    _sys.path.insert(0, str(cfg.PROJECT_DIR / "src"))
    from sca.data.curation_rules import passes_metadata_guardrails, score_reference_preference
    _CURATION_AVAILABLE = True
except ImportError:
    _CURATION_AVAILABLE = False


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


def count_tokens_in_semicolon_field(value: str) -> int:
    """
    对类似 "blood;lung" / "normal;tumor" 的字段计数。
    空值返回 0。
    """
    if pd.isna(value):
        return 0
    tokens = [x.strip() for x in str(value).split(";") if x.strip()]
    return len(tokens)


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

    logging.info("[%s] running concat() — may take several minutes for large tissues (blood ~23min) ...", tissue_general)
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

    # 衍生辅助字段，供自动打分使用
    filtered["n_tissues"] = filtered["tissues"].map(count_tokens_in_semicolon_field)
    filtered["n_diseases"] = filtered["diseases"].map(count_tokens_in_semicolon_field)

    # Phase-A: apply metadata guardrails
    if _CURATION_AVAILABLE:
        mask = filtered.apply(lambda row: passes_metadata_guardrails(row, cfg), axis=1)
        n_before = len(filtered)
        filtered = filtered[mask].copy()
        logging.info(
            "Metadata guardrails: %d -> %d datasets (dropped %d)",
            n_before, len(filtered), n_before - len(filtered),
        )
    else:
        logging.warning("curation_rules not available — skipping guardrail filtering")

    # 先做一个基础排序，便于查看 candidate 表
    filtered = filtered.sort_values(
        by=["tissues", "unique_cell_types", "cell_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return filtered


def normalize_series_to_01(s: pd.Series) -> pd.Series:
    """
    将数值列归一化到 [0, 1]。
    若该列所有值完全相同，则返回全 1，避免除零。
    """
    s = s.astype(float)
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series([0.0] * len(s), index=s.index)
    if s_max == s_min:
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - s_min) / (s_max - s_min)


def build_auto_score(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    为候选数据集构建自动打分。

    打分思路尽量不依赖生物学人工判断，而只依赖 metadata：
    1) unique_cell_types 越高越好
    2) cell_count 越高一般越稳定，但不追求极端超大；这里简单做归一化
    3) tissues 越单一越好（n_tissues 小）
    4) diseases 越单一越好（n_diseases 小）
    """
    df = filtered.copy()

    df["score_celltype"] = normalize_series_to_01(df["unique_cell_types"])
    df["score_cells"] = normalize_series_to_01(df["cell_count"])

    # 单 tissue / 低 disease 混杂偏好
    if cfg.AUTO_PREFER_SINGLE_TISSUE:
        # n_tissues 越少越好，因此用反向分数
        df["score_single_tissue"] = 1.0 - normalize_series_to_01(df["n_tissues"])
    else:
        df["score_single_tissue"] = 0.0

    if cfg.AUTO_PREFER_LOW_DISEASE_MIX:
        # n_diseases 越少越好，因此用反向分数
        df["score_low_disease_mix"] = 1.0 - normalize_series_to_01(df["n_diseases"])
    else:
        df["score_low_disease_mix"] = 0.0

    # Reference/atlas preference
    if _CURATION_AVAILABLE:
        df["score_reference"] = df.apply(lambda row: score_reference_preference(row, cfg), axis=1)
    else:
        df["score_reference"] = 0.0

    df["auto_score"] = (
        cfg.AUTO_SCORE_WEIGHT_CELLTYPE * df["score_celltype"]
        + cfg.AUTO_SCORE_WEIGHT_CELLS * df["score_cells"]
        + cfg.AUTO_SCORE_WEIGHT_SINGLE_TISSUE * df["score_single_tissue"]
        + cfg.AUTO_SCORE_WEIGHT_LOW_DISEASE_MIX * df["score_low_disease_mix"]
        + getattr(cfg, "AUTO_SCORE_WEIGHT_REFERENCE", 0.5) * df["score_reference"]
    )

    # 再按自动分数排序
    df = df.sort_values(
        by=["auto_score", "unique_cell_types", "cell_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def build_preselected_template(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    旧逻辑：自动生成一个很小的预选模板，供人工确认。
    仅在 cfg.SELECT_MODE == "manual_template" 时使用。
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
        subset["selected_reason"] = f"manual_template_top{cfg.AUTO_SELECT_TOPK_PER_TISSUE}_for_{tissue}"
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
        selected["selected_reason"] = "fallback_manual_template_selection"

    return selected


def build_auto_selected(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    新逻辑：全自动生成最终 selected 集合，直接给 02 使用。

    选择策略：
    1) 对所有 candidate 做自动打分
    2) 对每个目标 tissue，各自选 AUTO_SELECTED_PER_TISSUE 个高分 dataset
    3) 合并后按 dataset_id 去重
    4) 若数量仍低于 AUTO_SELECTED_MIN_DATASETS，则从全局剩余高分候选继续补齐
    5) 若数量高于 AUTO_SELECTED_MAX_DATASETS，则保留全局高分前若干个
    """
    scored = build_auto_score(filtered)

    selected_rows = []

    # 先按 tissue 做“均衡选取”
    for tissue in tqdm(cfg.TARGET_TISSUES, desc="Build auto selected", unit="tissue"):
        subset = scored[scored["tissues"].str.contains(tissue, na=False)].copy()

        if subset.empty:
            logging.warning("No scored candidate found for tissue=%s", tissue)
            continue

        subset = subset.head(cfg.AUTO_SELECTED_PER_TISSUE).copy()
        subset["use"] = 1
        subset["selected_reason"] = f"auto_balanced_top{cfg.AUTO_SELECTED_PER_TISSUE}_for_{tissue}"
        subset["matched_tissue"] = tissue
        selected_rows.append(subset)

    if selected_rows:
        selected = pd.concat(selected_rows, ignore_index=True)
    else:
        logging.warning("No per-tissue auto-selected rows found. Falling back to global top candidates.")
        selected = scored.head(cfg.AUTO_SELECTED_MIN_DATASETS).copy()
        selected["use"] = 1
        selected["selected_reason"] = "fallback_auto_balanced_global_top"
        selected["matched_tissue"] = ""

    # 对同一 dataset 去重：保留 auto_score 更高的记录
    selected = (
        selected.sort_values(by=["auto_score", "unique_cell_types", "cell_count"], ascending=[False, False, False])
        .drop_duplicates(subset=["dataset_id"])
        .reset_index(drop=True)
    )

    # 如果需要保存匹配到的所有 tissue，就做一个聚合说明
    if cfg.AUTO_SAVE_MATCHED_TISSUES and "matched_tissue" in selected.columns:
        # 先重新从初始 selected_rows 聚合 matched_tissue
        if selected_rows:
            tmp = pd.concat(selected_rows, ignore_index=True)
            agg = (
                tmp.groupby("dataset_id")["matched_tissue"]
                .agg(lambda x: ";".join(sorted(set(map(str, x)))))
                .reset_index()
                .rename(columns={"matched_tissue": "matched_tissues"})
            )
            selected = selected.merge(agg, on="dataset_id", how="left")
        else:
            selected["matched_tissues"] = ""

    selected_ids = set(selected["dataset_id"].tolist())

    # 若数量不足，则从剩余全局高分候选补齐
    if len(selected) < cfg.AUTO_SELECTED_MIN_DATASETS:
        need = cfg.AUTO_SELECTED_MIN_DATASETS - len(selected)
        remaining = scored[~scored["dataset_id"].isin(selected_ids)].copy()
        if not remaining.empty:
            extra = remaining.head(need).copy()
            extra["use"] = 1
            extra["selected_reason"] = "auto_balanced_global_fill"
            extra["matched_tissue"] = ""
            if cfg.AUTO_SAVE_MATCHED_TISSUES:
                extra["matched_tissues"] = ""
            selected = pd.concat([selected, extra], ignore_index=True)
            selected = (
                selected.sort_values(by=["auto_score", "unique_cell_types", "cell_count"], ascending=[False, False, False])
                .drop_duplicates(subset=["dataset_id"])
                .reset_index(drop=True)
            )

    # 若数量过多，则按全局得分裁剪
    if len(selected) > cfg.AUTO_SELECTED_MAX_DATASETS:
        selected = (
            selected.sort_values(by=["auto_score", "unique_cell_types", "cell_count"], ascending=[False, False, False])
            .head(cfg.AUTO_SELECTED_MAX_DATASETS)
            .reset_index(drop=True)
        )

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
    summary_lines.append(f"Select mode: {cfg.SELECT_MODE}")
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
    summary_lines.append(f"Selected rows: {len(selected)}")
    summary_lines.append("")
    summary_lines.append(f"candidate_datasets.csv: {cfg.CANDIDATE_DATASETS_CSV}")
    summary_lines.append(f"selected_datasets.csv: {cfg.SELECTED_DATASETS_CSV}")

    if cfg.SELECT_MODE == "auto_balanced":
        summary_lines.append("")
        summary_lines.append("[Auto-balanced settings]")
        summary_lines.append(f"AUTO_SELECTED_PER_TISSUE = {cfg.AUTO_SELECTED_PER_TISSUE}")
        summary_lines.append(f"AUTO_SELECTED_MIN_DATASETS = {cfg.AUTO_SELECTED_MIN_DATASETS}")
        summary_lines.append(f"AUTO_SELECTED_MAX_DATASETS = {cfg.AUTO_SELECTED_MAX_DATASETS}")

    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    overall_t0 = time.time()
    logging.info("Start querying candidate datasets from CELLxGENE Census.")

    version = resolve_census_version()
    logging.info("Using census_version=%s", version)
    logging.info("Using organism_key=%s", cfg.ORGANISM_KEY)
    logging.info("Target tissues=%s", cfg.TARGET_TISSUES)
    logging.info("Select mode=%s", cfg.SELECT_MODE)

    partial_stats: List[pd.DataFrame] = []
    tissue_row_stats: List[Dict] = []

    with cellxgene_census.open_soma(census_version=version) as census:
        logging.info("open_soma() succeeded")

        for tissue in tqdm(cfg.TARGET_TISSUES, desc="Query tissues", unit="tissue"):
            t0 = time.time()
            try:
                obs = fetch_tissue_obs(
                    census=census,
                    tissue_general=tissue,
                    save_obs_preview=True,
                )
            except Exception as e:
                logging.error(
                    "[%s] fetch_tissue_obs failed, skipping tissue. error=%r",
                    tissue, e,
                )
                tissue_row_stats.append(
                    {"tissue": tissue, "rows": 0, "elapsed_sec": time.time() - t0}
                )
                continue

            tissue_dataset_stats = summarize_one_tissue_obs(obs, tissue)
            partial_stats.append(tissue_dataset_stats)

            tissue_row_stats.append(
                {
                    "tissue": tissue,
                    "rows": int(len(obs)),
                    "elapsed_sec": time.time() - t0,
                }
            )

            logging.info("[%s] building per-tissue dataset summary...", tissue)
            per_tissue_summary_path = cfg.META_DIR / f"dataset_stats_{tissue}.csv"
            tissue_dataset_stats.to_csv(per_tissue_summary_path, index=False)
            logging.info("[%s] saved per-tissue dataset summary to %s", tissue, per_tissue_summary_path)

            del obs
            del tissue_dataset_stats

        logging.info("Merging per-tissue dataset statistics...")
        dataset_stats = merge_partial_stats(partial_stats)
        dataset_stats_path = cfg.META_DIR / "dataset_stats_merged.csv"
        dataset_stats.to_csv(dataset_stats_path, index=False)
        logging.info("Saved merged dataset stats to %s", dataset_stats_path)

        # 兼容原项目中的 obs_preview 位置；这里保存轻量汇总，不保存超大 obs_all
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

    logging.info("Building selected_datasets.csv ...")
    if cfg.SELECT_MODE == "manual_template":
        selected = build_preselected_template(filtered)
    elif cfg.SELECT_MODE == "auto_balanced":
        selected = build_auto_selected(filtered)
    else:
        raise ValueError(f"Unsupported SELECT_MODE: {cfg.SELECT_MODE}")

    selected.to_csv(cfg.SELECTED_DATASETS_CSV, index=False)
    logging.info("Saved selected datasets to %s | rows=%d", cfg.SELECTED_DATASETS_CSV, len(selected))

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
    print(f"1) Candidate pool: {cfg.CANDIDATE_DATASETS_CSV}")
    print(f"2) Selected set:   {cfg.SELECTED_DATASETS_CSV}")
    if cfg.SELECT_MODE == "auto_balanced":
        print("   - This selected file is already the final auto-selected set for downstream export.")
        print("   - You can directly use all rows with use=1 in step 02.")
    else:
        print("   - Edit selected_datasets.csv manually if needed.")
        print("   - keep rows you want")
        print("   - set use=1 for datasets to export")
        print("   - delete or set use=0 for datasets you do not want")


if __name__ == "__main__":
    main()

# 运行方式：
# cd /data/projects/shuke/code/singal_cell_annotation
# nohup python -u scripts/data_prep/01_list_candidate_datasets.py 2>&1 | tee data/meta/01_list_candidate_datasets.log
'''
2. 执行后的“三部曲”：确认、监控、退出
当你敲下回车后，不要直接关掉窗口，先做以下三件事：

    第一步：记录 PID（进程号）
    执行命令后，终端会弹出一行类似 [1] 3425433 的数字。那个 3425433 就是 PID。建议随手记下，或者通过以下命令查询：

    Bash
    ps -ef | grep 03_clean_and_standardize.py

    第二步：检查日志是否在正常写入
    观察日志内容，确保脚本已经跑起来了：

    Bash
    tail -f data/meta/03_clean_and_standardize.log
    （按 Ctrl + C 可以退出查看日志，但这不会停止后台运行的程序。）

    第三步：安全离场
    虽然 nohup 很强大，但建议先输入 exit 退出登录，再关闭终端窗口，这样最稳妥。
'''