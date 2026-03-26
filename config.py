from __future__ import annotations

from pathlib import Path

# ===== 基础路径 =====
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
META_DIR = DATA_DIR / "meta"
RAW_H5AD_DIR = DATA_DIR / "raw_h5ad"
CLEAN_H5AD_DIR = DATA_DIR / "clean_h5ad"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
SFT_DIR = DATA_DIR / "sft"
SPLIT_DIR = DATA_DIR / "splits"

for p in [META_DIR, RAW_H5AD_DIR, CLEAN_H5AD_DIR, INTERMEDIATE_DIR, SFT_DIR, SPLIT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ===== Census 配置 =====
CENSUS_VERSION = "2025-11-08"   # 先用 stable，正式实验时建议固定成具体版本
ORGANISM = "Homo sapiens"   # 第一版先做人类
ORGANISM_KEY = "homo_sapiens"

# ===== 初始目标组织 =====
# 建议第一版先选这些容易出现经典 marker 的组织
TARGET_TISSUES = [
    "blood",
    "lung",
    "liver",
    "intestine",
]

# ===== 数据集筛选规则 =====
MIN_DATASET_CELLS = 3_000
MAX_DATASET_CELLS = 120_000
MIN_UNIQUE_CELL_TYPES = 6
AUTO_SELECT_TOPK_PER_TISSUE = 2  # 自动给每个组织预选前几个数据集，后续你再人工确认

# ===== 清洗规则 =====
MIN_CELLS_PER_LABEL = 50
MIN_GENES_DETECTED_IN_CELLS = 10
MAX_CELLS_PER_LABEL_FOR_DE = 2_000  # 为了省内存，可对每个标签下采样到这个数量做差异分析
RANDOM_SEED = 42

# ===== marker 构建规则 =====
TOP_K_MARKERS = 10
MAX_CANDIDATE_MARKERS = 50

# 近期 benchmark 明确提到 top 10 marker genes 是较好的折中，
# 并且会移除常见高丰度但信息量低的基因，例如 MALAT1、NEAT1、XIST 等。:contentReference[oaicite:5]{index=5}
UNINFORMATIVE_GENES = {
    "MALAT1", "NEAT1", "XIST", "KCNQ1OT1", "RPPH1",
    "RN7SL1", "RMRP", "SNHG1", "MIAT", "H19"
}

# 一些常见的“不要拿来当 marker”的模式
BAD_GENE_PREFIXES = (
    "MT-", "RPL", "RPS", "HB", "HBA", "HBB", "IGJ"
)

# ===== 标签过滤规则 =====
AMBIGUOUS_LABEL_KEYWORDS = [
    "unknown",
    "unassigned",
    "ambiguous",
    "doublet",
    "multiplet",
    "low quality",
    "low-quality",
    "debris",
    "artifact",
]

# ===== SFT 配置 =====
SYSTEM_PROMPT = (
    "You are a transcriptomics assistant specialized in single-cell RNA-seq "
    "cell type annotation. Use the ranked marker genes and biological context "
    "to infer the most likely cell type. Be concise, structured, and avoid overclaiming."
)

# ===== 文件路径 =====
CANDIDATE_DATASETS_CSV = META_DIR / "candidate_datasets.csv"
SELECTED_DATASETS_CSV = META_DIR / "selected_datasets.csv"

RAW_MANIFEST_CSV = META_DIR / "raw_export_manifest.csv"
CLEAN_MANIFEST_CSV = META_DIR / "clean_manifest.csv"
MARKER_EXAMPLES_JSONL = INTERMEDIATE_DIR / "marker_examples.jsonl"

SFT_RECORDS_FULL_JSONL = SFT_DIR / "sft_records_full.jsonl"
SFT_MESSAGES_JSONL = SFT_DIR / "sft_messages.jsonl"
SFT_MESSAGES_NO_THINK_JSONL = SFT_DIR / "sft_messages_no_think.jsonl"