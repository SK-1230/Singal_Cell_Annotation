from __future__ import annotations

import sys
from pathlib import Path

# =========================================================
# data_prep_config.py — 数据预处理全流程统一配置
#
# 覆盖脚本：01 ~ 08
# 使用方式：在各脚本中 `import data_prep_config as cfg`
# 路径说明：本文件位于 scripts/data_prep/，PROJECT_DIR 自动指向项目根目录
# =========================================================


# =========================================================
# [共用] 路径配置  ·  被所有脚本使用
# =========================================================

PROJECT_DIR      = Path(__file__).resolve().parents[2]   # .../singal_cell_annotation/
DATA_DIR         = PROJECT_DIR / "data"
META_DIR         = DATA_DIR / "meta"          # CSV 统计表、运行摘要
RAW_H5AD_DIR     = DATA_DIR / "raw_h5ad"      # 02 导出的原始 h5ad
CLEAN_H5AD_DIR   = DATA_DIR / "clean_h5ad"    # 03 清洗后的 h5ad
INTERMEDIATE_DIR = DATA_DIR / "intermediate"  # 04 marker 中间文件
SFT_DIR          = DATA_DIR / "sft"           # 05 SFT 训练文件
SPLIT_DIR        = DATA_DIR / "splits"        # 06 划分后的 train/val/test

# Phase 1 新增目录
RESOURCES_DIR    = PROJECT_DIR / "resources"  # 静态知识资源（ontology/markers/schemas）
KNOWLEDGE_DIR    = DATA_DIR / "knowledge"     # 07/08 构建的运行时知识库

# 确保 src/ 在 sys.path 中，以便脚本直接 import sca.*
_SRC_DIR = PROJECT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

for _p in [META_DIR, RAW_H5AD_DIR, CLEAN_H5AD_DIR, INTERMEDIATE_DIR, SFT_DIR, SPLIT_DIR, KNOWLEDGE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


# =========================================================
# [共用] 输出文件路径  ·  跨脚本引用
# =========================================================

# 01 → 生成，所有通过硬过滤条件的数据集候选池
CANDIDATE_DATASETS_CSV = META_DIR / "candidate_datasets.csv"

# 01 → 生成 / 02 → 读取
# 在自动模式下，它将直接作为"最终自动选集"
SELECTED_DATASETS_CSV  = META_DIR / "selected_datasets.csv"

# 02 → 生成
RAW_MANIFEST_CSV       = META_DIR / "raw_export_manifest.csv"

# 03 → 生成
CLEAN_MANIFEST_CSV     = META_DIR / "clean_manifest.csv"

# 04 → 生成 / 05 → 读取（v1 原始版本，保留不变）
MARKER_EXAMPLES_JSONL  = INTERMEDIATE_DIR / "marker_examples.jsonl"

# 04 v2 → 生成 / 05 v2 → 读取（Phase 1 升级版本）
MARKER_EXAMPLES_V2_JSONL = INTERMEDIATE_DIR / "marker_examples_v2.jsonl"

# 05 → 生成 / 06 → 读取（v1 原始版本，保留不变）
SFT_RECORDS_FULL_JSONL      = SFT_DIR / "sft_records_full.jsonl"
SFT_MESSAGES_JSONL          = SFT_DIR / "sft_messages.jsonl"
SFT_MESSAGES_NO_THINK_JSONL = SFT_DIR / "sft_messages_no_think.jsonl"

# 05 v2 → 生成（Phase 2 升级版本）
SFT_RECORDS_FULL_V2_JSONL      = SFT_DIR / "sft_records_full_v2.jsonl"
SFT_MESSAGES_V2_JSONL          = SFT_DIR / "sft_messages_v2.jsonl"
SFT_MESSAGES_NO_THINK_V2_JSONL = SFT_DIR / "sft_messages_no_think_v2.jsonl"

# 07 → 生成
ONTOLOGY_INDEX_JSONL     = KNOWLEDGE_DIR / "ontology_index.jsonl"
CELL_ONTOLOGY_MIN_JSONL  = RESOURCES_DIR / "ontology" / "cell_ontology_min.jsonl"

# 08 → 生成
TRAIN_MARKER_KB_JSONL    = KNOWLEDGE_DIR / "train_marker_kb.jsonl"
MERGED_MARKER_KB_JSONL   = KNOWLEDGE_DIR / "merged_marker_kb.jsonl"

# ontology 资源文件（Phase 1 新增）
LABEL_ALIASES_TSV     = RESOURCES_DIR / "ontology" / "label_aliases.tsv"
ORGAN_HIERARCHY_TSV   = RESOURCES_DIR / "ontology" / "organ_hierarchy.tsv"
EXTERNAL_MARKER_KB_JSONL = RESOURCES_DIR / "markers" / "external_marker_kb.jsonl"


# =========================================================
# [01_list_candidate_datasets] Census 数据源配置
# =========================================================

# CELLxGENE Census 版本号；建议固定为具体日期，避免每次运行结果不一致
CENSUS_VERSION = "2025-11-08"

# 物种名称（两种格式对应 Census API 的不同接口）
ORGANISM     = "Homo sapiens"   # get_anndata() 中的 organism 参数
ORGANISM_KEY = "homo_sapiens"   # open_soma() 读取 obs 时的键名

# 目标组织列表；只查询 tissue_general 属于以下值的细胞
TARGET_TISSUES = [
    "blood",
    "lung",
    "liver",
    "intestine",
    "kidney",
    "brain",
    "skin",
]


# =========================================================
# [01_list_candidate_datasets] 数据集候选筛选规则
# =========================================================

# 数据集包含的细胞数下限 / 上限
MIN_DATASET_CELLS = 3_000
MAX_DATASET_CELLS = 350_000

# 数据集至少包含的独特细胞类型数量
MIN_UNIQUE_CELL_TYPES = 6

# 旧逻辑：每个组织自动预选 Top-K 数据集，写入 selected_datasets.csv 供人工核对
# 当 SELECT_MODE = "manual_template" 时才会用到
AUTO_SELECT_TOPK_PER_TISSUE = 2


# =========================================================
# [01_list_candidate_datasets] 全自动 selected 策略配置
# =========================================================

# selected 生成模式：
# - "manual_template"：保留原逻辑，只生成小模板供人工确认
# - "auto_balanced"：全自动生成最终 selected 集合，后续 02 直接全部使用
SELECT_MODE = "auto_balanced"

# 自动模式下，每个 tissue 目标选择的数据集数量
# 例如 8 表示对 blood/lung/liver/intestine 各自优先选 8 个左右
AUTO_SELECTED_PER_TISSUE = 12

# 自动模式下，selected 的全局最大 dataset 数
# 如果每个 tissue 的目标数量之和去重后仍超过该值，则只保留得分更高的前若干个
AUTO_SELECTED_MAX_DATASETS = 80

# 自动模式下，全局最少选择的数据集数量
# 如果按 tissue 配额选完后仍不足该数量，会从全局候选中继续补齐
AUTO_SELECTED_MIN_DATASETS = 40

# 是否优先偏好"单 tissue"数据集
AUTO_PREFER_SINGLE_TISSUE = True

# 是否优先偏好"单 disease/低混杂"数据集
AUTO_PREFER_LOW_DISEASE_MIX = True

# 自动打分时，对不同因素的权重
# 说明：
# - CELLTYPE：优先保留细胞类型更丰富的数据集
# - CELLS：偏好中等偏大的数据集，但不会无限追求超大规模
# - SINGLE_TISSUE：偏好组织更单一的数据集
# - LOW_DISEASE_MIX：偏好 disease 混杂更少的数据集
AUTO_SCORE_WEIGHT_CELLTYPE = 1.0
AUTO_SCORE_WEIGHT_CELLS = 0.6
AUTO_SCORE_WEIGHT_SINGLE_TISSUE = 0.8
AUTO_SCORE_WEIGHT_LOW_DISEASE_MIX = 0.8

# 自动模式下，如果一个 dataset 同时命中多个 tissue，
# selected_reason 中是否记录所有命中的 tissue
AUTO_SAVE_MATCHED_TISSUES = True


# =========================================================
# [02_export_selected_datasets] 数据集导出配置
# =========================================================

# 单个数据集下载/导出失败时的最大重试次数
EXPORT_MAX_RETRIES = 3

# 相邻两次重试之间的等待时间（秒）
EXPORT_RETRY_SLEEP_SECONDS = 8

# h5ad 写入压缩格式；None = 不压缩（速度快），"gzip" = 压缩（节省磁盘）
EXPORT_WRITE_COMPRESSION = None

# 是否在 raw_h5ad/ 同目录额外保存一份 <dataset_id>.source_meta.json
EXPORT_SAVE_SOURCE_META_JSON = True


# =========================================================
# [03_clean_and_standardize] 清洗规则
# =========================================================

# 单个 raw h5ad 文件大小上限（字节）；超过此值直接跳过，不尝试读取
# 目的：避免 25 GB+ 的超大文件把内存撑爆或让程序长时间卡住
# 8 GB = 8 * 1024 ** 3；如需放宽可改为 16 * 1024 ** 3，设为 None 则不限制
CLEAN_MAX_RAW_FILE_BYTES = 8 * 1024 ** 3

# 每个 cell_type 标签至少需要多少个细胞；低于此数的标签整体丢弃
MIN_CELLS_PER_LABEL = 50

# 基因至少在多少个细胞中有表达；低于此阈值的基因过滤掉
MIN_GENES_DETECTED_IN_CELLS = 10

# h5ad 写入压缩格式（同 EXPORT_WRITE_COMPRESSION，可单独控制）
CLEAN_WRITE_COMPRESSION = None

# True = 每完成一个文件就写一次 manifest（防止中途中断丢失进度）
CLEAN_SAVE_MANIFEST_EVERY_FILE = True

# 包含以下关键词的 cell_type 标签视为模糊/无效，过滤掉
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


# =========================================================
# [04_make_marker_examples] Marker 提取配置
# =========================================================

# 差异分析分组所用的 obs 列名（优先使用清洗后的标签）
MARKER_GROUP_COL = "cell_type_clean"

# 差异分析时每个标签最多保留的细胞数（下采样，防止超大类压制计算）
MAX_CELLS_PER_LABEL_FOR_DE = 2_000

# rank_genes_groups 计算时最多保留的候选基因数
MAX_CANDIDATE_MARKERS = 50

# 最终写入训练样本的 Top-K marker 基因数（正向 marker）
TOP_K_MARKERS = 10

# v2：负向 marker（低表达基因）保留数量
TOP_K_NEGATIVE_MARKERS = 5

# 每个标签至少需要满足数量的有效 marker 基因，否则跳过该标签
MARKER_MIN_TOP_GENES = 5

# marker 阶段对基因再次过滤：基因至少在多少个细胞中被检测到
MARKER_MIN_GENE_CELLS = 5

# True = 每处理完一个 h5ad 文件就写一次 JSONL（防中断）
MARKER_WRITE_JSONL_EVERY_FILE = True

# Phase 1 v2：差异分析方法（t-test | wilcoxon）
# 改为 wilcoxon 更稳健，对非正态分布数据更合适
MARKER_DE_METHOD = "wilcoxon"

# Phase 1 v2：标记低细胞数的阈值（用于 hardness_flags.low_cells）
MARKER_LOW_CELLS_THRESHOLD = 100

# Phase 1 v2：标记低 marker 质量分的阈值（用于 hardness_flags.low_marker_quality）
MARKER_LOW_QUALITY_SCORE_THRESHOLD = 0.3

# Phase 1 v2：稀有标签判断阈值（数据集内该标签占比低于此值）
MARKER_RARE_LABEL_FRACTION_THRESHOLD = 0.02

# 黑名单：高丰度但低信息量的基因，不作为 marker 输出
UNINFORMATIVE_GENES = {
    "MALAT1", "NEAT1", "XIST", "KCNQ1OT1", "RPPH1",
    "RN7SL1", "RMRP", "SNHG1", "MIAT", "H19",
}

# 黑名单：基因名前缀黑名单（线粒体、核糖体、血红蛋白等）
BAD_GENE_PREFIXES = (
    "MT-", "RPL", "RPS", "HB", "HBA", "HBB", "IGJ",
)

# 为了测试代码，临时加一个设定
# RUN_MARKER_V2 = False


# =========================================================
# [05_make_sft_jsonl] SFT 训练数据构建配置
# =========================================================

# 写入每条训练样本的系统提示词
SYSTEM_PROMPT = (
    "You are a transcriptomics assistant specialized in single-cell RNA-seq "
    "cell type annotation. Use the ranked marker genes and biological context "
    "to infer the most likely cell type. Be concise, structured, and avoid overclaiming."
)


# =========================================================
# [06_split_and_validate] 数据集划分配置
# =========================================================

# 随机种子（影响 dataset_id 级别的划分结果，固定后可复现）
RANDOM_SEED = 42

# train / val / test 划分比例（按 dataset_id 级别划分，防止数据泄漏）
# 三者之和必须等于 1.0
#
# 对当前"自动 selected + 中等规模 dataset 数量"的场景，
# 这里建议 test 稍微高一点，便于拿到更稳定的最终评估；
# val 可以稍小，因为在 dataset 还不算特别多时，过大的 val 会稀释 train。
SPLIT_TRAIN_RATIO = 0.75
SPLIT_VAL_RATIO   = 0.10
SPLIT_TEST_RATIO  = 0.15

# token 长度校验时使用的本地模型路径或 HuggingFace/ModelScope 模型名
# 仅在运行 `python 06_split_and_validate_v2.py --check-tokens` 时生效
SPLIT_TOKEN_CHECK_MODEL = "Qwen/Qwen3-8B"


# =========================================================
# [06_split_and_validate_v2] V2 划分增强配置
# =========================================================

# 是否启用"按 dataset 主 tissue_general 分层切分"
#
# 建议：
# - 当 01 自动 selected 出来的 dataset 总数 < 30 时：False
# - 当 dataset 总数 >= 30 且各 tissue 覆盖比较均衡时：True
#
# 你现在先设成 False 更稳妥，优先保证能稳定切出独立 test。
SPLIT_V2_USE_MAIN_TISSUE_STRATIFY = False

# 当 dataset 总数较少时，是否允许从 train records 中构造 pseudo-val
# 作用：
# - 当独立 val dataset 不足或为空时，至少提供一个训练过程中的监控集
# 注意：
# - pseudo-val 不是严格的独立验证集，因为它仍来自 train datasets
SPLIT_V2_ENABLE_PSEUDO_VAL = True

# pseudo-val 从 train_full 中抽样的比例
# 当前建议 0.08 ~ 0.10，避免抽走过多训练样本
SPLIT_V2_PSEUDO_VAL_RATIO = 0.08

# pseudo-val 的最小样本数
SPLIT_V2_MIN_PSEUDO_VAL_EXAMPLES = 2

# pseudo-val 的最大样本数
# 对你当前任务而言，过大的 pseudo-val 没必要，先控制在较小规模
SPLIT_V2_MAX_PSEUDO_VAL_EXAMPLES = 30

# 当 train_full 的样本量小于该阈值时，不再强行切 pseudo-val
# 目的是避免训练样本本来就不多，还被继续分走，影响 LoRA 微调
SPLIT_V2_MIN_TRAIN_EXAMPLES_TO_ENABLE_PSEUDO_VAL = 20

# 是否导出 hard test 子集
# 建议保留，便于观察模型在难样本上的泛化
SPLIT_V2_EXPORT_HARD_TEST = True

# hard test 规则 1：confidence 不等于 high 即视为更难样本
SPLIT_V2_HARD_TEST_USE_NON_HIGH_CONFIDENCE = True

# hard test 规则 2：n_cells 小于该阈值时，视为更难样本
# 你后面数据量变大后，这个阈值可以再略提高一点，比如 100
SPLIT_V2_HARD_TEST_MIN_N_CELLS_THRESHOLD = 80

# hard test 规则 3：在 test 集中出现次数小于等于该阈值的 cell type，
# 视为稀有 cell type，并纳入 hard test
SPLIT_V2_HARD_TEST_RARE_CELLTYPE_MAX_COUNT = 1

# 小数据集模式阈值：dataset 总数 <= 该值时，优先保证 train/test
# 由于你现在会自动选出更多 dataset，这里可以从 5 提高到 8，
# 让"小数据集特殊逻辑"覆盖更稳一些。
SPLIT_V2_SMALL_DATASET_UPPER_BOUND = 8

# 正常分层模式的最小 dataset 数（主要作为说明性参数）
# 当 dataset 数量明显上来后，再考虑把 stratify 打开。
SPLIT_V2_NORMAL_SPLIT_MIN_DATASETS = 9

# Phase 2 benchmark split：全局稀有 cell type 的最大出现次数阈值
# 跨 train+val+test 合计出现次数 <= 该值的 cell type 视为稀有，进入 test_rare 子集
SPLIT_V2_RARE_MAX_GLOBAL_COUNT = 3


# =========================================================
# [Phase 3] 推理决策阈值
# =========================================================

# 最终置信分数 >= 该值时，决策为 accept
FINAL_ACCEPT_THRESHOLD = 0.80

# 最终置信分数 >= 该值（但 < FINAL_ACCEPT_THRESHOLD）时，决策为 review
FINAL_REVIEW_THRESHOLD = 0.45

# 检索最高得分 < 该值时，视为 novel-like mismatch 候选
NOVELTY_THRESHOLD = 0.25