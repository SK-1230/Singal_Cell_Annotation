"""
diag_config.py — 诊断模块共用路径与默认参数

所有诊断脚本从此处统一读取路径，避免各脚本硬编码。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# 项目根目录（此文件在 scripts/diagnosis/ 下，上两级为根目录）
# ─────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────
# 数据文件
# ─────────────────────────────────────────────
TRAIN_FULL        = PROJECT_DIR / "data/splits/train_full_v2.jsonl"
VAL_FULL          = PROJECT_DIR / "data/splits/val_full_v2.jsonl"
TEST_FULL         = PROJECT_DIR / "data/splits/test_full_v2.jsonl"
TEST_HARD_FULL    = PROJECT_DIR / "data/splits/test_hard_full.jsonl"
TEST_RARE_MSGS    = PROJECT_DIR / "data/splits/test_rare_messages_no_think.jsonl"
TEST_UNMAPPED     = PROJECT_DIR / "data/splits/test_unmapped_messages_no_think.jsonl"

# ─────────────────────────────────────────────
# 知识库 & 本体资源
# ─────────────────────────────────────────────
ONTOLOGY_INDEX    = PROJECT_DIR / "data/knowledge/ontology_index.jsonl"
MERGED_KB         = PROJECT_DIR / "data/knowledge/merged_marker_kb.jsonl"
PARENT_SUPPLEMENT = PROJECT_DIR / "data/knowledge/parent_map_supplement.json"

# ─────────────────────────────────────────────
# 模型 & 推理
# ─────────────────────────────────────────────
BASE_MODEL        = PROJECT_DIR / "my_models/Qwen/Qwen3-4B"
# 最新最优 adapter（checkpoint-130，对应最低 eval loss）
BEST_ADAPTER      = PROJECT_DIR / "output/qwen3_4b_sc_sft_hf_trl_v2_20260405_193112/checkpoint-130"
INFER_SCRIPT      = PROJECT_DIR / "scripts/infer/infer_qwen3_kb_retrieval.py"

# ─────────────────────────────────────────────
# 最新已知推理结果（可直接复用，无需重跑）
# ─────────────────────────────────────────────
BEST_INFER_RESULTS = PROJECT_DIR / "output/infer_kb_retrieval_20260405_211943/results.jsonl"

# ─────────────────────────────────────────────
# 诊断输出目录
# ─────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DIAG_OUTPUT_DIR   = PROJECT_DIR / f"output/diagnosis/{_TIMESTAMP}"


def get_diag_output_dir(run_name: str = "") -> Path:
    """
    返回带时间戳的诊断输出目录，并自动创建。

    Args:
        run_name: 可选的运行名称后缀，如 "no_kb" 或 "cell_type_only"
    """
    suffix = f"_{run_name}" if run_name else ""
    d = PROJECT_DIR / f"output/diagnosis/{_TIMESTAMP}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d
