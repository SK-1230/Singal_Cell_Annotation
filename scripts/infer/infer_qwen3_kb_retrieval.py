"""
infer_qwen3_kb_retrieval.py — 带知识库检索的 Qwen3 单细胞注释推理脚本

整体推理流程
-----------
1. 加载 Marker KB（知识库）：包含训练集中所有细胞类型的 marker 基因条目
2. 加载 Base Model + LoRA Adapter，并合并为单一权重
3. 逐条读取测试样本，对每条样本：
   a. 从 KB 中检索与当前 cluster 的 marker 基因最相似的候选细胞类型（Jaccard 相似度）
   b. 将检索结果作为"提示"追加到 user 消息末尾（Retrieval-Augmented Generation, RAG）
   c. 调用模型生成答案
   d. 解析输出的 JSON 字段，提取预测细胞类型和本体 ID
4. 与标注的金标准（gold label）对比，计算多维度评估指标
5. 保存结果 JSONL 和指标 JSON

用法
----
  CUDA_VISIBLE_DEVICES=0 python -u scripts/infer/infer_qwen3_kb_retrieval.py \\
      --adapter output/qwen3_4b_sc_sft_hf_trl_v2_<时间戳>

  # 关闭 KB 检索，用纯模型推理
  CUDA_VISIBLE_DEVICES=0 python -u scripts/infer/infer_qwen3_kb_retrieval.py \\
      --adapter output/... --no-kb
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 将项目 src/ 目录加入 Python 路径，使 sca.* 包可被导入
_PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_DIR / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# =========================
# 本体层级父节点映射（用于层级匹配评估）
# =========================
# 从 ontology_index.jsonl 和 marker_examples_v2.jsonl 构建
# 格式: {normalized_label -> normalized_parent_label}
_PARENT_MAP: Dict[str, str] = {}


def _build_parent_map() -> Dict[str, str]:
    """
    构建细胞类型层级父节点映射表。

    数据来源（两者合并，覆盖更广）：
    1. ontology_index.jsonl：Cell Ontology 官方层级关系
    2. marker_examples_v2.jsonl：训练数据中 marker 记录的 cell_ontology_parent_label 字段

    示例映射：
      "fibroblast" -> "stromal cell"
      "cd8 positive alpha beta t cell" -> "t cell"
      "kupffer cell" -> "macrophage"

    Returns:
        {规范化细胞类型名 -> 规范化父类型名} 字典
    """
    pmap: Dict[str, str] = {}

    # 来源 1：Cell Ontology 索引文件
    ont_path = _PROJECT_DIR / "data/knowledge/ontology_index.jsonl"
    if ont_path.exists():
        with open(ont_path, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                label = normalize_text(d.get("label", ""))
                parent = normalize_text(d.get("parent_label", "") or "")
                if label and parent and label != parent:
                    pmap[label] = parent

    # 来源 2：marker 中间文件（覆盖训练集所有细胞类型）
    marker_path = _PROJECT_DIR / "data/intermediate/marker_examples_v2.jsonl"
    if marker_path.exists():
        with open(marker_path, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                label = normalize_text(d.get("cell_type_clean", ""))
                parent = normalize_text(d.get("cell_ontology_parent_label", "") or "")
                if label and parent and label != parent:
                    pmap[label] = parent

    # 来源 3：人工补充映射（覆盖 organ-specific 子类、发育中间态等常见关系）
    # 优先级最高：人工校正应覆盖自动化 CL 数据。
    # 例如 "alveolar macrophage" 在 CL 中父节点为 "macrophage"，但人工知道它是
    # "lung macrophage" 的子类，需要覆盖以得到正确的层级兼容匹配。
    supplement_path = _PROJECT_DIR / "data/knowledge/parent_map_supplement.json"
    if supplement_path.exists():
        with open(supplement_path, "r") as f:
            supplement = json.load(f)
        added = 0
        for raw_label, raw_parent in supplement.items():
            if raw_label.startswith("_"):
                continue  # 跳过注释字段
            label = normalize_text(raw_label)
            parent = normalize_text(raw_parent)
            if label and parent and label != parent:
                pmap[label] = parent  # 最高优先级，覆盖已有条目
                added += 1
        logging.info("Loaded %d entries from parent_map_supplement.json", added)

    logging.info("Built parent map with %d cell type entries total", len(pmap))
    return pmap


def _get_ancestors(label: str, max_depth: int = 3) -> set:
    """
    沿父节点链向上爬，收集 label 的所有祖先（最多 max_depth 层）。

    例如: "cd8 positive alpha beta t cell"
      → 父 "t cell" → 父 "lymphocyte" → 父 "leukocyte"

    Args:
        label: 已规范化的细胞类型名
        max_depth: 最多向上追溯几层

    Returns:
        祖先标签集合（不包含 label 本身）
    """
    ancestors = set()
    current = label
    for _ in range(max_depth):
        parent = _PARENT_MAP.get(current)
        if not parent or parent == current:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def is_ontology_compatible(pred_norm: str, gold_norm: str):
    """
    判断预测标签与金标准标签是否在本体层级上兼容。

    兼容的三种情况：
      1. exact：完全一致
      2. pred_is_parent：预测是金标准的父类（模型"太粗"）
           例: pred="t cell", gold="cd8 positive alpha beta t cell"
      3. pred_is_child：预测是金标准的子类（模型"太细"）
           例: pred="cd8 positive alpha beta t cell", gold="t cell"

    这三种情况在生物学上都是合理的预测，不应算严重错误。

    Args:
        pred_norm: 规范化后的预测标签
        gold_norm: 规范化后的金标准标签

    Returns:
        (is_compatible: bool, compat_type: str)
    """
    if pred_norm == gold_norm:
        return True, "exact"

    # 检查 pred 是否是 gold 的祖先（pred 更粗）
    gold_ancestors = _get_ancestors(gold_norm)
    if pred_norm in gold_ancestors:
        return True, "pred_is_parent"

    # 检查 gold 是否是 pred 的祖先（pred 更细）
    pred_ancestors = _get_ancestors(pred_norm)
    if gold_norm in pred_ancestors:
        return True, "pred_is_child"

    return False, "incompatible"

# =========================
# 默认参数（可通过 CLI 覆盖）
# =========================
BASE_MODEL_PATH = str(_PROJECT_DIR / "my_models/Qwen/Qwen3-4B")
# 训练完成后，将此路径更新为最新的 4B checkpoint 目录。
# 运行时也可通过 --adapter 参数覆盖此默认值。
# 示例：output/qwen3_4b_sc_sft_hf_trl_v2_<时间戳>
ADAPTER_PATH = str(_PROJECT_DIR / "output/qwen3_4b_sc_sft_hf_trl_v2_20260407_182321/checkpoint-620")  


# 测试集：完整格式（包含 markers、tissue 等元数据，用于 KB 检索）
TEST_FILE = str(_PROJECT_DIR / "data/splits/test_full_v2.jsonl")

# 知识库文件：08_build_marker_kb.py 生成，包含训练集所有细胞类型的 marker 信息
MERGED_KB_PATH = str(_PROJECT_DIR / "data/knowledge/merged_marker_kb.jsonl")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = str(_PROJECT_DIR / f"output/infer_kb_retrieval_{timestamp}")

MAX_NEW_TOKENS = 512      # 模型最多生成的 token 数，JSON 格式答案通常 200 token 以内
DO_SAMPLE = False         # False = 贪心解码（每步选概率最高的 token），输出稳定可复现
TEMPERATURE = 1.0         # DO_SAMPLE=False 时此参数无效
USE_KB_RETRIEVAL = True   # 是否启用知识库增强推理
KB_TOPK = 5               # 每次检索返回的候选细胞类型数量
KB_TISSUE_BONUS = 0.2     # 组织匹配时给检索分数加的奖励值（鼓励同组织的结果排更前）


# =========================
# 知识库（KB）加载与检索
# =========================

class SimpleMarkerKB:
    """
    轻量级 Marker 知识库。

    知识库格式：每行一个 JSON 条目，结构如：
    {
        "cell_type_clean": "Kupffer cell",
        "tissue_general": "liver",
        "marker_genes": ["CD163", "C1QC", ...],
        "positive_markers": [{"gene": "CD163", "logFC": 2.1, ...}, ...]
    }

    检索方法：Jaccard 相似度
      score = |query_genes ∩ kb_genes| / |query_genes ∪ kb_genes|
    直观含义：两个基因集合的交集占并集的比例，越高表示越相似。
    """

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.is_loaded = False

    def load(self, path: str) -> None:
        """从 JSONL 文件加载所有 KB 条目到内存。"""
        p = Path(path)
        if not p.exists():
            logging.warning("KB file not found: %s — retrieval will be disabled", path)
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.entries.append(json.loads(line))
                    except Exception:
                        pass  # 跳过格式异常行
        logging.info("Loaded %d KB entries from %s", len(self.entries), p.name)
        self.is_loaded = True

    def retrieve(
        self,
        query_genes: List[str],
        tissue: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        根据 query 基因列表从 KB 中检索最相似的细胞类型。

        Args:
            query_genes: 当前 cluster 的 marker 基因列表
            tissue: 当前样本的组织类型（用于加分，提升同组织条目的排名）
            top_k: 返回前 K 个候选

        Returns:
            按相似度降序排列的 KB 条目列表（最多 top_k 条）
        """
        if not self.is_loaded or not self.entries:
            return []

        # 将 query 基因名全部转大写，规避大小写不一致问题
        query_set = {g.upper() for g in query_genes}

        scored = []
        for entry in self.entries:
            kb_genes = entry.get("marker_genes", [])
            if not kb_genes:
                continue

            kb_set = {g.upper() for g in kb_genes}

            # Jaccard 相似度：交集 / 并集
            union = len(query_set | kb_set)
            overlap = len(query_set & kb_set)
            if union == 0 or overlap == 0:
                continue  # 没有任何重叠，跳过
            score = overlap / union

            # 组织匹配奖励：如果 KB 条目的组织与查询组织匹配，给分数加 bonus
            # 这有助于在同名细胞类型有多个组织来源时，优先返回同组织的
            if tissue:
                kb_tissue = str(
                    entry.get("tissue_general", entry.get("tissue", "")) or ""
                ).lower()
                if tissue.lower() in kb_tissue or kb_tissue in tissue.lower():
                    score += KB_TISSUE_BONUS

            scored.append((score, entry))

        # 按得分降序排列，取前 top_k 个
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


# =========================
# Prompt 构建（RAG 注入）
# =========================

def build_prompt_with_kb(
    messages: List[Dict[str, str]],
    kb: SimpleMarkerKB,
    full_record: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    在原始消息基础上注入 KB 检索结果，构建增强版推理 prompt。

    RAG（Retrieval-Augmented Generation）思路：
      - 先用当前 cluster 的 marker 基因在 KB 中检索相似细胞类型
      - 将检索到的候选列表作为"参考提示"追加到 user 消息末尾
      - 模型在生成答案时可以参考这些提示，减少幻觉（hallucination）

    如果 KB 未加载或检索不到结果，则直接返回去掉 assistant 消息的原始 messages。

    Args:
        messages: 原始对话消息列表（含 system/user/assistant）
        kb: 已加载的 SimpleMarkerKB 实例
        full_record: 完整测试记录，包含 marker 基因、tissue 等用于检索的元数据

    Returns:
        用于推理的消息列表（不含 assistant 消息，assistant 内容由模型生成）
    """
    if not USE_KB_RETRIEVAL or not kb.is_loaded or full_record is None:
        # 不使用 KB：直接去掉 assistant 消息，返回 system + user
        return [m for m in messages if m.get("role") != "assistant"]

    # 从完整记录中提取 marker 基因（支持两种数据格式）
    positive_markers = full_record.get("positive_markers", [])
    if isinstance(positive_markers, list) and positive_markers and isinstance(positive_markers[0], dict):
        # v2 格式：positive_markers 是带详细统计的字典列表
        query_genes = [m.get("gene", "") for m in positive_markers if m.get("gene")]
    else:
        # v1 格式：marker_genes 是基因名字符串列表
        query_genes = full_record.get("marker_genes", [])

    tissue = full_record.get("tissue_general", full_record.get("tissue", ""))

    # 执行 KB 检索
    candidates = kb.retrieve(query_genes, tissue=tissue, top_k=KB_TOPK)

    if not candidates:
        # 检索无结果，不注入任何提示
        return [m for m in messages if m.get("role") != "assistant"]

    # 将检索结果格式化为可读的提示文本
    hint_lines = ["\n--- KB Candidate Hints (ranked by marker overlap) ---"]
    for i, c in enumerate(candidates, 1):
        # 优先取本体标准名，其次取清洗后的标签
        label = c.get("cell_type_clean") or c.get("cell_type_target_label") or c.get("label", "unknown")
        ont_id = c.get("cell_type_target_id") or c.get("cell_ontology_id") or ""
        # 同样支持两种 marker 格式
        kb_genes = c.get("positive_markers", c.get("marker_genes", []))
        if isinstance(kb_genes, list) and kb_genes and isinstance(kb_genes[0], dict):
            kb_genes = [m.get("gene", "") for m in kb_genes if m.get("gene")]
        # 只展示前 5 个 marker 基因，避免提示过长
        hint_lines.append(f"  {i}. {label} ({ont_id}): {', '.join(kb_genes[:5])}")
    hint_lines.append("---")
    kb_hint = "\n".join(hint_lines)

    # 将 KB 提示注入到 user 消息末尾
    result = []
    for m in messages:
        if m.get("role") == "assistant":
            continue  # 推理时去掉 assistant 消息（让模型自己生成）
        if m.get("role") == "user":
            # 在 user 消息后追加 KB 提示和说明
            new_content = (
                m["content"]
                + "\n"
                + kb_hint
                + "\n\nNote: Use the KB hints above as supporting evidence, but make your own judgment."
            )
            result.append({"role": "user", "content": new_content})
        else:
            result.append(m)
    return result


# =========================
# 文本处理与评估工具函数
# =========================

def normalize_text(x: Any) -> str:
    """
    规范化文本用于比较：转小写、替换分隔符、去除多余空格、移除 "human" 等干扰词。

    为什么需要规范化：
      模型可能输出 "CD4-Positive T Cell"，金标准是 "cd4 positive t cell"，
      字符串完全不同但语义相同。规范化后均变为 "cd4 positive t cell"，
      可以正确判断为 exact match。
    """
    if x is None:
        return ""
    s = str(x).strip().lower()
    # 将常见分隔符统一替换为空格
    s = s.replace("_", " ").replace("-", " ").replace("/", " ").replace(",", " ")
    # 去掉 "human" 这个无意义的修饰词（CellOntology 标签中常见）
    s = re.sub(r"\bhuman\b", "", s)
    # 合并多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_json_block(text: str) -> Optional[str]:
    """
    从模型输出的原始文本中提取 JSON 字符串。

    Qwen3 输出格式（thinking 模式关闭时）：
      <think>\n\n</think>\n\n{"cell_type": "...", ...}

    此函数尝试多种策略：
      1. 直接解析（输出可能已是纯 JSON）
      2. 去掉 </think> 标签后解析
      3. 用正则找 {...} 块

    Args:
        text: 模型原始输出文本

    Returns:
        有效的 JSON 字符串，或 None（无法提取时）
    """
    text = text.strip()

    # 策略 1：整个输出就是合法 JSON
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 策略 2：去掉 thinking 标签后再试
    text = re.sub(r"^\s*</think>\s*", "", text, flags=re.IGNORECASE).strip()

    # 策略 3：用正则提取第一个 {...} 块
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except Exception:
            return None
    return None


def parse_prediction(raw_text: str) -> Dict[str, Any]:
    """
    解析模型输出的原始文本，提取结构化预测结果。

    Args:
        raw_text: 模型生成的原始文本

    Returns:
        字典，包含：
          - parse_ok (bool): 是否成功解析
          - parsed (dict|None): 解析后的 JSON 字典
          - parse_error (str): 解析失败时的错误信息
    """
    json_block = extract_json_block(raw_text.strip())
    if json_block is None:
        return {"parse_ok": False, "parsed": None, "parse_error": "No valid JSON"}
    try:
        return {"parse_ok": True, "parsed": json.loads(json_block), "parse_error": ""}
    except Exception as e:
        return {"parse_ok": False, "parsed": None, "parse_error": repr(e)}


def infer_lineage(label: Any) -> str:
    """
    从细胞类型名称推断其谱系（lineage）大类。

    用于计算"同谱系准确率"指标：即使细胞类型名称不完全匹配，
    只要谱系相同（如都是 T 细胞），也算部分正确。

    反之，如果预测为 T 细胞但金标准是 B 细胞，则为"严重错误"（跨谱系混淆）。

    Args:
        label: 细胞类型名称（任意大小写）

    Returns:
        谱系标识字符串，如 "t_cell", "b_cell", "macrophage" 等
    """
    s = normalize_text(label)
    if not s:
        return "unknown"
    if "gamma delta t" in s:
        return "gamma_delta_t"
    if "regulatory t" in s:
        return "t_cell"
    if "natural killer" in s or "nk cell" in s:
        return "nk"
    if "t cell" in s or "alpha beta t" in s:
        return "t_cell"
    if "b cell" in s or "plasma" in s:
        return "b_cell"
    if "monocyte" in s:
        return "monocyte"
    if "dendritic" in s:
        return "dendritic"
    if "macrophage" in s:
        return "macrophage"
    if "erythrocyte" in s or "erythroid" in s:
        return "erythroid"
    if "megakaryocyte" in s or "platelet" in s:
        return "megakaryocyte_platelet"
    if "neutrophil" in s:
        return "neutrophil"
    if "hepatocyte" in s:
        return "hepatocyte"
    if "endothelial" in s:
        return "endothelial"
    if "fibroblast" in s:
        return "fibroblast"
    if "epithelial" in s:
        return "epithelial"
    return "other"


def compare_labels(pred: Any, gold: Any) -> Dict[str, Any]:
    """
    多维度比较预测标签与金标准标签。

    评估维度：
      - exact: 规范化后完全匹配
      - token_overlap: 词级别的 Jaccard 相似度（衡量部分正确）
      - same_lineage: 谱系相同（粗粒度正确）
      - severe_error: 谱系不同（跨大类混淆，临床意义最差）

    Args:
        pred: 预测的细胞类型名称
        gold: 金标准细胞类型名称

    Returns:
        包含各维度评估结果的字典
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    # 完全匹配（规范化后）
    exact = pred_norm == gold_norm and bool(pred_norm)

    # token 级 Jaccard 相似度
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if pred_tokens and gold_tokens:
        overlap = len(pred_tokens & gold_tokens) / max(len(pred_tokens), len(gold_tokens))
    else:
        overlap = 0.0

    # 谱系比较
    pred_lineage = infer_lineage(pred_norm)
    gold_lineage = infer_lineage(gold_norm)
    same_lineage = pred_lineage == gold_lineage and pred_lineage != "unknown"

    # 严重错误：两者谱系均已知但不同（跨大类错误，最不可接受）
    severe = (
        pred_lineage != gold_lineage
        and pred_lineage != "unknown"
        and gold_lineage != "unknown"
    )

    # 层级兼容判断：预测是金标准的祖先或后代，在生物学上都合理
    ontology_compatible, compat_type = is_ontology_compatible(pred_norm, gold_norm)

    return {
        "exact": exact,
        "token_overlap": round(overlap, 4),
        "same_lineage": same_lineage,
        "severe_error": severe,
        "pred_lineage": pred_lineage,
        "gold_lineage": gold_lineage,
        "ontology_compatible": ontology_compatible,   # 层级兼容（比 exact 更宽松）
        "compat_type": compat_type,                   # exact/pred_is_parent/pred_is_child/incompatible
    }


def load_examples(test_file: str) -> List[Dict[str, Any]]:
    """
    加载测试集 JSONL 文件。

    使用 test_full_v2.jsonl 而非 test_messages_*.jsonl，
    因为 full 格式包含 marker 基因等 KB 检索所需的元数据。

    Args:
        test_file: JSONL 文件路径

    Returns:
        测试样本列表
    """
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except Exception as e:
                logging.warning("Skip invalid JSON line %d: %s", line_idx, e)
    return examples


def write_results_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """将推理结果逐行写入 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_and_print_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    汇总计算并打印所有评估指标。

    指标说明：
      - parse_ok_rate: JSON 解析成功率（模型输出格式是否符合预期）
      - cell_type_exact_accuracy: 规范化后完全匹配准确率
      - cell_type_normalized_exact_accuracy: 同上（此处两者相同）
      - cell_type_same_lineage_rate: 谱系级准确率（粗粒度）
      - cell_type_severe_error_rate: 跨大类严重错误率（越低越好）
      - cell_ontology_id_accuracy: 细胞本体 ID 精确匹配率

    Args:
        results: 每条样本的推理结果列表

    Returns:
        指标字典
    """
    n = len(results)
    if n == 0:
        logging.warning("No results to evaluate")
        return {}

    parse_ok = sum(1 for r in results if r.get("parse_ok", False))
    parse_ok_rate = parse_ok / n

    exact_hits = [r for r in results if r.get("cell_type_exact", False)]
    norm_exact_hits = [r for r in results if r.get("cell_type_normalized_exact", False)]
    lineage_hits = [r for r in results if r.get("cell_type_same_lineage", False)]
    severe_errors = [r for r in results if r.get("cell_type_severe_error", False)]
    high_risk = [
        r for r in results
        if r.get("cell_type_severe_error", False) and not r.get("cell_type_same_lineage", False)
    ]

    # 只对有金标准本体 ID 的样本计算 ID 准确率
    ont_eval = [r for r in results if r.get("gold_ontology_id") and r.get("gold_ontology_id") != ""]
    ont_exact = [
        r for r in ont_eval
        if r.get("pred_ontology_id")
        and normalize_text(r.get("pred_ontology_id")) == normalize_text(r.get("gold_ontology_id"))
    ]
    ont_id_accuracy = len(ont_exact) / len(ont_eval) if ont_eval else None

    # 层级匹配指标
    # ontology_compatible = exact + pred是父类 + pred是子类
    compat_hits = [r for r in results if r.get("ontology_compatible", False)]
    pred_parent_hits = [r for r in compat_hits if r.get("compat_type") == "pred_is_parent"]
    pred_child_hits  = [r for r in compat_hits if r.get("compat_type") == "pred_is_child"]
    exact_only_hits  = [r for r in compat_hits if r.get("compat_type") == "exact"]

    metrics = {
        "n_total": n,
        "n_parse_ok": parse_ok,
        "parse_ok_rate": round(parse_ok_rate, 4),
        # --- 严格指标（字符串完全匹配）---
        "cell_type_exact_accuracy": round(len(exact_hits) / n, 4),
        # --- 层级宽松指标（推荐作为主指标）---
        # 预测在本体层级上与金标准兼容（父/子/完全匹配均算正确）
        "ontology_compatible_accuracy": round(len(compat_hits) / n, 4),
        # 其中：预测比金标准更粗（如 pred=t_cell, gold=cd8+ t cell）
        "pred_more_general_rate": round(len(pred_parent_hits) / n, 4),
        # 其中：预测比金标准更细（如 pred=cd8+ t cell, gold=lymphocyte）
        "pred_more_specific_rate": round(len(pred_child_hits) / n, 4),
        # --- 谱系级指标 ---
        "cell_type_same_lineage_rate": round(len(lineage_hits) / n, 4),
        "cell_type_severe_error_rate": round(len(severe_errors) / n, 4),
        "high_risk_error_count": len(high_risk),
    }
    if ont_id_accuracy is not None:
        metrics["cell_ontology_id_accuracy"] = round(ont_id_accuracy, 4)
        metrics["n_ontology_id_evaluated"] = len(ont_eval)

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


# =========================
# 主推理流程
# =========================

def main() -> None:
    """
    主函数：执行完整的推理与评估流程。

    流程：
      1. 解析 CLI 参数
      2. 加载 Marker 知识库（若启用）
      3. 加载 Base Model + LoRA Adapter → 合并权重 → 设为 eval 模式
      4. 逐条样本：检索 KB → 构建 prompt → 生成 → 解析输出 → 与金标准比较
      5. 计算指标，保存结果
    """
    # ------ 步骤 1：解析 CLI 参数 ------
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3 SCA KB-retrieval 推理脚本")
    parser.add_argument("--base-model", default=BASE_MODEL_PATH, help="Base model 路径")
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="LoRA adapter 路径（checkpoint 目录）")
    parser.add_argument("--test-file", default=TEST_FILE, help="测试集 JSONL 路径（full 格式）")
    parser.add_argument("--kb-path", default=MERGED_KB_PATH, help="Marker KB JSONL 路径")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--no-kb", action="store_true", help="禁用 KB 检索，使用纯模型推理")
    args = parser.parse_args()

    use_retrieval = USE_KB_RETRIEVAL and not args.no_kb

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------ 初始化本体层级映射（用于层级匹配评估）------
    global _PARENT_MAP
    _PARENT_MAP = _build_parent_map()

    # ------ 步骤 2：加载知识库 ------
    kb = SimpleMarkerKB()
    if use_retrieval:
        kb.load(args.kb_path)
        if not kb.is_loaded:
            logging.warning("KB not loaded — running without retrieval")

    # ------ 步骤 3：加载模型 ------
    # 自动从 adapter_config.json 读取 base_model_name_or_path，
    # 避免手动指定 --base-model 时 4B/8B 搞错导致维度不匹配
    base_model_path = args.base_model
    if args.adapter and Path(args.adapter).exists():
        adapter_cfg_path = Path(args.adapter) / "adapter_config.json"
        if adapter_cfg_path.exists():
            adapter_cfg = json.loads(adapter_cfg_path.read_text())
            detected = adapter_cfg.get("base_model_name_or_path", "")
            if detected and detected != base_model_path:
                logging.info(
                    "Auto-detected base model from adapter_config.json: %s", detected
                )
                base_model_path = detected

    logging.info("Loading base model from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 以 float16 加载（推理不需要 bf16 训练的全精度，fp16 节省显存且速度稍快）
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",   # 自动分配到可用 GPU（单卡时等价于 "cuda:0"）
        trust_remote_code=True,
    )

    if args.adapter and Path(args.adapter).exists():
        logging.info("Loading adapter from %s", args.adapter)
        # PeftModel.from_pretrained：在 base model 上加载 LoRA adapter
        peft_model: PeftModel = PeftModel.from_pretrained(model, args.adapter)
        # merge_and_unload：将 LoRA 权重 (A×B) 合并回 base 权重 W
        # 合并后模型变回普通模型（无 PEFT 层），推理速度更快
        # 数学：W_new = W + (alpha/r) × A × B
        model = peft_model.merge_and_unload()  # type: ignore[operator]

    # 推理模式：关闭 dropout，不计算梯度
    model.eval()

    # ------ 步骤 4：逐条推理 ------
    examples = load_examples(args.test_file)
    logging.info("Loaded %d test examples", len(examples))

    results = []
    for ex in tqdm(examples, desc="Inference", unit="example"):
        messages = ex.get("messages", [])
        full_record = ex  # 保留完整记录供 KB 检索使用

        # 构建推理 prompt（可选注入 KB 提示）
        infer_messages = build_prompt_with_kb(messages, kb, full_record)

        # 用 chat_template 将 messages 列表格式化为模型输入字符串
        # add_generation_prompt=True 会在末尾加上 <|im_start|>assistant\n，触发模型生成
        try:
            text_input = tokenizer.apply_chat_template(
                infer_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # 关闭思维链，直接输出 JSON 答案
            )
        except TypeError:
            text_input = tokenizer.apply_chat_template(
                infer_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize 并移至 GPU
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

        # 推理生成（不计算梯度，节省显存）
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,    # False = 贪心解码，输出确定性最强
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 解码：只取新生成的 token（去掉 prompt 部分）
        # inputs["input_ids"].shape[1] 是 prompt 的 token 数，之后的才是模型生成的
        raw_output = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,  # 去掉 <|im_end|> 等特殊 token
        )

        # 解析模型输出 JSON
        pred_info = parse_prediction(raw_output)
        parsed = pred_info.get("parsed") or {}

        # 从 messages 中提取金标准 assistant 回复，解析金标准 JSON
        gold_content = ""
        for m in messages:
            if m.get("role") == "assistant":
                gold_content = m.get("content", "")
                break
        gold_info = parse_prediction(gold_content)
        gold_parsed = gold_info.get("parsed") or {}

        # 提取预测和金标准的细胞类型名称
        pred_ct = parsed.get("cell_type", "")
        gold_ct = gold_parsed.get("cell_type", full_record.get(
            "cell_type_target_label", full_record.get("cell_type_clean", "")
        ))

        # 提取细胞本体 ID
        pred_ont_id = parsed.get("cell_ontology_id", "")
        gold_ont_id = full_record.get(
            "cell_type_target_id", full_record.get("cell_ontology_id", "")
        )

        # 多维度标签比较
        label_cmp = compare_labels(pred_ct, gold_ct)

        # 构建结果记录
        result_row = {
            "dataset_id": full_record.get("dataset_id", ""),
            "tissue_general": full_record.get("tissue_general", ""),
            "gold_cell_type": gold_ct,
            "pred_cell_type": pred_ct,
            "gold_ontology_id": str(gold_ont_id or ""),
            "pred_ontology_id": str(pred_ont_id or ""),
            "parse_ok": pred_info["parse_ok"],
            "cell_type_exact": label_cmp["exact"],
            "cell_type_normalized_exact": label_cmp["exact"],
            "cell_type_same_lineage": label_cmp["same_lineage"],
            "cell_type_severe_error": label_cmp["severe_error"],
            "token_overlap": label_cmp["token_overlap"],
            "pred_lineage": label_cmp["pred_lineage"],
            "gold_lineage": label_cmp["gold_lineage"],
            "raw_output": raw_output[:500],      # 只保存前 500 字符，避免结果文件过大
            "kb_retrieval_used": use_retrieval and kb.is_loaded,
            "ontology_compatible": label_cmp.get("ontology_compatible", False),
            "compat_type": label_cmp.get("compat_type", "incompatible"),
        }
        results.append(result_row)

    # ------ 步骤 5：保存结果与指标 ------
    write_results_jsonl(out_dir / "results.jsonl", results)
    metrics = compute_and_print_metrics(results)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()


'''
nohup bash -c '
CUDA_VISIBLE_DEVICES=0 python -u scripts/infer/infer_qwen3_kb_retrieval.py \
    --base-model my_models/Qwen/Qwen3-4B \
    --adapter output/qwen3_4b_sc_sft_hf_trl_v2_20260408_210606/checkpoint-620 \
    --test-file data/splits/test_full_v3.jsonl \
    --output-dir output/infer_v3_on_v3_testset
' > output/infer_v3.log 2>&1 &

tail -f output/infer_v3.log
'''