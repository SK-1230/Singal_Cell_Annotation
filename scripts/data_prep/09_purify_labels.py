"""
09_purify_labels.py — 净化 marker_examples_v2.jsonl 的标签质量

问题背景
--------
marker_examples_v2.jsonl 中约 33% 的样本 cell_ontology_id 为空，
根本原因是 label_aliases.tsv 和 ontology_index.jsonl 仅有 ~120 条，
远少于实际数据中 204 种细胞类型，导致大量合法 CL 术语被误判为"noisy"。

本脚本
------
1. 内嵌 CL 标准映射表（~180 条），修正 cell_ontology_id / cell_ontology_label
   / cell_ontology_parent_label / cell_type_target_label / cell_type_target_id
2. 丢弃无法映射且明显非正常细胞类型的样本（malignant / abnormal / neoplastic）
3. 将 marker_examples_v2.jsonl 原地更新（保留备份）
4. 同步扩充 label_aliases.tsv

用法
----
  python scripts/data_prep/09_purify_labels.py
  python scripts/data_prep/09_purify_labels.py --dry-run   # 只统计，不写文件

输出
----
  data/intermediate/marker_examples_v2.jsonl      (覆盖更新)
  data/intermediate/marker_examples_v2_backup.jsonl
  resources/ontology/label_aliases.tsv            (追加新条目)
  data/meta/09_purify_labels_run_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]

MARKER_V2 = _PROJECT / "data/intermediate/marker_examples_v2.jsonl"
MARKER_V2_BACKUP = _PROJECT / "data/intermediate/marker_examples_v2_backup.jsonl"
ALIAS_TSV = _PROJECT / "resources/ontology/label_aliases.tsv"
SUMMARY_PATH = _PROJECT / "data/meta/09_purify_labels_run_summary.txt"

# ─────────────────────────────────────────────────────────────────────────────
# CL 映射表
# key  : cell_type_clean (小写，已规范化)
# value: (canonical_label, cl_id_or_None, parent_label_or_None)
#        cl_id = None 表示该类型合法但 CL 中无精确 ID，使用 parent 作为 target
# "DROP" 表示应丢弃该样本
# ─────────────────────────────────────────────────────────────────────────────
FIX_TABLE: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    # ── 应丢弃的非正常细胞类型 ────────────────────────────────────────────
    "malignant cell":       "DROP",
    "abnormal cell":        "DROP",
    "neoplastic cell":      "DROP",

    # ── T 细胞亚群 ────────────────────────────────────────────────────────
    "gamma-delta t cell":               ("gamma-delta T cell",                  "CL:0000798", "t cell"),
    "cd4-positive, alpha-beta t cell":  ("CD4-positive, alpha-beta T cell",     "CL:0000624", "t cell"),
    "cd8-positive, alpha-beta t cell":  ("CD8-positive, alpha-beta T cell",     "CL:0000625", "t cell"),
    "naive thymus-derived cd4-positive, alpha-beta t cell":
        ("naive thymus-derived CD4-positive, alpha-beta T cell", "CL:0000895", "cd4-positive, alpha-beta t cell"),
    "naive thymus-derived cd8-positive, alpha-beta t cell":
        ("naive thymus-derived CD8-positive, alpha-beta T cell", "CL:0000900", "cd8-positive, alpha-beta t cell"),
    "cd4-positive, alpha-beta memory t cell":
        ("CD4-positive, alpha-beta memory T cell", "CL:0000897", "cd4-positive, alpha-beta t cell"),
    "cd8-positive, alpha-beta memory t cell":
        ("CD8-positive, alpha-beta memory T cell", "CL:0000909", "cd8-positive, alpha-beta t cell"),
    "central memory cd4-positive, alpha-beta t cell":
        ("central memory CD4-positive, alpha-beta T cell", "CL:0000904", "cd4-positive, alpha-beta t cell"),
    "effector memory cd4-positive, alpha-beta t cell":
        ("effector memory CD4-positive, alpha-beta T cell", "CL:0000906", "cd4-positive, alpha-beta t cell"),
    "effector memory cd8-positive, alpha-beta t cell":
        ("effector memory CD8-positive, alpha-beta T cell", "CL:0000913", "cd8-positive, alpha-beta t cell"),
    "effector memory cd8-positive, alpha-beta t cell, terminally differentiated":
        ("terminally differentiated effector memory CD8-positive, alpha-beta T cell",
         "CL:0001062", "effector memory cd8-positive, alpha-beta t cell"),
    "effector cd4-positive, alpha-beta t cell":
        ("effector CD4-positive, alpha-beta T cell", "CL:0001044", "cd4-positive, alpha-beta t cell"),
    "effector cd8-positive, alpha-beta t cell":
        ("effector CD8-positive, alpha-beta T cell", "CL:0001050", "cd8-positive, alpha-beta t cell"),
    "activated cd4-positive, alpha-beta t cell":
        ("activated CD4-positive, alpha-beta T cell", "CL:0000896", "cd4-positive, alpha-beta t cell"),
    "activated cd8-positive, alpha-beta t cell":
        ("activated CD8-positive, alpha-beta T cell", "CL:0000906", "cd8-positive, alpha-beta t cell"),
    "activated cd4-positive, alpha-beta t cell, human":
        ("activated CD4-positive, alpha-beta T cell", "CL:0000896", "cd4-positive, alpha-beta t cell"),
    "mucosal invariant t cell":
        ("mucosal invariant T cell", "CL:0000940", "t cell"),
    "cd4-positive, cd25-positive, alpha-beta regulatory t cell":
        ("CD4-positive, CD25-positive, alpha-beta regulatory T cell", "CL:0000792", "regulatory t cell"),
    "lung resident memory cd8-positive, alpha-beta t cell":
        ("tissue-resident memory T cell", "CL:0002103", "cd8-positive, alpha-beta t cell"),
    "double negative thymocyte":
        ("double negative thymocyte", "CL:0002489", "thymocyte"),
    "double-positive, alpha-beta thymocyte":
        ("double-positive, alpha-beta thymocyte", "CL:0000809", "thymocyte"),
    "type i nk t cell":
        ("type I NK T cell", "CL:0000922", "nk t cell"),
    "mature nk t cell":
        ("mature NK T cell", "CL:0000815", "nk t cell"),

    # ── NK 细胞 ─────────────────────────────────────────────────────────
    "cd16-negative, cd56-bright natural killer cell, human":
        ("CD56-bright natural killer cell", "CL:0000938", "natural killer cell"),
    "cd16-positive, cd56-dim natural killer cell, human":
        ("CD56-dim natural killer cell", "CL:0000939", "natural killer cell"),
    "decidual natural killer cell, human":
        ("decidual natural killer cell", "CL:0002001", "natural killer cell"),

    # ── B 细胞 ─────────────────────────────────────────────────────────
    "mature b cell":          ("mature B cell",          "CL:0000785", "b cell"),
    "immature b cell":        ("immature B cell",        "CL:0000816", "b cell"),
    "transitional stage b cell": ("transitional stage B cell", "CL:0000818", "immature b cell"),
    "early pro-b cell":       ("early pro-B cell",       "CL:0000826", "pro-b cell"),
    "precursor b cell":       ("precursor B cell",       "CL:0000817", "b cell"),
    "pro-b cell":             ("pro-B cell",             "CL:0000826", "b cell"),
    "plasmacytoid dendritic cell, human": ("plasmacytoid dendritic cell", "CL:0000784", "dendritic cell"),

    # ── 单核 / 髓系 ──────────────────────────────────────────────────────
    "cd14-positive monocyte":
        ("CD14-positive monocyte", "CL:0002057", "monocyte"),
    "cd14-low, cd16-positive monocyte":
        ("CD14-low, CD16-positive monocyte", "CL:0002397", "monocyte"),
    "cd14-positive, cd16-negative classical monocyte":
        ("classical monocyte", "CL:0000860", "monocyte"),
    "cd14-positive, cd16-positive monocyte":
        ("CD14-positive, CD16-positive monocyte", "CL:0002396", "monocyte"),
    "intermediate monocyte":
        ("intermediate monocyte", "CL:0002396", "monocyte"),
    "mononuclear phagocyte":
        ("mononuclear phagocyte", "CL:0000113", "phagocyte"),
    "inflammatory macrophage":
        ("classically activated macrophage", "CL:0000863", "macrophage"),
    "alternatively activated macrophage":
        ("alternatively activated macrophage", "CL:0000890", "macrophage"),
    "kidney interstitial alternatively activated macrophage":
        ("alternatively activated macrophage", "CL:0000890", "macrophage"),
    "meningeal macrophage":
        ("meningeal macrophage", "CL:0002428", "macrophage"),
    "myeloid leukocyte":
        ("myeloid leukocyte", "CL:0000766", "leukocyte"),
    "cycling myeloid cell":
        ("myeloid cell", "CL:0000763", "leukocyte"),
    "granulocyte":
        ("granulocyte", "CL:0000094", "leukocyte"),
    "immature neutrophil":
        ("immature neutrophil", "CL:0000763", "granulocyte"),
    "granulocyte monocyte progenitor cell":
        ("granulocyte monocyte progenitor cell", "CL:0000557", "hematopoietic progenitor cell"),
    "liver dendritic cell":
        ("conventional dendritic cell", "CL:0000990", "dendritic cell"),
    "dendritic cell, human":
        ("dendritic cell", "CL:0000451", "leukocyte"),

    # ── 造血祖细胞 ──────────────────────────────────────────────────────
    "hematopoietic precursor cell":
        ("hematopoietic precursor cell", "CL:0000037", "hematopoietic cell"),
    "hematopoietic multipotent progenitor cell":
        ("hematopoietic multipotent progenitor cell", "CL:0000837", "hematopoietic progenitor cell"),
    "common lymphoid progenitor":
        ("common lymphoid progenitor", "CL:0000051", "hematopoietic progenitor cell"),
    "common dendritic progenitor":
        ("common dendritic progenitor", "CL:0001009", "hematopoietic progenitor cell"),
    "megakaryocyte-erythroid progenitor cell":
        ("megakaryocyte-erythroid progenitor cell", "CL:0000050", "hematopoietic progenitor cell"),
    "erythroid progenitor cell":
        ("erythroid progenitor cell", "CL:0000038", "hematopoietic progenitor cell"),
    "erythroid lineage cell":
        ("erythroid lineage cell", "CL:0000764", "hematopoietic cell"),
    "lymphoid lineage restricted progenitor cell":
        ("lymphoid lineage restricted progenitor cell", "CL:0000838", "hematopoietic progenitor cell"),
    "myeloid lineage restricted progenitor cell":
        ("myeloid lineage restricted progenitor cell", "CL:0000839", "hematopoietic progenitor cell"),
    "proerythroblast":
        ("proerythroblast", "CL:0000547", "erythroid progenitor cell"),
    "erythroblast":
        ("erythroblast", "CL:0000765", "erythroid lineage cell"),
    "cycling plasma cell":
        ("plasmablast", "CL:0000946", "b cell"),

    # ── 肝脏 ────────────────────────────────────────────────────────────
    "endothelial cell of hepatic sinusoid":
        ("liver sinusoidal endothelial cell", "CL:0000091", "endothelial cell"),
    "endothelial cell of pericentral hepatic sinusoid":
        ("liver sinusoidal endothelial cell", "CL:0000091", "endothelial cell"),
    "endothelial cell of periportal hepatic sinusoid":
        ("liver sinusoidal endothelial cell", "CL:0000091", "endothelial cell"),
    "centrilobular region hepatocyte":
        ("hepatocyte", "CL:0000182", "hepatic cell"),
    "midzonal region hepatocyte":
        ("hepatocyte", "CL:0000182", "hepatic cell"),
    "periportal region hepatocyte":
        ("hepatocyte", "CL:0000182", "hepatic cell"),
    "intrahepatic cholangiocyte":
        ("intrahepatic cholangiocyte", "CL:1000488", "epithelial cell"),
    "hepatic pit cell":
        ("hepatic stellate cell", "CL:0000632", "stromal cell"),

    # ── 肺 ──────────────────────────────────────────────────────────────
    "fibroblast of lung":
        ("lung fibroblast", "CL:0002503", "fibroblast"),
    "pulmonary alveolar type 1 cell":
        ("type I pneumocyte", "CL:0002062", "epithelial cell of lung"),
    "pulmonary alveolar type 2 cell":
        ("type II pneumocyte", "CL:0002063", "epithelial cell of lung"),
    "lung multiciliated epithelial cell":
        ("ciliated columnar cell of the tracheobronchial tree", "CL:0002145", "epithelial cell of lung"),
    "multiciliated columnar cell of tracheobronchial tree":
        ("ciliated columnar cell of the tracheobronchial tree", "CL:0002145", "epithelial cell of lung"),
    "pulmonary capillary endothelial cell":
        ("lung microvascular endothelial cell", "CL:0002144", "endothelial cell"),
    "lung microvascular endothelial cell":
        ("lung microvascular endothelial cell", "CL:0002144", "endothelial cell"),
    "lung endothelial cell":
        ("lung endothelial cell", "CL:0002144", "endothelial cell"),
    "pulmonary artery endothelial cell":
        ("pulmonary artery endothelial cell", "CL:1001567", "endothelial cell"),
    "lung macrophage":
        ("alveolar macrophage", "CL:0000583", "macrophage"),
    "lung goblet cell":
        ("goblet cell", "CL:0000160", "epithelial cell of lung"),
    "lung pericyte":
        ("pericyte cell", "CL:0000669", "stromal cell"),
    "epithelial cell of lung":
        ("epithelial cell of lung", "CL:0002632", "epithelial cell"),
    "respiratory basal cell":
        ("basal cell of respiratory epithelium", "CL:0002633", "epithelial cell of lung"),
    "respiratory tract goblet cell":
        ("respiratory tract goblet cell", "CL:0002370", "goblet cell"),
    "respiratory tract suprabasal cell":
        ("basal cell of respiratory epithelium", "CL:0002633", "epithelial cell of lung"),
    "pulmonary neuroendocrine cell":
        ("pulmonary neuroendocrine cell", "CL:0002088", "epithelial cell of lung"),
    "deuterosomal cell":
        ("deuterosomal cell", "CL:4033054", "epithelial cell of lung"),
    "mucus secreting cell of bronchus submucosal gland":
        ("mucus secreting cell", "CL:0000319", "epithelial cell"),
    "serous cell of epithelium of lobular bronchiole":
        ("serous secreting epithelial cell", "CL:0002596", "epithelial cell"),
    "serous secreting cell of bronchus submucosal gland":
        ("serous secreting epithelial cell", "CL:0002596", "epithelial cell"),
    "tracheobronchial serous cell":
        ("serous secreting epithelial cell", "CL:0002596", "epithelial cell"),
    "myoepithelial cell of bronchus submucosal gland":
        ("myoepithelial cell", "CL:0000185", "epithelial cell"),
    "bronchiolar smooth muscle cell":
        ("bronchial smooth muscle cell", "CL:0002598", "smooth muscle cell"),
    "bronchus fibroblast of lung":
        ("lung fibroblast", "CL:0002503", "fibroblast"),
    "vein endothelial cell of respiratory system":
        ("vein endothelial cell", "CL:0002543", "endothelial cell"),
    "pulmonary ionocyte":
        ("pulmonary ionocyte", "CL:4028002", "epithelial cell of lung"),
    "schwann cell":
        ("Schwann cell", "CL:0002573", "glial cell"),
    "lung resident memory cd8-positive, alpha-beta t cell":
        ("tissue-resident memory T cell", "CL:0002103", "cd8-positive, alpha-beta t cell"),

    # ── 血管内皮 ────────────────────────────────────────────────────────
    "endothelial cell of artery":
        ("artery endothelial cell", "CL:1000413", "endothelial cell"),
    "endothelial cell of lymphatic vessel":
        ("endothelial cell of lymphatic vessel", "CL:0002138", "endothelial cell"),
    "vein endothelial cell":
        ("vein endothelial cell", "CL:0002543", "endothelial cell"),
    "blood vessel endothelial cell":
        ("endothelial cell of vascular tree", "CL:0002139", "endothelial cell"),
    "blood vessel smooth muscle cell":
        ("vascular smooth muscle cell", "CL:0000359", "smooth muscle cell"),
    "myofibroblast cell":
        ("myofibroblast cell", "CL:0000186", "stromal cell"),
    "mesothelial cell":
        ("mesothelial cell", "CL:0000077", "epithelial cell"),
    "chondrocyte":
        ("chondrocyte", "CL:0000138", "connective tissue cell"),

    # ── 肠道 ────────────────────────────────────────────────────────────
    "enteroendocrine cell":
        ("enteroendocrine cell", "CL:0000164", "epithelial cell"),
    "intestinal enteroendocrine cell":
        ("enteroendocrine cell", "CL:0000164", "epithelial cell"),
    "intestinal epithelial cell":
        ("epithelial cell of intestine", "CL:0002253", "epithelial cell"),
    "gut absorptive cell":
        ("intestinal absorptive cell", "CL:0002563", "epithelial cell"),
    "intestinal crypt stem cell of colon":
        ("intestinal crypt stem cell", "CL:0002328", "somatic stem cell"),
    "transit amplifying cell":
        ("transit amplifying cell of colon", "CL:0002577", "somatic stem cell"),
    "paneth cell":
        ("Paneth cell of colon", "CL:0000510", "epithelial cell"),
    "tuft cell":
        ("tuft cell", "CL:0002209", "epithelial cell"),
    "intestine goblet cell":
        ("goblet cell", "CL:0000160", "epithelial cell"),

    # ── 肾脏 ────────────────────────────────────────────────────────────
    "kidney collecting duct intercalated cell":
        ("kidney collecting duct intercalated cell", "CL:1001107", "kidney tubule cell"),
    "kidney collecting duct principal cell":
        ("kidney collecting duct principal cell", "CL:0002518", "kidney tubule cell"),
    "kidney distal convoluted tubule epithelial cell":
        ("kidney distal convoluted tubule epithelial cell", "CL:1000849", "kidney tubule cell"),
    "kidney loop of henle thick ascending limb epithelial cell":
        ("kidney loop of Henle thick ascending limb epithelial cell", "CL:1001108", "kidney tubule cell"),
    "kidney loop of henle thin ascending limb epithelial cell":
        ("kidney loop of Henle thin ascending limb epithelial cell", "CL:1001111", "kidney tubule cell"),
    "kidney loop of henle thin descending limb epithelial cell":
        ("kidney loop of Henle thin descending limb epithelial cell", "CL:1001109", "kidney tubule cell"),
    "kidney connecting tubule epithelial cell":
        ("kidney connecting tubule epithelial cell", "CL:1000768", "kidney tubule cell"),
    "kidney interstitial cell":
        ("kidney interstitial cell", "CL:1000692", "stromal cell"),
    "parietal epithelial cell":
        ("parietal epithelial cell", "CL:0000077", "epithelial cell"),
    "epithelial cell of proximal tubule":
        ("kidney proximal convoluted tubule epithelial cell", "CL:1000838", "kidney tubule cell"),
    "podocyte":
        ("podocyte", "CL:0000653", "epithelial cell"),
    "glomerular mesangial cell":
        ("glomerular mesangial cell", "CL:0000650", "kidney cell"),
    "kidney cell":
        ("kidney cell", "CL:1000497", "animal cell"),
    "kidney epithelial cell":
        ("epithelial cell of kidney tubule", "CL:0002518", "epithelial cell"),
    "mesonephric nephron tubule epithelial cell":
        ("kidney tubule cell", "CL:0002518", "epithelial cell"),

    # ── 脑 / 神经系统 ────────────────────────────────────────────────────
    "bergmann glial cell":
        ("Bergmann glial cell", "CL:0000644", "glial cell"),
    "brainstem motor neuron":
        ("motor neuron", "CL:0000100", "neuron"),
    "cerebellar granule cell precursor":
        ("cerebellar granule cell", "CL:0001031", "neuron"),
    "choroid plexus epithelial cell":
        ("choroid plexus epithelial cell", "CL:0002079", "epithelial cell"),
    "ependymal cell":
        ("ependymal cell", "CL:0000065", "glial cell"),
    "gabaergic interneuron":
        ("GABAergic interneuron", "CL:0011005", "interneuron"),
    "inhibitory interneuron":
        ("inhibitory interneuron", "CL:0008030", "interneuron"),
    "interneuron":
        ("interneuron", "CL:0000099", "neuron"),
    "granule cell":
        ("granule cell", "CL:0000120", "neuron"),
    "purkinje cell":
        ("Purkinje cell", "CL:0000121", "neuron"),
    "peripheral nervous system neuron":
        ("peripheral neuron", "CL:0000108", "neuron"),
    "mesenchymal stem cell":
        ("mesenchymal stem cell", "CL:0000134", "somatic stem cell"),
    "mesenchymal cell":
        ("mesenchymal cell", "CL:0008019", "connective tissue cell"),

    # ── 胎盘 ────────────────────────────────────────────────────────────
    "decidual cell":
        ("decidual cell", "CL:0000448", "stromal cell"),
    "extravillous trophoblast":
        ("extravillous trophoblast", "CL:0002486", "trophoblast cell"),
    "glandular secretory epithelial cell":
        ("glandular epithelial cell", "CL:0000150", "epithelial cell"),
    "hofbauer cell":
        ("Hofbauer cell", "CL:0002489", "macrophage"),
    "placental villous trophoblast":
        ("villous trophoblast", "CL:0002488", "trophoblast cell"),
    "syncytiotrophoblast cell":
        ("syncytiotrophoblast cell", "CL:0000525", "trophoblast cell"),

    # ── 其他 ────────────────────────────────────────────────────────────
    "vascular associated smooth muscle cell":
        ("vascular smooth muscle cell", "CL:0000359", "smooth muscle cell"),
    "oligodendrocyte precursor cell":
        ("oligodendrocyte precursor cell", "CL:0002453", "glial cell"),
}

# 明确要丢弃的标签（字符串集合）
_DROP_LABELS: set = {
    k for k, v in FIX_TABLE.items() if v == "DROP"
}


def normalize(s: str) -> str:
    """简单归一化：去空格、小写。"""
    return (s or "").strip().lower()


def apply_fix(record: dict) -> Optional[dict]:
    """
    对单条记录应用标签修正。
    返回 None 表示应丢弃该记录。
    """
    ct = normalize(record.get("cell_type_clean", ""))
    fix = FIX_TABLE.get(ct)

    if fix is None:
        # 没有特殊处理 → 保留原样
        return record

    if fix == "DROP":
        return None

    canonical_label, cl_id, parent_label = fix

    r = dict(record)
    r["cell_ontology_label"] = canonical_label
    r["cell_ontology_id"] = cl_id
    r["cell_ontology_parent_label"] = parent_label

    # 同步更新 target 字段（供 05_make_sft_jsonl 使用）
    r["cell_type_target_label"] = canonical_label
    r["cell_type_target_id"] = cl_id

    return r


def update_alias_tsv(alias_tsv: Path, fixes: Dict[str, Tuple]) -> int:
    """
    将新映射追加到 label_aliases.tsv，跳过已存在的条目。
    返回新增行数。
    """
    # 读取已有 raw_label（用于去重）
    existing_raw: set = set()
    with open(alias_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            existing_raw.add(row.get("raw_label", "").strip().lower())

    new_rows = []
    for ct_clean, fix in fixes.items():
        if fix == "DROP":
            continue
        canonical_label, cl_id, parent_label = fix
        if ct_clean in existing_raw:
            continue
        new_rows.append({
            "raw_label": ct_clean,
            "normalized_label": canonical_label.lower(),
            "cell_ontology_id": cl_id or "",
            "parent_label": parent_label or "",
            "notes": "auto from 09_purify_labels",
        })

    if new_rows:
        with open(alias_tsv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["raw_label", "normalized_label", "cell_ontology_id", "parent_label", "notes"],
                delimiter="\t",
            )
            for row in new_rows:
                writer.writerow(row)

    return len(new_rows)


def main(dry_run: bool = False) -> None:
    # ── 读取 ─────────────────────────────────────────────────────────────
    logger.info("读取 %s", MARKER_V2)
    records = []
    with open(MARKER_V2, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("总计 %d 条记录", len(records))

    # ── 应用修正 ─────────────────────────────────────────────────────────
    kept, dropped, fixed, unchanged = [], [], 0, 0
    for rec in records:
        ct = normalize(rec.get("cell_type_clean", ""))
        result = apply_fix(rec)
        if result is None:
            dropped.append(ct)
            logger.debug("DROP: %s", ct)
        else:
            kept.append(result)
            if ct in FIX_TABLE and FIX_TABLE[ct] != "DROP":
                fixed += 1
            else:
                unchanged += 1

    logger.info("保留: %d  丢弃: %d  修正: %d  未改: %d",
                len(kept), len(dropped), fixed, unchanged)

    # ── 统计修正 ─────────────────────────────────────────────────────────
    from collections import Counter
    drop_cnt = Counter(dropped)
    print("\n=== 丢弃的样本 ===")
    for label, cnt in drop_cnt.most_common():
        print(f"  {cnt:3d}x  {label}")

    # 还有哪些 cell_type_clean 仍然没有 ontology_id？
    still_missing = [
        r["cell_type_clean"] for r in kept
        if not r.get("cell_ontology_id")
    ]
    still_missing_cnt = Counter(still_missing)
    print(f"\n=== 修正后仍缺少 ontology_id 的 ({len(still_missing)} 条) ===")
    for label, cnt in still_missing_cnt.most_common(30):
        print(f"  {cnt:3d}x  {label}")

    # ── 写文件 ───────────────────────────────────────────────────────────
    if dry_run:
        logger.info("[DRY-RUN] 不写文件")
        return

    # 备份
    logger.info("备份 → %s", MARKER_V2_BACKUP)
    shutil.copy2(MARKER_V2, MARKER_V2_BACKUP)

    # 写回
    logger.info("写回 %s", MARKER_V2)
    with open(MARKER_V2, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 更新 alias_tsv
    n_new = update_alias_tsv(ALIAS_TSV, FIX_TABLE)
    logger.info("label_aliases.tsv 新增 %d 条", n_new)

    # 写摘要
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("=== 09_purify_labels run summary ===\n\n")
        f.write(f"Input  : {len(records)} records\n")
        f.write(f"Kept   : {len(kept)}\n")
        f.write(f"Dropped: {len(dropped)}\n")
        f.write(f"Fixed  : {fixed}\n")
        f.write(f"Alias TSV new entries: {n_new}\n\n")
        f.write("Dropped labels:\n")
        for label, cnt in Counter(dropped).most_common():
            f.write(f"  {cnt:3d}x  {label}\n")
        f.write("\nStill missing ontology_id after fix:\n")
        for label, cnt in still_missing_cnt.most_common():
            f.write(f"  {cnt:3d}x  {label}\n")

    logger.info("摘要写入 %s", SUMMARY_PATH)
    print(f"\n完成：保留 {len(kept)} 条，丢弃 {len(dropped)} 条，修正 {fixed} 条")
    print("下一步：python scripts/data_prep/05_make_sft_jsonl.py")
    print("        python scripts/data_prep/06_split_and_validate_v2.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="净化 marker_examples_v2 标签质量")
    p.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = p.parse_args()
    main(dry_run=args.dry_run)
