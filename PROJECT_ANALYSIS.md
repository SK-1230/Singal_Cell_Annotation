# SCA-Specialist 项目分析报告

> 撰写日期：2026-04-06  
> 当前最优性能：Exact Accuracy 31.9% | Ontology Compatible 59.4% | Severe Error 30.4%

---

## 目录

1. [项目目录结构](#1-项目目录结构)
2. [数据流水线（01–08）详解](#2-数据流水线0108详解)
3. [训练脚本与配置](#3-训练脚本与配置)
4. [推理脚本](#4-推理脚本)
5. [三次实验分析](#5-三次实验分析)
6. [与生产可用标准的差距及改进路线](#6-与生产可用标准的差距及改进路线)

---

## 1. 项目目录结构

```
singal_cell_annotation/
│
├── config.py                          # 项目级全局路径常量（少量使用）
├── execute.txt                        # 手动记录的常用命令备忘
├── PROJECT_ANALYSIS.md                # 本文件
├── claude_code_refactor_spec.md       # 重构规格说明（历史文档）
│
├── my_models/                         # 本地模型权重（不入 git）
│   └── Qwen/
│       ├── Qwen3-4B/                  # 基础模型 4B
│       └── Qwen3-8B/                  # 基础模型 8B
│
├── data/
│   ├── download_qwen3_4b_modelscope.py    # 从 ModelScope 下载模型脚本
│   │
│   ├── raw_h5ad/                      # 从 CELLxGENE 下载的原始 h5ad 文件（不入 git）
│   ├── clean_h5ad/                    # 清洗后的 h5ad 文件（不入 git）
│   │
│   ├── meta/                          # 元数据 CSV + 各步骤运行摘要
│   │   ├── candidate_datasets.csv         # 候选数据集列表
│   │   ├── selected_datasets.csv          # 最终选中的数据集
│   │   ├── raw_export_manifest.csv        # 导出清单
│   │   ├── clean_manifest.csv             # 清洗后清单
│   │   ├── dataset_stats_merged.csv       # 多组织数据集统计
│   │   ├── obs_preview.csv                # 元数据字段预览
│   │   ├── 01_list_candidate_datasets_run_summary.txt
│   │   ├── 02_export_selected_datasets_run_summary.txt
│   │   ├── 03_clean_and_standardize_run_summary.txt
│   │   └── (各步骤 .log 文件)
│   │
│   ├── intermediate/                  # 中间产物
│   │   ├── marker_examples.jsonl          # v1 marker 样本
│   │   ├── marker_examples_v2.jsonl       # v2 marker 样本（当前使用）
│   │   ├── marker_examples_manifest.csv
│   │   └── marker_examples_v2_manifest.csv
│   │
│   ├── sft/                           # SFT 训练数据
│   │   ├── sft_records_full_v2.jsonl      # 完整 SFT 记录（含所有字段）
│   │   ├── sft_messages_v2.jsonl          # Chat 格式（含 think）
│   │   ├── sft_messages_no_think_v2.jsonl # Chat 格式（no_think，当前训练使用）
│   │   ├── sft_messages_no_think_v3.jsonl # v3 版本（未完整使用）
│   │   └── 05_make_sft_jsonl_v2_run_summary.txt
│   │
│   ├── splits/                        # 训练/验证/测试集切分
│   │   ├── train_messages_no_think_v2.jsonl   # 训练集 425 条
│   │   ├── val_messages_no_think_v2.jsonl     # 验证集 50 条
│   │   ├── test_messages_no_think_v2.jsonl    # 测试集 69 条
│   │   ├── test_full_v2.jsonl                 # 测试集完整字段版（infer 使用）
│   │   ├── test_hard_messages_no_think.jsonl  # Hard 子集 49 条
│   │   ├── test_rare_messages_no_think.jsonl  # Rare 子集 33 条
│   │   ├── test_unmapped_messages_no_think.jsonl # Unmapped 子集 36 条
│   │   ├── benchmark_manifest.csv
│   │   ├── dataset_profiles_v2.csv
│   │   └── 06_split_and_validate_v2_run_summary.txt
│   │
│   └── knowledge/                     # 知识库资源
│       ├── ontology_index.jsonl           # Cell Ontology 官方层级索引
│       ├── train_marker_kb.jsonl          # 从训练集提取的 Marker KB
│       ├── merged_marker_kb.jsonl         # 合并后 KB（外部 + 训练集），322 条目
│       ├── parent_map_supplement.json     # 人工补充的细胞类型父节点映射
│       ├── 07_build_ontology_resources_manifest.json
│       └── 08_build_marker_kb_manifest.json
│
├── scripts/
│   ├── data_prep/                     # 数据处理流水线
│   │   ├── data_prep_config.py            # 各步骤共用的路径/参数配置
│   │   ├── 01_list_candidate_datasets.py  # 从 CELLxGENE 筛选候选数据集
│   │   ├── 02_export_selected_datasets.py # 下载并导出 h5ad 文件
│   │   ├── 03_clean_and_standardize.py    # 清洗 + 标准化元数据
│   │   ├── 04_make_marker_examples.py     # 从 h5ad 提取 DE marker 基因
│   │   ├── 05_make_sft_jsonl.py           # 构建 SFT 训练样本（含 LLM distill）
│   │   ├── 06_split_and_validate.py       # v1 切分（已弃用）
│   │   ├── 06_split_and_validate_v2.py    # v2 切分（当前使用）
│   │   ├── 07_build_ontology_resources.py # 构建 Cell Ontology 层级索引
│   │   ├── 08_build_marker_kb.py          # 构建 Marker 知识库（KB）
│   │   └── evaluate_predictions.py        # 离线评估工具
│   │
│   ├── train/
│   │   ├── train_config.yaml              # 训练超参数配置文件（核心）
│   │   ├── train_qwen3_hf_trl.py          # SFT 训练主脚本（HF + TRL）
│   │   ├── run_qwen3_hf_trl.sh            # 启动训练的 shell 脚本
│   │   ├── calibrate_confidence.py        # 置信度校准工具（实验性）
│   │   ├── train_qwen3_swift.sh           # Swift 框架训练脚本（已弃用）
│   │   └── train_qwen3_swift_v2.sh        # Swift v2（已弃用）
│   │
│   └── infer/
│       ├── infer_qwen3_kb_retrieval.py    # 带 KB 检索的推理脚本（当前主力）
│       ├── infer_qwen3_hf_trl.py          # 早期 HF 推理脚本（无 KB）
│       ├── infer_qwen3_grounded.py        # Grounded 推理（实验性）
│       ├── infer_qwen3_swift.py           # Swift 推理（已弃用）
│       ├── infer_qwen3_swift_batch.py     # Swift 批量推理（已弃用）
│       └── infer_qwen3_swift_batch_V2.py  # Swift 批量推理 v2（已弃用）
│
├── src/sca/                           # 核心业务逻辑库
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── label_normalization.py     # 细胞类型名称规范化
│       ├── marker_extraction.py       # Marker 基因提取逻辑
│       ├── marker_features.py         # Marker 特征工程
│       ├── ontology_mapping.py        # 本体 ID 映射
│       ├── sft_builder.py             # SFT 样本构建
│       ├── split_builder.py           # 数据集切分
│       ├── curation_rules.py          # 数据清理规则
│       ├── split_grouping.py          # 切分分组策略
│       └── target_labeling.py         # 标签生成
│
└── output/                            # 训练 + 推理输出（不入 git）
    ├── qwen3_4b_sc_sft_hf_trl_v1/            # 早期 v1 实验模型
    ├── qwen3_4b_sc_sft_hf_trl_v2_20260331_210804/  # Exp1 对应模型
    ├── qwen3_4b_sc_sft_hf_trl_v2_20260404_014953/  # Exp2（使用 8B 数据配置）
    ├── qwen3_4b_sc_sft_hf_trl_v2_20260405_193112/  # Exp3 最新模型（当前最优）
    ├── infer_kb_retrieval_20260405_015855/    # Exp1 推理结果
    ├── infer_kb_retrieval_20260405_063125/    # Exp2 推理结果
    └── infer_kb_retrieval_20260405_211943/    # Exp3 推理结果（当前最优）
```

---

## 2. 数据流水线（01–08）详解

数据流水线从公共单细胞数据库出发，经过 8 个步骤生成训练数据和推理时的知识库。

### 01 — 筛选候选数据集（`01_list_candidate_datasets.py`）

从 CELLxGENE Discover（人类单细胞 RNA-seq 公共数据库）按以下条件筛选：
- 物种：Homo sapiens
- 组织：blood、liver、lung、kidney、brain、intestine 等主要器官
- 要求包含规范化的 `cell_type` ontology 标注
- 输出：`candidate_datasets.csv`，`selected_datasets.csv`

**当前规模**：最终选中数据集 36 个，覆盖 liver、blood、lung、kidney、brain、intestine 等 13 种组织类型。

### 02 — 导出 h5ad 文件（`02_export_selected_datasets.py`）

- 从 CELLxGENE API 下载原始 `.h5ad` 格式的单细胞数据集
- 导出 obs（细胞元数据）和基因表达矩阵
- 记录导出状态到 `raw_export_manifest.csv`

### 03 — 清洗与标准化（`03_clean_and_standardize.py`）

- 处理 obs_names 维度不匹配的 h5ad 文件（共 39 个文件失败，原因为原始数据维度不一致）
- 标准化 `cell_type` 字段，映射至 Cell Ontology 标准名称
- 使用 `src/sca/data/label_normalization.py` 处理大小写、连字符、缩写等差异
- 输出 `clean_manifest.csv`，记录成功处理的 34 个数据集

**问题**：原始数据质量参差不齐，约 50% 的数据集因格式问题跳过，是数据规模受限的核心原因。

### 04 — 提取 Marker 基因（`04_make_marker_examples.py`）

对每个清洗后的数据集，按 `cell_type` 分组做差异表达分析：
- 方法：Wilcoxon 秩和检验（与 Scanpy/Seurat 一致）
- 每个 cluster 保留 Top-20 正向 marker（按 logFC × pct_ratio 排序）
- 计算 `marker_quality_score`（基于 pct_in/pct_out 分离度）
- 标记难度 flag：`low_cells`、`mixed_label`、`low_marker_quality` 等
- 输出：`marker_examples_v2.jsonl`（中间文件，共 544 个 cluster-level 样本）

### 05 — 构建 SFT 训练样本（`05_make_sft_jsonl.py`）

将 marker 样本转为 LLM 可直接训练的对话格式：
- **System prompt**：明确任务（单细胞 RNA-seq 细胞类型注释）、输出格式（JSON）
- **User message**：包含 Organism、Tissue、Disease、Cluster size、Top marker 列表（含 logFC、pct_in、pct_out）
- **Assistant message**：JSON 格式输出，包含 `cell_type`、`cell_ontology_id`、`parent_cell_type`、`supporting_markers`、`confidence_label`、`confidence_score`、`need_manual_review`、`decision`、`novelty_flag`、`evidence_support_level`、`rationale`

**答案生成方式（Knowledge Distillation）**：
- 使用 Qwen3-4B 自身推理生成 rationale 和 confidence，再以规则后处理校正 cell_type 和 ontology_id
- 这导致了答案中存在"自洽但不精确"的问题（模型学的是自己的输出风格，而非专家标注）

**v2 版本输出**：544 条原始记录，其中：
- 置信度分布：high 337 / medium 202 / low 5
- decision 分布：accept 337 / review 158 / novel_candidate 47
- Ontology 映射成功：250/544（45.9%）

### 06 — 切分与验证（`06_split_and_validate_v2.py`）

按数据集粒度（dataset-level）而非样本粒度切分，避免训练集/测试集数据泄露：
- 训练集：27 个数据集，425 个样本
- 验证集：4 个独立数据集，50 个样本（`independent_dataset_val` 模式）
- 测试集：5 个独立数据集，69 个样本
- Hard 子集：49 个（低细胞数、低 marker 质量等 hardness flag）
- Rare 子集：33 个（训练集中出现 ≤2 次的细胞类型）
- Unmapped 子集：36 个（Ontology ID 未能映射的样本）

**当前数据集的核心限制**：
- 训练集 188 种 unique 细胞类型中，有 **98 种只有 1 个训练样本**（约 52%），严重的 long-tail 分布
- 测试集有 **9 种细胞类型从未出现在训练集中**（zero-shot 场景）
- 总样本量 544 条，对于 Fine-tuning 细粒度分类任务而言数量极少

### 07 — 构建 Cell Ontology 资源（`07_build_ontology_resources.py`）

- 解析 Cell Ontology（OBO 格式）建立层级父子关系索引
- 生成 `ontology_index.jsonl`，每条记录包含 `label`、`ontology_id`、`parent_label`、`parent_id`
- 用于推理时的 Ontology Compatible 评估和层级关系判断

### 08 — 构建 Marker 知识库（`08_build_marker_kb.py`）

构建 RAG 检索使用的 Marker Knowledge Base：
- **外部 KB**：26 条人工整理的经典 marker 条目
- **训练集衍生 KB**：544 条来自训练数据的 marker 样本
- **合并去重后**：322 条目（以细胞类型为 key，合并同类型的 marker 证据）
- **检索方式**：Jaccard 相似度（基于 marker 基因集合交集/并集）
- **额外资源**：`parent_map_supplement.json`——人工补充的约 50 条 organ-specific 子类的父节点映射（如 `lung macrophage` → `macrophage`）

---

## 3. 训练脚本与配置

### `train_config.yaml` — 当前训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_size` | `"4B"` | 使用 Qwen3-4B，bf16 全精度 LoRA |
| `train_file` | `data/splits/train_messages_no_think_v2.jsonl` | 425 条训练样本 |
| `val_file` | `data/splits/val_messages_no_think_v2.jsonl` | 50 条验证样本 |
| `num_train_epochs` | 5 | 训练 5 轮 |
| `per_device_train_batch_size` | 2 | 每步 2 条样本 |
| `gradient_accumulation_steps` | 8 | 等效 batch size = 16 |
| `learning_rate` | 2.0e-5 | 余弦衰减，warmup 10% |
| `lora_r` | 8 | LoRA 秩，低容量防过拟合 |
| `lora_alpha` | 16 | alpha/r = 2 |
| `lora_dropout` | 0.1 | 较强的 dropout 正则化 |
| `lora_target_modules` | `"all-linear"` | 对所有线性层注入 LoRA |
| `max_length` | 1024 | 序列最大长度 |
| `use_bf16` | true | bf16 混合精度 |
| `gradient_checkpointing` | true | 节省显存约 30-50% |

### `train_qwen3_hf_trl.py` — 训练主脚本

- 框架：HuggingFace Transformers + TRL（SFTTrainer）
- 支持 4B（bf16 LoRA）和 8B（QLoRA 4-bit）两种模式
- 自动从 `train_config.yaml` 读取所有超参数
- 训练完成后保存 adapter 权重（非全量权重）、tokenizer、训练曲线图
- 支持 `--config` 参数指定配置文件路径

**运行方式**：
```bash
cd /data/projects/shuke/code/singal_cell_annotation
CUDA_VISIBLE_DEVICES=0 python -u scripts/train/train_qwen3_hf_trl.py
```

---

## 4. 推理脚本

### `infer_qwen3_kb_retrieval.py` — 当前主力推理脚本

**推理流程（RAG + LLM）**：

```
测试样本 (marker genes)
    ↓
Jaccard 相似度检索 Marker KB（322 条目）
    ↓
检索到 Top-K 候选细胞类型及其 marker 证据
    ↓
追加到 user message 末尾作为"参考提示"
    ↓
Qwen3-4B + LoRA adapter 生成 JSON 输出
    ↓
解析 cell_type, cell_ontology_id 字段
    ↓
与 gold_cell_type 对比，计算多维度指标
```

**评估指标体系**：

| 指标 | 含义 | 严格程度 |
|------|------|---------|
| `cell_type_exact_accuracy` | 规范化后名称完全一致 | 最严格 |
| `ontology_compatible_accuracy` | 预测是 gold 的祖先/后代/同义词 | 次严格 |
| `cell_type_same_lineage_rate` | 预测与 gold 属于同一细胞谱系 | 宽松 |
| `cell_type_severe_error_rate` | 谱系完全不同（如 T 细胞→肝细胞）| 最严重错误 |
| `pred_more_general_rate` | 预测比 gold 更粗粒度（如 T 细胞→macrophage） | 粒度偏差 |
| `pred_more_specific_rate` | 预测比 gold 更细粒度 | 粒度偏差 |
| `cell_ontology_id_accuracy` | Ontology ID 完全匹配 | 最严格 |
| `parse_ok_rate` | 输出能被解析为合法 JSON | 格式合规性 |

---

## 5. 三次实验分析

### 实验总览

| 实验 | 模型 | 训练数据 | Adapter | Exact | Ontology Compat | Same Lineage | Severe Error |
|------|------|----------|---------|-------|-----------------|--------------|--------------|
| **Exp1** | 4B bf16 | v1 数据（较早版本） | `qwen3_4b_sc_sft_hf_trl_v2_20260331_210804` | 21.7% | 49.3% | 62.3% | 36.2% |
| **Exp2** | 4B bf16 | v2 数据（新配置，早期训练） | `qwen3_4b_sc_sft_hf_trl_v2_20260404_014953` | 14.5% | 30.4% | 49.3% | 50.7% |
| **Exp3** | 4B bf16 | v2 数据（最新超参） | `qwen3_4b_sc_sft_hf_trl_v2_20260405_193112` | **31.9%** | **59.4%** | **69.6%** | **30.4%** |

### Exp1（20260405_015855）— 基线建立

**模型**：Qwen3-4B，LoRA 微调，旧版数据集  
**训练特征**：total_steps=1392，best_eval_loss=0.0705，final_train_loss=0.024  
**问题**：eval loss 极低（0.07）但 exact accuracy 仅 21.7%，说明模型过拟合了答案的格式/风格，但没有真正学到 marker→细胞类型 的映射关系。train loss 下降太快而 exact accuracy 停滞，是典型的"记住格式但没学到内容"症状。

### Exp2（20260405_063125）— 失败实验

**模型**：4B 配置，但 `014953` 训练了 856 步，train_loss=0.003，严重过拟合  
**结果**：Exact 跌至 14.5%，Severe Error 高达 50.7%  
**根本原因**：
1. 训练轮数过多（等效 8+ epoch），模型死记硬背训练集
2. 旧超参（lora_r=16, dropout=0.05, batch=4）正则化不足
3. KB 检索作为 RAG 引入了噪声，模型无法合理利用检索结果

### Exp3（20260405_211943）— 当前最优

**模型**：Qwen3-4B，`qwen3_4b_sc_sft_hf_trl_v2_20260405_193112`  
**训练特征**：total_steps=135，best_eval_loss=0.3303（step 130），step 135 回升至 0.3306  
**调整内容**：
- epochs 5→lora_r 16→8，dropout 0.05→0.1，batch size 4→16（梯度累积×8）
- eval loss 从 0.378（step 90）平滑下降至 0.330（step 130），收敛稳定

**错误分布分析（47 个 non-exact 样本）**：

**严重错误（21 个，30.4%）**——谱系完全错误，生产环境不可接受：
- 脑组织：把 `lymphocyte` 预测为 `CD8+ T cell`，把 `pericyte` 预测为 `endothelial cell`
- 肺组织：把 `erythrocyte` 预测为 `PBMC`，把 `smooth muscle cell` 预测为 `myofibroblast`
- 肾组织：把 `leukocyte` 预测为 `kidney interstitial epithelial cell`（跨谱系混淆）
- 根本原因：9 种 zero-shot 细胞类型从未出现在训练集（如 `lung goblet cell`、`kidney cell`），模型退化为最近邻猜测

**可接受错误（26 个）**——粒度或同义词问题：
- 预测更粗粒度（9个）：`lung macrophage→macrophage`、`lung endothelial cell→endothelial cell`（模型偷懒给了泛化答案）
- 预测更细粒度（10个）：`B cell→plasma cell`、`NK cell→CD56 dim NK cell`
- 同谱系偏差（7个）：`pulmonary alveolar type 1 cell→type 2 cell`、`vein endothelial→artery endothelial`

**Ontology ID 准确率（42.4%，n=33）**：在 exact cell_type 正确的基础上，约 13% 的样本 ID 还是错了，说明模型对 ontology ID 体系理解不够深。

---

## 6. 与生产可用标准的差距及改进路线

### 生产可用的性能基准

对于一个能在真实单细胞分析流程中作为自动注释工具部署的模型，业界参考标准（基于 scGPT、CellTypist 等工具在 hold-out 数据集上的表现）大致如下：

| 指标 | 当前水平 | 生产可用目标 | SOTA 目标 |
|------|---------|------------|----------|
| Exact Accuracy | **31.9%** | ≥ 70% | ≥ 85% |
| Ontology Compatible | **59.4%** | ≥ 85% | ≥ 95% |
| Same Lineage | **69.6%** | ≥ 92% | ≥ 98% |
| Severe Error Rate | **30.4%** | ≤ 5% | ≤ 2% |
| Ontology ID Accuracy | **42.4%** | ≥ 75% | ≥ 90% |

**结论**：当前模型距离生产可用还有 2–3 倍的性能提升空间。Severe Error 30.4% 是最关键的问题——这意味着每三个预测就有一个是完全离谱的，在实际分析中会产生严重误导。

### 差距的根本原因分析

#### 数据层面（最关键）

1. **数据量严重不足**：
   - 当前 544 条训练样本，覆盖 188 种细胞类型
   - 98 种细胞类型（52%）只有 1 个训练样本——模型根本无法学习其特征
   - CellTypist（业界 SOTA）使用了 >30 万个细胞、>300 种细胞类型的训练数据

2. **组织覆盖不均衡**：
   - 训练集以 blood/liver（合计 57%）为主，但测试集有 40 条 lung 样本（58%）
   - kidney 和 brain 在训练中严重欠代表，导致 kidney 预测混淆率极高

3. **答案质量问题（Knowledge Distillation 的副作用）**：
   - SFT 答案由模型自身生成，不是专家标注——模型在学习"自己的猜测"
   - 250 个样本（46%）的 ontology_id 是未能从 CL 数据库确认的自造 ID

4. **Zero-shot 场景无法处理**：
   - 测试集 9 种细胞类型从未出现在训练集，触发严重错误

#### 模型层面

5. **基础模型参数量不足**：
   - Qwen3-4B 对专业生物医学知识的覆盖有限
   - 4B 模型在细粒度专业分类（如区分 type 1 vs type 2 alveolar cell）上能力受限

6. **LoRA 微调信号太弱**：
   - 只有 425 个训练样本让 LoRA 学习 188 种细胞类型的映射，信号/参数比过低

#### 推理层面

7. **KB 检索质量有限**：
   - Jaccard 相似度是词级别匹配，无法处理同义 marker（如同一基因的别名）
   - KB 仅 322 条目，覆盖度不足

### 具体改进路线（按优先级排序）

---

#### 优先级 1：大幅扩充训练数据（最高 ROI）

**目标**：从 544 条扩充到 5000+ 条，覆盖 500+ 种细胞类型

**方案 A：扩大 CELLxGENE 数据集采集范围**
- 当前只成功处理了 34/77 个数据集（失败 39 个）
- 修复 `03_clean_and_standardize.py` 中的维度不匹配问题（obs_names 长度与 AnnData shape 不一致），可恢复 ~50% 失败数据集
- 扩展组织类型覆盖：增加 heart、skin、pancreas、thyroid、muscle 等
- 优先增加 lung、kidney、brain 的数据集（当前测试集主要组织）

**方案 B：使用高质量公共标注数据集**
- Human Cell Atlas（HCA）数据门户提供专家标注的大规模数据
- Tabula Sapiens：428 种细胞类型，~500k 细胞，覆盖 24 种组织，专家标注
- 直接以这些专家标注作为 ground truth，不再依赖 LLM distillation

**方案 C：改进 SFT 答案质量**
- 将当前的 LLM self-distillation 替换为：用更强的模型（GPT-4o 或 Claude Sonnet）生成 rationale
- 或直接使用 CellTypist 的预测结果作为 label，再用 LLM 生成 rationale
- 确保所有 cell_ontology_id 来自官方 Cell Ontology 数据库

---

#### 优先级 2：解决 Zero-shot 问题（消除严重错误）

**方案**：增强 KB 检索，让模型在未见过细胞类型时也能给出合理预测

- 将 Jaccard 相似度替换为**向量检索**（如 BM-25 + dense embedding 混合检索）
- 使用 marker 基因专用嵌入模型（如 GenePT、scFoundation）将 marker 转为语义向量
- 构建更大的外部 KB：整合 CellMarker 2.0（>3000 种细胞类型的 marker 数据库）、PanglaoDB

---

#### 优先级 3：升级基础模型

**方案 A：使用生物医学预训练 LLM**
- 替换 Qwen3-4B 为专门在生物医学文献上预训练的模型：
  - BioMedLM（2.7B，Stanford，生物医学专用）
  - LLaMA3-Med（Meta，医学领域）
  - 或更强的通用模型 Qwen3-7B/14B（生物医学知识覆盖更广）

**方案 B：全量微调而非 LoRA**
- 当数据量达到 5000+ 时，考虑对 4B 模型做全量 SFT（而非 LoRA）
- 或使用更大的 LoRA rank（r=32/64）配合更多数据

---

#### 优先级 4：改进训练策略

**方案 A：数据增强**
- 对同一个 cluster，用不同的 marker subset（Top-10、Top-15、Top-20）生成多条训练样本
- 对 marker 基因列表做随机 shuffle（marker 排序不应影响预测）
- 在 system prompt 中加入随机化的措辞变体

**方案 B：课程学习（Curriculum Learning）**
- 先训练 high confidence（accept）样本，再加入 review/novel 样本
- 先训练常见细胞类型（≥5个训练样本），再加入 rare 类型

**方案 C：改进损失函数**
- 对 severe error（谱系错误）的样本加大惩罚权重
- 使用 Ontology-aware loss：同谱系错误的惩罚 < 跨谱系错误的惩罚

**方案 D：多任务学习**
- 同时训练：细胞类型预测 + Ontology ID 预测 + 置信度预测
- 让模型学会什么时候说"I don't know"（novelty_flag=True）而不是乱猜

---

#### 优先级 5：后处理与工程优化

**方案 A：基于规则的后处理**
- 对输出的 cell_type 名称做规范化后，查询 Cell Ontology 数据库核实 ID
- 如果预测的 cell_type 在 CL 中不存在，回退到父类型

**方案 B：集成多模型**
- 训练多个不同种子/不同超参的模型，取 majority vote
- 在置信度低时（confidence_score < 0.6）自动标记为 need_manual_review

**方案 C：评估体系扩展**
- 当前只在 69 条测试样本上评估，样本量太小（置信区间约 ±12%）
- 扩充测试集至 500+ 条，覆盖更多组织类型和细胞类型
- 增加 cross-dataset 泛化评估：在完全陌生的数据集上测试

---

### 改进路线里程碑

```
当前状态                 近期目标（~1-2月）        中期目标（~3-6月）        生产可用
Exact: 31.9%      →      Exact: 50%+         →      Exact: 70%+         →  Exact: 85%+
Severe: 30.4%     →      Severe: ≤ 15%       →      Severe: ≤ 8%        →  Severe: ≤ 5%
n_train: 544      →      n_train: 2000+      →      n_train: 5000+      →  n_train: 10000+
n_celltypes: 188  →      n_celltypes: 300+   →      n_celltypes: 500+   →  n_celltypes: 800+
tissues: 5        →      tissues: 10+        →      tissues: 20+        →  tissues: all major

关键动作：
近期：修复03脚本失败案例，扩充CELLxGENE数据，引入Tabula Sapiens
中期：替换LLM-distill为专家标注，升级KB检索为向量检索，引入CellMarker 2.0
长期：考虑全量SFT或更大模型，部署REST API，与Scanpy/Seurat工作流集成
```

---

### 一句话总结

> 当前模型已经建立了正确的技术框架（RAG+LLM、本体评估、KB检索），但受制于极小的训练数据量（544条、188种细胞类型）和数据质量问题（LLM自蒸馏而非专家标注），导致严重错误率高达30%。**最高优先级的改进是数据扩充**——将训练样本增至5000+条，引入专家标注的公共数据集（Tabula Sapiens、HCA），覆盖更多组织类型，这一步预计可将性能提升至接近生产可用水平。
