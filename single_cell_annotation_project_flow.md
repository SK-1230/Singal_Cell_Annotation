
# 单细胞注释项目整体流程说明

## 1. 项目定位

本项目是一个围绕 **单细胞 RNA-seq 细胞类型注释** 构建的完整工程流程。核心目标是搭建一条从 **公开数据获取、清洗标准化、标签净化、marker 证据提取、SFT 数据构建、知识资源构建，到大模型微调、知识检索增强推理和自动评测** 的全链路管线。

从工程视角看，这个项目主要解决了几类问题：

1. 如何从公开单细胞数据库中自动筛选出适合做细胞类型注释的数据集。
2. 如何把原始 h5ad 数据整理成适合大模型学习的结构化监督样本。
3. 如何对训练标签进行 Cell Ontology 标准化，消除标签噪声。
4. 如何把 marker gene 证据转成模型可学习的 no-think 指令数据。
5. 如何在本地 Qwen3-4B 上开展 LoRA 微调，得到领域化注释模型。
6. 如何结合 marker 知识库做检索增强推理，并从生物学层面做分层评估。
7. 如何系统诊断模型性能瓶颈（标签噪声、KB 贡献、误差分桶）。

---

## 2. 项目目录结构说明

```text
singal_cell_annotation/
├── scripts/
│   ├── data_prep/
│   │   ├── 01_list_candidate_datasets.py
│   │   ├── 02_export_selected_datasets.py
│   │   ├── 03_clean_and_standardize.py
│   │   ├── 04_make_marker_examples.py
│   │   ├── 05_make_sft_jsonl.py
│   │   ├── 06_split_and_validate_v2.py
│   │   ├── 07_build_ontology_resources.py
│   │   ├── 08_build_marker_kb.py
│   │   ├── 09_purify_labels.py
│   │   └── data_prep_config.py
│   ├── train/
│   │   ├── run_qwen3_hf_trl.sh
│   │   ├── train_qwen3_hf_trl.py
│   │   └── train_config.yaml
│   ├── infer/
│   │   └── infer_qwen3_kb_retrieval.py
│   └── diagnosis/
│       ├── analyze_label_noise.py
│       ├── analyze_error_buckets.py
│       ├── ablate_kb_retrieval.py
│       ├── ablate_ontology_target.py
│       ├── ablate_output_schema.py
│       ├── generate_diagnosis_report.py
│       └── run_all_diagnosis.py
├── src/sca/
│   ├── data/
│   │   ├── curation_rules.py
│   │   ├── split_grouping.py
│   │   └── target_labeling.py
│   └── diagnosis/
│       ├── metrics.py
│       ├── bucket_analysis.py
│       ├── label_quality.py
│       └── report_utils.py
├── data/
│   ├── raw_h5ad/
│   ├── clean_h5ad/
│   ├── intermediate/
│   ├── knowledge/
│   ├── meta/
│   ├── sft/
│   └── splits/
├── resources/
│   └── ontology/
│       └── label_aliases.tsv
├── my_models/Qwen/Qwen3-4B/
└── output/
```

---

## 3. 整体流程总览

整个项目拆成三大阶段：

### 第一阶段：数据准备与知识构建（01–09）

- 从 CELLxGENE Census 筛选候选数据集并导出 h5ad
- 清洗标签、加入 ontology 字段
- 用 `09_purify_labels.py` 对标签做 CL 标准化净化
- 基于差异表达分析提取 marker gene
- 将 marker 样本转成 Qwen3 SFT 对话数据（v3 canonical labels）
- 按 dataset 级别切分 train / val / test
- 构建 ontology 索引和 marker 知识库

### 第二阶段：模型训练

- 使用 HF TRL `SFTTrainer` + PEFT LoRA 对 Qwen3-4B 做 SFT
- 所有超参数通过 `train_config.yaml` 配置
- 训练数据为 v3 no-think 格式，assistant 端使用 canonical CL 标签

### 第三阶段：推理与评估

- 加载 base model + LoRA adapter
- 推理前从 marker KB 做 Jaccard 检索增强
- 自动解析 JSON 输出，做字段级和谱系级评估
- 配套诊断工具定位性能瓶颈

---

## 4. 数据准备流程详解（01–09）

### 4.1 01_list_candidate_datasets.py：候选数据集筛选

整个流程的入口。基于 CELLxGENE Census 元数据，从大量公开数据集中自动筛出适合后续使用的数据集。

**主要工作**：
1. 按 `TARGET_TISSUES` 逐个查询 Census，读取 `dataset_id`、`cell_type`、`tissue_general`、`disease` 等字段。
2. 聚合得到 dataset 级统计：`cell_count`、`unique_cell_types`、`tissues`、`diseases`。
3. 按配置阈值过滤（细胞数上下限、最少细胞类型数）。
4. 自动输出 `candidate_datasets.csv` 和 `selected_datasets.csv`（`SELECT_MODE = "auto_balanced"`）。

**输出**：候选数据集表、selected 数据集表、各组织统计 CSV、run summary

> 把"哪些数据值得进入训练流程"自动化。

---

### 4.2 02_export_selected_datasets.py：导出原始 h5ad

在 01 选好数据集之后，真正从 Census 导出表达矩阵。

**主要工作**：
1. 读取 `selected_datasets.csv`，仅处理 `use=1` 的数据集。
2. 调用 `cellxgene_census.get_anndata()` 拉取 AnnData，保存为 `.h5ad`。
3. 工程保护：已存在则跳过，失败自动重试，中途失败删除半成品，每完成一个 dataset 就更新 manifest。

**输出**：`data/raw_h5ad/*.h5ad`、raw 导出 manifest、run summary

> 把已选中的 dataset 真正落地到本地。

---

### 4.3 03_clean_and_standardize.py：清洗与标准化

把原始 h5ad 整理成适合训练的 clean h5ad。

**主要工作**：
1. 统一基因名（优先使用 `feature_name`，保证 `var_names` 唯一）。
2. 标准化 obs 字段：`cell_type`、`tissue_general`、`tissue`、`disease`。
3. 过滤无效样本：非 primary data、缺失/含糊标签、过少基因/细胞。
4. 加入 ontology 字段：`cell_ontology_id`、`cell_type_status`、`cell_type_level`。
5. 保留 `layers["counts"]` 以便后续 marker 提取。
6. 支持断点续跑。

**输出**：`data/clean_h5ad/*.h5ad`、标签统计、clean manifest、run summary

> 把原始单细胞数据整理成统一、干净的标准输入。

---

### 4.4 04_make_marker_examples.py：提取 marker 样本

围绕细胞类型做差异表达分析，把"表达矩阵 + 标签"转化为"细胞类型注释证据样本"。

**主要工作**：
1. 读取 clean h5ad，对每类细胞做下采样避免大类压制。
2. 用 `counts` 层做归一化和 log1p，调用 `scanpy.tl.rank_genes_groups()` 做差异分析。
3. 提取 top positive markers，过滤线粒体/核糖体等无信息基因。
4. 构建 marker example 记录：dataset 信息、tissue/disease 背景、cell_type、marker 基因列表及 logFC/p 值。
5. v2 增强版还包含 negative markers、ontology 信息、`marker_quality_score`、`hardness_flags`。

**注意**：一条样本 = 一个 cell type 的 marker 证据记录，不是单个细胞。

**输出**：`data/intermediate/marker_examples.jsonl`（及 v2 版本）、manifest、run summary

---

### 4.5 05_make_sft_jsonl.py：构建 SFT 对话数据

把 marker 记录包装成 Qwen3 可训练的 messages 格式。

**主要工作**：
1. 读取 `marker_examples.jsonl`（或 v2 版）。
2. 构建 user prompt：写入 organism、tissue、disease/context、ranked marker genes 和任务要求。
3. 构建 assistant 端答案：**v3 builder** 直接从 marker 记录的 `cell_type_target_id` / `cell_ontology_id` 字段读取 canonical CL 标签和 ID，不经过 LLM 生成，消除了 v2 的标签噪声问题。
4. 输出标准 messages 和 no-think 兼容版（user 末尾加 `/no_think`，assistant 含空 `<think></think>`）。

**输出**：`data/sft/sft_records_full_v3.jsonl`、`sft_messages_v3.jsonl`、`sft_messages_no_think_v3.jsonl`、manifest

> 把 marker 样本变成大模型能学的"指令—回答"监督数据。

---

### 4.6 06_split_and_validate_v2.py：切分 train / val / test

按 `dataset_id` 级别切分，保证评测无泄漏。

**主要工作**：
1. 构建 dataset profile（主组织、主疾病、样本数、细胞类型数）。
2. 优先按主组织分层切分；dataset 数不足时构造 pseudo-val。
3. 额外导出 hard test 子集。
4. 对 v1/v2/v3 复用相同 dataset split 划分，保证可比性。

**输出**：`data/splits/train/val/test_messages_no_think_v3.jsonl` 等（各版本）、dataset_profiles.csv、run summary

---

### 4.7 07_build_ontology_resources.py：构建 ontology 资源

基于 `resources/ontology/label_aliases.tsv` 构建本地轻量级 ontology 索引。

**主要工作**：
- 按 `cell_ontology_id` 聚合别名表，得到标准标签、同义词、parent label、organ scope、label level。
- 写出 JSONL 格式供运行时查表。

**输出**：`data/knowledge/ontology_index.jsonl`、`cell_ontology_min.jsonl`、manifest

---

### 4.8 08_build_marker_kb.py：构建 marker 知识库

将外部整理的 marker 知识与训练数据提炼的 marker 合并成统一 KB。

**主要工作**：
- 读取外部 marker 数据库 + `marker_examples_v2.jsonl`。
- 按 `(species, tissue, label)` 去重合并，外部知识优先。
- 统一 schema：`species`、`tissue_general`、`cell_type_label`、`cell_ontology_id`、`parent_label`、`marker_genes`。

**输出**：`data/knowledge/train_marker_kb.jsonl`、`merged_marker_kb.jsonl`、manifest

---

### 4.9 09_purify_labels.py：标签净化

对 marker 记录的 cell_type 标签进行 CL 标准化，这是提升 ontology_id 准确率的关键步骤。

**主要工作**：
- 批量检查 marker records 中的 `cell_type_clean` 是否有对应的 canonical CL 名和 CL ID。
- 修复错误映射，将非标准写法统一到 Cell Ontology 规范名称。
- 过滤恶性/异常细胞标签（不适合作为训练目标）。
- 同步扩展 `resources/ontology/label_aliases.tsv`（本次净化新增 151 条）。
- 净化后重新运行 04→06 更新全部训练数据。

**效果**：训练样本从 544 条扩展至 1341 条，ontology_id accuracy 从 1.1% 提升至 43.9%。

> 这一步是 v3 数据质量提升的核心，消除了 v2 版本中 CL ID 几乎全错的问题。

---

## 5. 训练流程详解

数据准备完成后进入训练阶段，使用 **Hugging Face TRL + PEFT**，对本地 Qwen3-4B 做 LoRA 监督微调。

### 5.1 训练配置

所有超参数在 `scripts/train/train_config.yaml` 中配置。当前 v3 推荐配置：

```yaml
model_size: "4B"
train_file: "data/splits/train_messages_no_think_v3.jsonl"
val_file:   "data/splits/val_messages_no_think_v3.jsonl"
num_train_epochs: 10
per_device_train_batch_size: 2
gradient_accumulation_steps: 8    # 等效 batch=16
learning_rate: 2.0e-5
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: "all-linear"
max_length: 1536
use_bf16: true
gradient_checkpointing: true
```

显存估算：4B bf16 + r=16 + batch=2 + gc ≈ 14GB，适合单张 24GB GPU。

### 5.2 数据转换逻辑

训练脚本先做 prompt-completion 转换：
1. 找到每条样本中最后一条 assistant 消息。
2. 将其前面的 system + user 作为 prompt。
3. 用 `apply_chat_template()` 渲染成模型真正使用的 token 序列。
4. 只在 completion 部分计算 loss（system/user 部分 loss mask 掉）。

### 5.3 训练前检查

- 检查训练/验证文件是否存在且非空。
- 逐条校验 messages 格式，跳过无效样本并打印统计。
- 打印少量样本 preview，便于人工核查 prompt/completion。

### 5.4 训练器与保存逻辑

使用 `SFTTrainer`，按 step 记录日志和验证，基于 `eval_loss` 自动保留 best model。训练结束保存 adapter、tokenizer、metrics 和 trainer state。

---

## 6. 推理与评估流程详解

### 6.1 KB 检索增强推理

推理脚本（`infer_qwen3_kb_retrieval.py`）在生成前先从 `merged_marker_kb.jsonl` 做检索增强：

1. 提取 user prompt 中的 marker gene 列表。
2. 用 Jaccard 相似度在 KB 中检索最相关的条目。
3. 把检索结果注入到 prompt context 中。
4. 然后调用 `model.generate()` 生成输出。

**KB 检索的贡献**（消融实验验证）：+21.7pp exact accuracy。

### 6.2 输出解析

模型输出不一定总是干净的 JSON，脚本做了鲁棒解析：
1. 先直接尝试整体 JSON 解析。
2. 失败则移除 `</think>` 干扰，再尝试正则匹配第一个 `{...}` 块。
3. 解析失败记录 `parse_ok = False`。

### 6.3 多层级评估指标

| 指标 | 含义 |
|------|------|
| `cell_type_exact_accuracy` | 严格字符串匹配 |
| `ontology_compatible_accuracy` | ontology 兼容（父子关系也算对） |
| `cell_type_same_lineage_rate` | 同一大谱系（T cell、B cell 等） |
| `cell_type_severe_error_rate` | 跨谱系严重错误 |
| `pred_more_general_rate` | 预测比 gold 更粗（父节点） |
| `pred_more_specific_rate` | 预测比 gold 更细（子节点） |
| `cell_ontology_id_accuracy` | CL ID 准确率 |
| `parse_ok_rate` | JSON 解析成功率 |

### 6.4 v3 模型当前结果

训练集：1341 条，测试集：215 条（10 个数据集）

| 指标 | v3 模型 |
|------|---------|
| exact accuracy | 26.5% |
| ontology_compatible | 39.1% |
| same_lineage | 64.2% |
| severe_error | 35.8% |
| ontology_id accuracy | 43.9% |
| parse_ok_rate | 100% |

---

## 7. 诊断框架

`scripts/diagnosis/` 提供系统化诊断能力，用于定位训练/评估瓶颈。

### 7.1 诊断模块

| 脚本 | 功能 |
|------|------|
| `analyze_label_noise.py` | 量化训练标签噪声，识别高噪声标签 |
| `analyze_error_buckets.py` | 对错误样本分桶（tissue/cell_type/hardness） |
| `ablate_kb_retrieval.py` | KB 消融：with vs without KB |
| `ablate_ontology_target.py` | ontology 目标消融 |
| `ablate_output_schema.py` | 输出 schema 消融 |
| `generate_diagnosis_report.py` | 汇总生成诊断报告 |

`src/sca/diagnosis/` 提供底层工具库（metrics.py、bucket_analysis.py、label_quality.py、report_utils.py）。

### 7.2 主要诊断结论

- **标签噪声**：33.7% 的训练样本存在 cell_type 标签问题（v2 时期）
- **KB 贡献**：+21.7pp exact accuracy（消融对比）
- **v3 vs v2**：ontology_id accuracy +42.8pp，exact accuracy +8.8pp，severe_error -6.1pp

---

## 8. 整个项目的数据流转关系

```text
CELLxGENE Census
   ↓
01 筛选合适数据集
   ↓
02 导出 raw h5ad
   ↓
03 清洗、标准化、ontology 映射
   ↓ ←——————————————— 09 标签净化（循环回 04）
04 提取 marker examples
   ↓
05 构建 SFT messages（v3 canonical labels）
   ↓
06 切分 train / val / test（dataset-level）
   ↓                      ↑
07 ontology 资源 ——————→   |
08 marker KB ——————————→ 推理检索增强
   ↓
训练：Qwen3-4B + LoRA（HF TRL）
   ↓
推理：KB 检索增强 + 结构化生成
   ↓
评估：字段级 + 谱系级 + CL ID 准确率
   ↓
诊断：标签噪声 / 误差分桶 / KB 消融
```

---

## 9. 项目设计关键特点

**dataset-level 切分**：按 `dataset_id` 切分，避免同一数据集既出现在训练又出现在测试中，保证评测真实。

**marker 驱动的任务建模**：不直接处理表达矩阵，围绕 marker gene 证据构造任务，贴近人工注释的实际思路。

**v3 canonical labels**：assistant 端标签直接来自 CL ontology，不经过 LLM 生成，从根本上消除了标签噪声累积问题。

**结构化 JSON 输出**：模型学习输出带明确字段的 JSON，使自动评测、误差分析和系统集成都更方便。

**知识库增强推理**：推理时检索相关 marker KB 条目注入 prompt，弥补小模型先验知识不足。

**多层级评估**：不止看 exact accuracy，还看 ontology 兼容性、谱系一致性和 severe error，更贴近生物学任务实际。

---

## 10. 项目总结

> 本项目构建了一条面向单细胞 RNA-seq 细胞类型注释的完整工程流程：从 CELLxGENE Census 自动筛选并导出数据集，经清洗、标签净化（09_purify_labels）和 ontology 标准化后，基于差异表达分析提取 marker gene，构建 v3 canonical label SFT 指令数据，按 dataset-level 切分训练/验证/测试集；在此基础上，使用 HF TRL 对本地 Qwen3-4B 进行 LoRA 监督微调；推理阶段结合 marker 知识库做检索增强，输出结构化 JSON 注释结果，并通过字段级、谱系级和 CL ID 准确率等多层指标进行系统评估；配套诊断框架可定量分析标签噪声、KB 贡献和误差分布。
