# Single-Cell Cell Type Annotation with Qwen3

一个面向 **single-cell RNA-seq（scRNA-seq）细胞类型注释** 的数据构建、大模型微调与推理评估项目。

本项目以 **CELLxGENE Census** 为主要数据源，搭建了一条从候选数据集筛选、标准化导出、清洗标准化、标签净化、marker 提取、SFT 数据构造、知识库构建，到 **Qwen3-4B + LoRA** 微调、知识检索增强推理与结构化评估的完整端到端流程。

---

## 项目概览

核心思路：不把单细胞表达矩阵原样送入大模型，而是先从标准化单细胞数据中提取 **cell-type 级别的 marker gene 证据**，再构建适合语言模型学习的结构化指令样本，对 Qwen3-4B 做 LoRA 监督微调，让模型能够根据 marker gene 列表和生物学上下文输出结构化的细胞类型注释。

整个流程分为三个主要阶段：

1. **数据准备与知识构建**（01–09）：从公开数据库获取数据，经过清洗、标签净化、marker 提取、SFT 数据构造，以及 ontology 资源和 marker 知识库的构建。
2. **模型训练**：使用 Hugging Face TRL + PEFT 对 Qwen3-4B 进行 LoRA 监督微调。
3. **推理与评估**：基于 KB 检索增强的批量推理，配合结构化字段解析和多层级生物学评估指标。

---

## 整体流程

```text
scripts/data_prep/
  01_list_candidate_datasets.py      ← 筛选候选数据集
  02_export_selected_datasets.py     ← 导出原始 h5ad
  03_clean_and_standardize.py        ← 清洗与标准化
  04_make_marker_examples.py         ← 提取 marker 样本
  05_make_sft_jsonl.py               ← 构建 SFT 数据（v3 canonical labels）
  06_split_and_validate_v2.py        ← dataset-level 切分
  07_build_ontology_resources.py     ← 构建 ontology 索引
  08_build_marker_kb.py              ← 构建 marker 知识库
  09_purify_labels.py                ← 标签净化与 CL 标准化
          ↓
scripts/train/
  run_qwen3_hf_trl.sh                ← Qwen3-4B LoRA SFT（HF TRL）
          ↓
scripts/infer/
  infer_qwen3_kb_retrieval.py        ← KB 检索增强推理 + 评估
```

---

## 目录结构

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
│   │   ├── ontology_index.jsonl
│   │   ├── cell_ontology_min.jsonl
│   │   ├── merged_marker_kb.jsonl
│   │   └── train_marker_kb.jsonl
│   ├── meta/
│   ├── sft/
│   └── splits/
├── resources/
│   └── ontology/
│       ├── label_aliases.tsv
│       └── cell_ontology_min.jsonl
├── my_models/
│   └── Qwen/
│       └── Qwen3-4B/
├── output/
│   └── <训练输出目录>/
└── README.md
```

---

## 1. 环境依赖

建议使用独立 conda 环境：

```bash
conda create -n shuke_SCA python=3.10 -y
conda activate shuke_SCA
```

安装数据处理依赖：

```bash
pip install cellxgene-census anndata scanpy pandas numpy tqdm
```

安装模型训练与推理依赖：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按 CUDA 版本调整
pip install transformers peft trl accelerate
pip install modelscope
pip install bitsandbytes   # 可选，仅 8B QLoRA 时需要
```

---

## 2. 数据源与模型

- **数据源**：CELLxGENE Census，通过 `cellxgene_census.get_anndata()` 获取标准化 AnnData
- **基础模型**：Qwen3-4B，通过 ModelScope 下载，保存到 `my_models/Qwen/Qwen3-4B/`
- **微调方式**：LoRA（all-linear），使用 HF TRL `SFTTrainer`
- **推理**：本地 base model + LoRA adapter，配合 marker 知识库 Jaccard 检索增强

---

## 3. 配置说明

### 数据准备配置

```text
scripts/data_prep/data_prep_config.py
```

统一管理 01–09 的路径、筛选阈值、清洗参数、marker 提取参数、SFT 构造参数等。

### 训练配置

```text
scripts/train/train_config.yaml
```

所有训练超参数均在此文件中配置，无需修改训练脚本即可调整。主要配置项：

```yaml
model_size: "4B"              # "4B"（全精度 LoRA）或 "8B"（QLoRA）
train_file: "data/splits/train_messages_no_think_v3.jsonl"
val_file:   "data/splits/val_messages_no_think_v3.jsonl"
output_dir: ""                # 留空则自动生成带时间戳的目录名

num_train_epochs: 10
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
lora_r: 16
lora_alpha: 32
max_length: 1536
```

---

## 4. 数据准备流程（01–09）

### Step 1 — 候选数据集筛选

```bash
python -u scripts/data_prep/01_list_candidate_datasets.py \
  2>&1 | tee data/meta/01_list_candidate_datasets.log
```

从 Census 按目标组织查询候选 dataset，统计 cell_count、unique_cell_types、tissues、diseases，按配置阈值过滤后输出 `candidate_datasets.csv` 和 `selected_datasets.csv`。

### Step 2 — 导出原始 h5ad

```bash
python -u scripts/data_prep/02_export_selected_datasets.py \
  2>&1 | tee data/meta/02_export_selected_datasets.log
```

读取 `selected_datasets.csv`，导出 `use=1` 的数据集为 `.h5ad`，支持断点续跑和失败重试。

### Step 3 — 清洗与标准化

```bash
python -u scripts/data_prep/03_clean_and_standardize.py \
  2>&1 | tee data/meta/03_clean_and_standardize.log
```

统一基因名与 obs 字段，过滤 unknown/ambiguous/doublet 等标签，加入 ontology 相关字段（`cell_ontology_id`、`cell_type_status`、`cell_type_level`），输出训练导向的 clean h5ad。

### Step 4 — marker 样本提取

```bash
python -u scripts/data_prep/04_make_marker_examples.py \
  2>&1 | tee data/meta/04_make_marker_examples.log
```

基于 `one-vs-rest` 差异表达分析，为每个 cell type 提取 top positive markers，构造 marker-level 样本记录（一条样本 = 一个 cell type 的 marker 证据，不是单个细胞）。

### Step 5 — 构建 SFT 数据

```bash
python -u scripts/data_prep/05_make_sft_jsonl.py \
  2>&1 | tee data/meta/05_make_sft_jsonl.log
```

将 marker 记录包装成 Qwen3 可训练的 `messages` 格式。当前使用 **v3 builder**，assistant 端的 `cell_type` 和 `cell_ontology_id` 直接来自 marker 记录中的 canonical CL 标准标签，不经过 LLM 生成（避免标签噪声）。同时输出标准版和 no-think 兼容版。

### Step 6 — dataset-level 切分

```bash
python -u scripts/data_prep/06_split_and_validate_v2.py \
  2>&1 | tee data/meta/06_split_and_validate_v2.log
```

按 `dataset_id` 级别切分 train/val/test，避免同一 dataset 同时出现在训练和测试中。同时为 v3 数据生成对应的 split 文件。

### Step 7 — 构建 ontology 资源

```bash
python -u scripts/data_prep/07_build_ontology_resources.py \
  2>&1 | tee data/meta/07_build_ontology_resources.log
```

基于 `label_aliases.tsv` 构建本地 ontology 索引，输出 `data/knowledge/ontology_index.jsonl` 和 `cell_ontology_min.jsonl`，供运行时标签标准化和 ID 查找使用。

### Step 8 — 构建 marker 知识库

```bash
python -u scripts/data_prep/08_build_marker_kb.py \
  2>&1 | tee data/meta/08_build_marker_kb.log
```

将外部 marker 知识与训练数据提炼的 marker 合并，构建统一 marker KB（`merged_marker_kb.jsonl` 和 `train_marker_kb.jsonl`），供推理时 Jaccard 相似度检索使用。

### Step 9 — 标签净化

```bash
python -u scripts/data_prep/09_purify_labels.py \
  2>&1 | tee data/meta/09_purify_labels_run_summary.txt
```

对 marker 记录中的 cell_type 标签进行 CL 标准名对齐，修复错误映射，过滤恶性/异常细胞标签，同步扩展 `label_aliases.tsv`。净化后重新运行 04→06 更新训练数据。

**当前数据规模**：~1341 条训练样本，来自 10+ 个数据集，覆盖血液、肺、肝脏、肠道等组织。

---

## 5. 训练

```bash
cd /data/projects/shuke/code/singal_cell_annotation
nohup bash scripts/train/run_qwen3_hf_trl.sh > nohup_train.log 2>&1 &
```

训练脚本读取 `train_config.yaml` 中的所有配置，使用 HF TRL `SFTTrainer` 训练，保存 LoRA adapter 到 `output_dir`（可自动生成带时间戳的目录）。

**当前推荐配置（4B）**：bf16 全精度 LoRA，显存约 14GB，适合单张 24GB GPU。

---

## 6. 推理与评估

```bash
nohup python -u scripts/infer/infer_qwen3_kb_retrieval.py \
  --model_dir my_models/Qwen/Qwen3-4B \
  --adapter_dir output/<训练输出目录> \
  --test_file data/splits/test_messages_no_think_v3.jsonl \
  --kb_file data/knowledge/merged_marker_kb.jsonl \
  --output_dir output/<推理输出目录> \
  > nohup_infer.log 2>&1 &
```

推理脚本在生成前先用 Jaccard 相似度从 marker KB 中检索相关条目，注入到 prompt 中（KB 检索增强）。推理完成后自动解析 JSON 输出，计算多层级评估指标。

### 评估指标

| 指标 | 含义 |
|------|------|
| `cell_type_exact_accuracy` | 严格字符串匹配 |
| `ontology_compatible_accuracy` | 包含 ontology 兼容匹配（父子关系） |
| `cell_type_same_lineage_rate` | 预测与 gold 在同一大谱系 |
| `cell_type_severe_error_rate` | 跨谱系严重错误率 |
| `cell_ontology_id_accuracy` | CL ID 准确率 |
| `parse_ok_rate` | JSON 解析成功率 |

### v3 模型当前结果（n=215）

| 指标 | 数值 |
|------|------|
| exact accuracy | 26.5% |
| ontology_compatible | 39.1% |
| same_lineage | 64.2% |
| severe_error | 35.8% |
| ontology_id accuracy | 43.9%（n=180） |

---

## 7. 诊断工具

`scripts/diagnosis/` 提供了一套系统化诊断脚本，用于定位模型性能瓶颈：

```bash
# 运行全套诊断
python scripts/diagnosis/run_all_diagnosis.py

# 单独运行各诊断模块
python scripts/diagnosis/analyze_label_noise.py      # 训练标签噪声分析
python scripts/diagnosis/analyze_error_buckets.py    # 错误分桶分析
python scripts/diagnosis/ablate_kb_retrieval.py      # KB 消融（with/without KB）
python scripts/diagnosis/ablate_ontology_target.py   # Ontology 目标消融
python scripts/diagnosis/generate_diagnosis_report.py  # 生成汇总报告
```

**主要诊断结论**：
- 训练数据标签噪声率：33.7%
- KB 检索对 exact accuracy 的贡献：+21.7pp
- v3 canonical labels 对 ontology_id accuracy 的提升：+42.8pp（vs v2 的 1.1%）

---

## 8. 方法设计说明

**为什么按 dataset_id 切分**：同一个 dataset 往往生成多条 marker 样本，按记录随机切分会导致测试集泄漏，dataset-level split 能保证评测更真实。

**为什么样本单位是 marker record 而不是单细胞**：任务目标是"根据 marker gene 列表推断 cell type"，使用 cell-type 级别的 marker 汇总记录比单细胞表达向量更贴近实际注释思路，也适合语言模型的输入形式。

**为什么 v3 不使用 LLM 生成标签**：早期版本（v2）用 LLM 生成 assistant 端的 cell_type 和 cell_ontology_id，导致标签噪声累积（量化为 33.7%），且 CL ID 普遍错误（v2 的 ontology_id_accuracy 仅 1.1%）。v3 直接从 marker 记录的 canonical 字段读取，从根本上消除这一问题。

**为什么使用 no-think 训练数据**：当前训练和推理均使用 Qwen3 no-think 模式（user 末尾 `/no_think`，assistant 含空 `<think></think>`），在注释任务中不需要 chain-of-thought 推理，no-think 输出更简洁稳定。

---

## 9. 当前局限性

1. **训练样本规模偏小**：约 1341 条，覆盖细胞类型有限，泛化能力受制于数据多样性。
2. **文字版任务的竞争压力**：基于 marker gene list 的文字推理任务，大型通用 LLM 也能处理，小模型的优势主要在本地部署和隐私场景。
3. **标签噪声仍然存在**：即使经过 09_purify_labels 净化，部分长尾细胞类型的标签仍难以完全对齐 CL 标准。
4. **推理评估仍为半自动**：severe_error 判断基于谱系规则，不能完全替代生物学专家审核。

---

## 10. 致谢

- [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/)
- [Qwen](https://github.com/QwenLM/Qwen3) / ModelScope
- [Hugging Face Transformers](https://github.com/huggingface/transformers) / PEFT / TRL
- 各公开单细胞数据集贡献者
