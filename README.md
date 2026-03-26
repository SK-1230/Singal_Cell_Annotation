# Single-Cell Cell Type Annotation with Qwen3

一个面向 **single-cell RNA-seq（scRNA-seq）细胞类型注释** 的数据构建与大模型微调项目。  
本项目以 **CELLxGENE Census** 为主要数据源，从候选数据集筛选、标准化导出、清洗、marker 提取、SFT 样本构造，到 **Qwen3 + LoRA** 微调与推理验证，形成了一条可复现的端到端流程。

---

## 项目概览

本项目旨在构建一个基于大语言模型的细胞类型注释原型系统。核心思想是：

1. 从公开单细胞数据库中筛选合适的数据集。
2. 导出标准化的 `AnnData` 数据。
3. 清洗细胞标签与表达矩阵。
4. 基于人工标签提取 marker gene 样本。
5. 将 marker 样本转换为 **Qwen3 / ms-swift 可用的 SFT 数据格式**。
6. 使用 **Qwen3 + LoRA** 进行监督微调。
7. 在测试样本上验证模型是否能够根据 marker gene 列表和生物学上下文输出合理的细胞类型判断。

当前项目更偏向于一个 **“单细胞注释 + 大模型微调”的工程验证与研究原型**，适合后续逐步扩展数据规模、标签体系和训练策略。

---

## 项目特点

- 基于 **CELLxGENE Census** 获取标准化单细胞数据。
- 支持按候选数据集筛选并批量导出 `.h5ad`。
- 支持单细胞标签清洗与标准化。
- 基于 `one-vs-rest` 差异表达分析提取 marker 样本。
- 自动构造 **Qwen3 / ms-swift** 所需的 `messages` 格式 SFT 数据。
- 支持 **dataset-level split**，避免同一数据集泄漏到训练和测试中。
- 使用 **Qwen3-4B + LoRA + ms-swift** 完成轻量化微调。
- 支持本地模型目录加载，避免训练时依赖外网。

---

## 当前流程图

```text
01_list_candidate_datasets.py
    ↓
02_export_selected_datasets.py
    ↓
03_clean_and_standardize.py
    ↓
04_make_marker_examples.py
    ↓
05_make_sft_jsonl.py
    ↓
06_split_and_validate.py
    ↓
download_qwen3_4b_modelscope.py
    ↓
train_qwen3_swift.sh
    ↓
infer_qwen3_swift.py
```

---

## 目录结构

```text
singal_cell_annotation/
├── config.py
├── 01_list_candidate_datasets.py
├── 02_export_selected_datasets.py
├── 03_clean_and_standardize.py
├── 04_make_marker_examples.py
├── 05_make_sft_jsonl.py
├── 06_split_and_validate.py
├── download_qwen3_4b_modelscope.py
├── train_qwen3_swift.sh
├── infer_qwen3_swift.py
├── my_models/
│   └── Qwen3-4B/
├── output/
│   └── qwen3_4b_sc_sft_swift_v2/
├── data/
│   ├── raw_h5ad/
│   ├── clean_h5ad/
│   ├── intermediate/
│   ├── meta/
│   ├── sft/
│   └── splits/
└── README.md
```

### 目录说明

- `raw_h5ad/`：从 Census 导出的原始标准化 `.h5ad`
- `clean_h5ad/`：清洗后的 `.h5ad`
- `intermediate/`：中间结果，如 `marker_examples.jsonl`
- `meta/`：各步骤 manifest、summary、log 等元数据
- `sft/`：构造好的 SFT 样本
- `splits/`：划分后的 `train / val / test` 数据
- `my_models/`：本地缓存的基础模型目录
- `output/`：训练输出目录、checkpoint、日志、曲线图等

---

## 1. 环境依赖

建议使用 conda 创建独立环境，例如：

```bash
conda create -n shuke_SCA python=3.10 -y
conda activate shuke_SCA
```

安装核心依赖：

```bash
pip install cellxgene-census anndata scanpy pandas numpy tqdm
pip install transformers peft trl
pip install modelscope
pip install ms-swift
```

如果使用 GPU 训练，还需要根据 CUDA 环境安装对应版本的 PyTorch。

---

## 2. 数据源与模型

### 数据源

- **CELLxGENE Census**
- 导出方式：`cellxgene_census.get_anndata(...)`
- 导出对象：标准化 `AnnData`

### 基础模型

- **Qwen3-4B**
- 下载方式：**ModelScope**
- 微调方式：**LoRA**
- 训练框架：**ms-swift**

---

## 3. 整体流程

### Step 1. 候选数据集筛选

**脚本：** `01_list_candidate_datasets.py`

**作用：**

- 查询 Census 中指定组织的候选 dataset
- 统计每个 dataset 的：
  - `cell_count`
  - `unique_cell_types`
  - `tissues`
  - `diseases`

**输出：**

- `candidate_datasets.csv`
- `selected_datasets.csv`

**说明：**

`selected_datasets.csv` 需要人工筛选，将想保留的数据集设置为 `use=1`。

建议优先保留：

- `cell_count` 适中
- `unique_cell_types` 较丰富
- `tissues` 相对单一
- `disease` 不过于混杂
- 标题看起来不是过于复杂的大型整合分析

**运行：**

```bash
python -u 01_list_candidate_datasets.py 2>&1 | tee data/meta/01_list_candidate_datasets.log
```

### Step 2. 导出标准化 h5ad

**脚本：** `02_export_selected_datasets.py`

**作用：**

- 读取 `selected_datasets.csv`
- 导出 `use=1` 的数据集为标准化 `.h5ad`

**特点：**

- 支持跳过已存在文件
- 支持失败重试
- 自动生成导出 manifest

**输出：**

- `data/raw_h5ad/*.h5ad`
- `raw_export_manifest.csv`

**运行：**

```bash
python -u 02_export_selected_datasets.py 2>&1 | tee data/meta/02_export_selected_datasets.log
```

### Step 3. 清洗与标准化

**脚本：** `03_clean_and_standardize.py`

**作用：**

- 读取 raw `.h5ad`
- 清洗标签与元数据
- 去除无效 `cell type`
- 去除 `ambiguous / unknown / doublet` 等标签
- 过滤低频基因
- 过滤样本数过少的标签
- 输出 cleaned `.h5ad`

**输出：**

- `data/clean_h5ad/*.h5ad`
- `clean_manifest.csv`

**说明：**

这里的目标不是做“最严格的生物质控”，而是先构建适合训练的大模型监督样本。当前实现更偏向训练导向的标签清洗。

**运行：**

```bash
python -u 03_clean_and_standardize.py 2>&1 | tee data/meta/03_clean_and_standardize.log
```

### Step 4. Marker 样本构造

**脚本：** `04_make_marker_examples.py`

**作用：**

- 基于 `cell_type_clean` 进行 `one-vs-rest` 差异分析
- 为每个 `cell type` 提取 `top-k marker genes`
- 构造 marker example

**输出：**

- `data/intermediate/marker_examples.jsonl`
- `marker_examples_manifest.csv`
- `04_make_marker_examples_run_summary.txt`

**说明：**

这里的样本单位不是“单个细胞”，而是“一个 `cell type` 的 marker 记录”。

`Total marker examples` 的含义是：可用于训练的 marker 样本数，而不是细胞数。

**运行：**

```bash
python -u 04_make_marker_examples.py 2>&1 | tee data/meta/04_make_marker_examples.log
```

### Step 5. 构造 SFT 数据

**脚本：** `05_make_sft_jsonl.py`

**作用：**

- 将 marker 样本转换为 Qwen3 / ms-swift 可用的 `messages` 格式
- 同时输出普通版与 `no-think` 兼容版

**输出：**

- `sft_records_full.jsonl`
- `sft_messages.jsonl`
- `sft_messages_no_think.jsonl`

**说明：**

- `system` 来自 `config.py` 中的 `SYSTEM_PROMPT`
- `user` 由 `build_user_prompt(...)` 构造
- `assistant` 由 `build_assistant_answer(...)` 构造

**运行：**

```bash
python -u 05_make_sft_jsonl.py 2>&1 | tee data/meta/05_make_sft_jsonl.log
```

### Step 6. 数据划分

**脚本：** `06_split_and_validate.py`

**作用：**

- 按 `dataset_id` 级别划分 `train / val / test`
- 保证不同 split 之间无 dataset 泄漏
- 输出 split summary

**输出：**

- `train_messages*.jsonl`
- `val_messages*.jsonl`
- `test_messages*.jsonl`

**说明：**

当前实现采用 `dataset-level global split`。  
不再按记录级 tissue 分层，避免多组织 dataset 导致同一 `dataset_id` 重复分配。  
这一步的重点是控制 dataset 泄漏，而不是追求复杂分层抽样。

**运行：**

```bash
python -u 06_split_and_validate.py 2>&1 | tee data/meta/06_split_and_validate.log
```

### Step 7. 下载基础模型

**脚本：** `download_qwen3_4b_modelscope.py`

**作用：**

- 从 ModelScope 下载 `Qwen/Qwen3-4B`
- 保存到本地目录 `my_models/`

**运行：**

```bash
python download_qwen3_4b_modelscope.py
```

### Step 8. LoRA 微调

**脚本：** `train_qwen3_swift.sh`

**作用：**

使用 `ms-swift` 对 `Qwen3-4B` 做 LoRA SFT。

**当前版本特点：**

- 使用 `train_messages_no_think.jsonl`
- 支持 `val_messages_no_think.jsonl`
- 使用 `ignore_empty_think`
- 使用 LoRA 训练
- 保存 checkpoint 与训练日志

**运行：**

```bash
bash train_qwen3_swift.sh 2>&1 | tee data/meta/train_qwen3_swift.log
```

### Step 9. 推理验证

**脚本：** `infer_qwen3_swift.py`

**作用：**

- 加载基础模型 + LoRA adapter
- 读取测试样本
- 进行推理生成
- 检查模型是否学会任务格式与细胞类型判断

**运行：**

```bash
python infer_qwen3_swift.py 2>&1 | tee data/meta/infer_qwen3_swift.log
```

---

## 4. 快速开始

如果你已经完成环境安装，并准备从头跑完整流程，推荐顺序如下：

```bash
python -u 01_list_candidate_datasets.py 2>&1 | tee data/meta/01_list_candidate_datasets.log
# 手动编辑 selected_datasets.csv，设置 use=1

python -u 02_export_selected_datasets.py 2>&1 | tee data/meta/02_export_selected_datasets.log
python -u 03_clean_and_standardize.py 2>&1 | tee data/meta/03_clean_and_standardize.log
python -u 04_make_marker_examples.py 2>&1 | tee data/meta/04_make_marker_examples.log
python -u 05_make_sft_jsonl.py 2>&1 | tee data/meta/05_make_sft_jsonl.log
python -u 06_split_and_validate.py 2>&1 | tee data/meta/06_split_and_validate.log

python download_qwen3_4b_modelscope.py
bash train_qwen3_swift.sh 2>&1 | tee data/meta/train_qwen3_swift.log
python infer_qwen3_swift.py 2>&1 | tee data/meta/infer_qwen3_swift.log
```

---

## 5. 配置说明

项目主要配置集中在 `config.py` 中，常见配置包括：

- Census 版本
- 目标组织列表
- 数据目录路径
- 候选数据集过滤阈值
- 标签清洗关键词
- marker 数量与过滤条件
- SFT 输出路径
- split 输出目录
- 系统提示词 `SYSTEM_PROMPT`

### 典型配置项

```python
CENSUS_VERSION = "2025-11-08"
ORGANISM = "Homo sapiens"

MIN_DATASET_CELLS = ...
MAX_DATASET_CELLS = ...
MIN_UNIQUE_CELL_TYPES = ...
MIN_CELLS_PER_LABEL = ...
MIN_GENES_DETECTED_IN_CELLS = ...

SYSTEM_PROMPT = "You are a transcriptomics assistant ..."
```

---

## 6. 方法设计说明

### 6.1 为什么优先使用 Census 标准化数据

不同投稿者的原始 `.h5ad` 往往存在以下问题：

- 字段命名不一致
- 数据结构差异较大
- 工程处理成本更高

因此，本项目优先使用 Census 标准化切片，以降低后续清洗与训练管线的复杂度。

### 6.2 为什么需要再次清洗标签

虽然 Census 已经做了标准化，但直接用于训练仍可能存在：

- 缺失标签
- `ambiguous` 标签
- `doublet / unknown` 类标签
- 标签大小写与文本不统一
- 长尾极小标签

因此需要进行训练导向的再次清洗。

### 6.3 为什么样本单位是 marker record 而不是单细胞

本项目当前的训练目标是：

> 给定 marker gene 列表和上下文，让模型输出最可能的 cell type。

因此更适合使用“一个 `cell type` 的 marker gene 记录”作为一条监督样本，而不是直接把单细胞表达矩阵原样送入大模型。

### 6.4 为什么按 dataset_id 划分

如果随机按样本切分，会导致：

- 同一个 dataset 的样本同时出现在 `train` 和 `test` 中
- 模型可能记住 dataset 特征，造成评估泄漏

因此当前采用 `dataset-level split`。

### 6.5 为什么使用 no-think 版本数据

当前项目使用 Qwen3，采用了兼容 `no-think` 的 SFT 样本格式：

- `user` 末尾添加 `/no_think`
- `assistant` 保留空 `<think></think>`
- 训练时配合 `ignore_empty_think`

这样更适合当前 Qwen3 的训练与推理方式。

---

## 7. 当前项目状态

目前项目已经完成以下验证：

- 候选数据集筛选流程已跑通
- Census 标准化导出流程已跑通
- cleaned `.h5ad` 构造已跑通
- marker 样本提取已跑通
- Qwen3 / ms-swift SFT 数据构造已跑通
- dataset-level split 已跑通
- Qwen3-4B + LoRA + ms-swift 训练已成功启动并完成验证
- 推理脚本已成功加载基础模型与 LoRA adapter，并输出结构化 JSON

这说明整个工程链路已经打通。

---

## 8. 当前局限性

当前版本仍存在以下局限：

1. **数据集数量仍偏少**  
   虽然比初始版本有所扩充，但训练样本规模仍较小。

2. **dataset 之间异质性较强**  
   部分数据集跨多个组织或疾病背景，增加了样本复杂度。

3. **标签统一仍较粗糙**  
   当前只做了基础 label normalization，尚未构建系统性的 ontology-level label mapping。

4. **训练样本形式较单一**  
   当前只使用 marker record，尚未引入：
   - `cluster summary`
   - differential expression statistics
   - 多轮问答
   - hard negative 样本

5. **评估仍偏工程验证**  
   当前更适合验证链路与初步效果，尚不足以形成最终实验结论。

---

## 9. 后续优化方向

### 数据层

- 增加更多单组织、标签清晰的数据集
- 引入更多正常组织与疾病组织
- 提升数据集多样性与代表性

### 标签层

- 构建统一 label dictionary
- 做 ontology-level 对齐
- 规范同义标签、层级标签和细分类标签

### 样本层

- 增加更多 marker 样本
- 构造 harder examples
- 引入 manual review 标签
- 构造多种 prompt 风格

### 训练层

- 增加 `train / val / test` dataset 数量
- 引入更稳健的验证集
- 比较不同 `epoch / lr / LoRA rank`
- 尝试更大模型（如 `Qwen3-8B`）

### 评估层

- 构建标准 benchmark
- 对比 `zero-shot / few-shot / SFT`
- 评估 JSON 结构正确率、cell type 准确率、marker 解释合理性

---

## 10. 常见问题

### Q1. `04_make_marker_examples` 输出的 `Total marker examples` 是不是细胞数？

不是。  
它表示 marker 样本数 / cell-type 记录数，不是单细胞数。

### Q2. 为什么 `06_split_and_validate` 不能按记录级 tissue 分层？

因为一个 dataset 可能跨多个 tissue。  
如果按记录级 tissue 分层，同一个 `dataset_id` 可能被重复分到多个 split，导致 overlap。

### Q3. 为什么训练时优先使用 `train_messages_no_think.jsonl`？

因为当前使用的是 Qwen3，项目采用了兼容 `no-think` 的 SFT 样本格式：

- `user` 末尾添加 `/no_think`
- `assistant` 保留空 `<think></think>`
- 训练时配合 `ignore_empty_think`

### Q4. 为什么 `swift --help` 报错，但 `swift sft --help` 正常？

因为当前安装版本的 `swift` 顶层 CLI 对 `--help` 处理不完整，但 `swift sft` 子命令正常，不影响训练。

### Q5. 为什么推理时要加载 checkpoint 目录而不是训练输出根目录？

因为 `PeftModel.from_pretrained(...)` 需要找到：

- `adapter_config.json`
- `adapter_model.safetensors`

这些通常位于具体的 checkpoint 目录中，而不是训练输出根目录。

---

## 11. 结果与观察

当前阶段的主要结论包括：

- 使用少量数据时，模型可以较快学会任务输出格式
- 增加 dataset 数量后，marker 样本数明显增加，训练效果优于极小样本版本
- 当前模型已经能够在部分测试样本上输出较合理的细胞类型与 supporting markers
- 但在更复杂样本上，仍然可能受到数据规模和标签体系限制

因此，当前结果更适合作为**工程原型验证**，后续仍需扩充数据并优化标签与评估体系。

---

## 12. 致谢

本项目的数据获取依赖于：

- CELLxGENE Census
- 相关公开单细胞数据集提供者

模型与训练框架依赖于：

- Qwen
- ModelScope
- Hugging Face Transformers
- PEFT
- ms-swift

---

## 13. License

当前项目主要用于研究与实验验证。  
若涉及公开发布，请根据以下内容进一步补充具体 license 信息：

- Census 数据使用协议
- 各原始数据集使用条款
- Qwen 模型许可证
