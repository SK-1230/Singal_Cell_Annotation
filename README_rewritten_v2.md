# Single-Cell Cell Type Annotation with Qwen3

一个面向 **single-cell RNA-seq（scRNA-seq）细胞类型注释** 的数据构建、监督微调与推理验证项目。  
本项目以 **CELLxGENE Census** 为主要数据源，围绕 **候选数据集筛选 → 标准化导出 → 清洗与标准化 → marker 提取 → SFT 数据构造 → dataset-level 划分 → Qwen3 LoRA 微调 → 推理评估** 建立了一条可复现的端到端工程流程。

---

## 项目简介

本项目的目标，是构建一个能够根据 **marker gene 排序列表** 和基本生物学上下文信息，自动输出细胞类型判断结果的原型系统。  
整体思路不是把单细胞表达矩阵直接输入大模型，而是先从标准化单细胞数据中提取更适合语言模型处理的 **marker-level 监督样本**，再使用 **Qwen3 + LoRA** 做轻量化监督微调。

当前版本更偏向于：

- 一个 **单细胞注释任务的大模型工程原型**
- 一个 **从公开数据库构建训练数据的实践流程**
- 一个 **适合后续扩展数据规模、标签体系和训练策略的研究起点**

---

## 项目特点

- 基于 **CELLxGENE Census** 获取标准化单细胞数据
- 支持候选数据集自动筛选与批量导出 `.h5ad`
- 支持标签清洗、标准化与训练导向过滤
- 基于 `one-vs-rest` 差异表达分析构造 marker 样本
- 自动生成 **Qwen3 / ms-swift** 可直接使用的 `messages` 格式 SFT 数据
- 支持 **dataset-level split**，避免同一 dataset 同时出现在训练集和测试集中
- 支持 **pseudo-val** 与 **hard test** 的增强划分策略
- 使用 **Qwen3-4B + LoRA + ms-swift** 进行轻量化微调
- 支持本地模型目录加载，便于服务器离线训练与推理
- 数据预处理多个阶段支持 manifest / summary 输出，便于中断后续跑和结果核查

---

## 整体流程

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
06_split_and_validate_v2.py
    ↓
train_qwen3_swift.sh
    ↓
infer_qwen3_swift_batch.py
```

---

## 项目目录结构

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
│   │   └── data_prep_config.py
│   ├── train/
│   │   └── train_qwen3_swift.sh
│   └── infer/
│       └── infer_qwen3_swift_batch.py
├── data/
│   ├── raw_h5ad/
│   ├── clean_h5ad/
│   ├── intermediate/
│   ├── meta/
│   ├── sft/
│   └── splits/
├── my_models/
│   └── Qwen/
│       └── Qwen3-4B/
├── output/
│   ├── qwen3_4b_sc_sft_swift_v2/
│   └── infer_qwen3_swift_batch/
└── README.md
```

### 目录说明

- `scripts/data_prep/`：01～06 数据准备脚本与统一配置
- `scripts/train/`：训练脚本
- `scripts/infer/`：推理与评估脚本
- `data/raw_h5ad/`：从 Census 导出的原始 `.h5ad`
- `data/clean_h5ad/`：经过清洗与标准化后的 `.h5ad`
- `data/intermediate/`：中间产物，例如 `marker_examples.jsonl`
- `data/meta/`：manifest、summary、日志等辅助文件
- `data/sft/`：SFT 训练样本
- `data/splits/`：train / val / test 划分结果
- `my_models/`：本地缓存的基础模型
- `output/`：训练输出、checkpoint、推理结果等

---

## 1. 环境依赖

建议使用独立 conda 环境：

```bash
conda create -n shuke_SCA python=3.10 -y
conda activate shuke_SCA
```

安装数据处理相关依赖：

```bash
pip install cellxgene-census anndata scanpy pandas numpy tqdm
```

安装模型训练与推理依赖：

```bash
pip install transformers peft trl
pip install ms-swift
pip install modelscope
```

如果使用 GPU，请根据本机 CUDA 环境安装匹配版本的 PyTorch。

---

## 2. 数据源与模型

### 数据源

- **CELLxGENE Census**
- 通过 `cellxgene_census.open_soma(...)` 与 `cellxgene_census.get_anndata(...)` 获取数据
- 导出对象为标准化的 `AnnData`

### 基础模型

- **Qwen3-4B**
- 本地路径示例：`my_models/Qwen/Qwen3-4B`
- 微调方式：**LoRA**
- 训练框架：**ms-swift**

---

## 3. 配置文件说明

项目核心配置集中在：

```text
scripts/data_prep/data_prep_config.py
```

该文件统一管理 01～06 数据处理脚本的参数，包括：

- 路径配置
- Census 版本与物种信息
- 候选数据集筛选阈值
- 自动 selected 策略
- 导出与重试参数
- 清洗阈值
- marker 提取参数
- SFT 数据构造参数
- split 策略与增强选项

### 关键配置示例

```python
CENSUS_VERSION = "2025-11-08"
ORGANISM = "Homo sapiens"
TARGET_TISSUES = ["blood", "lung", "liver", "intestine"]

MIN_DATASET_CELLS = 3000
MAX_DATASET_CELLS = 120000
MIN_UNIQUE_CELL_TYPES = 6

MIN_CELLS_PER_LABEL = 50
MIN_GENES_DETECTED_IN_CELLS = 10

SYSTEM_PROMPT = "You are a transcriptomics assistant ..."
```

---

## 4. 数据准备流程

### Step 1. 候选数据集筛选

**脚本：**

```text
scripts/data_prep/01_list_candidate_datasets.py
```

**作用：**

- 查询指定 `tissue_general` 下的候选 dataset
- 统计每个 dataset 的：
  - `cell_count`
  - `unique_cell_types`
  - `tissues`
  - `diseases`
- 根据配置自动生成候选池和 selected 集

**主要输出：**

- `data/meta/candidate_datasets.csv`
- `data/meta/selected_datasets.csv`

**当前模式说明：**

在当前配置中：

```python
SELECT_MODE = "auto_balanced"
```

表示脚本会自动生成可直接供 02 使用的 selected 集合，而不是只生成人工模板。

**运行方式：**

```bash
cd /data/projects/shuke/code/singal_cell_annotation
python -u scripts/data_prep/01_list_candidate_datasets.py 2>&1 | tee data/meta/01_list_candidate_datasets.log
```

---

### Step 2. 导出标准化 h5ad

**脚本：**

```text
scripts/data_prep/02_export_selected_datasets.py
```

**作用：**

- 读取 `selected_datasets.csv`
- 导出 `use=1` 的数据集到标准化 `.h5ad`

**特点：**

- 支持跳过已存在输出
- 支持失败重试
- 每处理完一个 dataset 都会写 manifest
- 可附带保存 `source_meta.json`

**主要输出：**

- `data/raw_h5ad/*.h5ad`
- `data/meta/raw_export_manifest.csv`

**运行方式：**

```bash
python -u scripts/data_prep/02_export_selected_datasets.py 2>&1 | tee data/meta/02_export_selected_datasets.log
```

---

### Step 3. 清洗与标准化

**脚本：**

```text
scripts/data_prep/03_clean_and_standardize.py
```

**作用：**

- 读取原始 `.h5ad`
- 标准化标签与元数据字段
- 清洗无效或模糊标签
- 去除低频基因
- 去除样本数过少的标签
- 生成训练导向的 clean 数据

**当前清洗逻辑包括：**

- 过滤 `unknown / ambiguous / doublet / multiplet / debris` 等标签
- 统一 `cell_type_clean`
- 保留 `cell_type_gold`
- 规范 `tissue_general / tissue / disease`
- 过滤过低表达基因
- 过滤过少样本标签
- 保留 `counts` 层以便后续 marker 提取

**主要输出：**

- `data/clean_h5ad/*.h5ad`
- `data/meta/clean_manifest.csv`

**说明：**

这一步并不是完整生物学质控流程，而是更偏向于 **适合监督训练的数据清洗**。

**运行方式：**

```bash
python -u scripts/data_prep/03_clean_and_standardize.py 2>&1 | tee data/meta/03_clean_and_standardize.log
```

---

### Step 4. Marker 样本提取

**脚本：**

```text
scripts/data_prep/04_make_marker_examples.py
```

**作用：**

- 基于 `cell_type_clean` 分组
- 对每个标签做 `one-vs-rest` 差异分析
- 提取每个 cell type 的 top marker genes
- 构造成 marker-level 样本

**当前实现特点：**

- 使用 `counts` 层做差异分析
- 每类细胞最多下采样到一定数量，避免超大类压制计算
- 过滤无信息或干扰性基因，如：
  - 线粒体基因
  - 核糖体基因
  - 部分高丰度低信息基因
- 只保留正向 marker 候选
- 输出每个 cell type 的 marker 及统计信息

**主要输出：**

- `data/intermediate/marker_examples.jsonl`
- `data/intermediate/marker_examples_manifest.csv`
- `data/intermediate/04_make_marker_examples_run_summary.txt`

**说明：**

这里的一条样本不是一个细胞，而是一个 **cell type marker record**。

**运行方式：**

```bash
python -u scripts/data_prep/04_make_marker_examples.py 2>&1 | tee data/meta/04_make_marker_examples.log
```

---

### Step 5. 构造 SFT 数据

**脚本：**

```text
scripts/data_prep/05_make_sft_jsonl.py
```

**作用：**

- 将 marker records 转换为大模型监督微调样本
- 输出标准 `messages` 格式与 `no-think` 兼容格式

**输入内容包括：**

- organism
- tissue
- disease/context
- ranked marker genes

**模型输出目标包括：**

- `cell_type`
- `supporting_markers`
- `confidence`
- `need_manual_review`
- `rationale`

**主要输出：**

- `data/sft/sft_records_full.jsonl`
- `data/sft/sft_messages.jsonl`
- `data/sft/sft_messages_no_think.jsonl`
- `data/sft/sft_records_manifest.csv`

**说明：**

当前项目同时构造：

1. **标准 messages**
2. **Qwen3 no-think 兼容 messages**

其中 `no-think` 版本会：

- 在 user prompt 结尾加 `/no_think`
- 在 assistant 中加入空的 `<think></think>`

**运行方式：**

```bash
python -u scripts/data_prep/05_make_sft_jsonl.py 2>&1 | tee data/meta/05_make_sft_jsonl.log
```

---

### Step 6. 数据划分与校验

**脚本：**

```text
scripts/data_prep/06_split_and_validate_v2.py
```

**作用：**

- 读取 `sft_records_full.jsonl`
- 基于 `dataset_id` 做 **dataset-level split**
- 输出 train / val / test
- 在 dataset 数较少时可自动构造 pseudo-val
- 可额外导出 hard test 子集
- 可选进行 token 长度检查

**当前 v2 的增强点包括：**

- dataset-level 切分，避免数据泄漏
- 支持 `dataset profile` 统计
- 支持按主组织分层切分
- 支持小数据集场景的特殊逻辑
- 支持 `pseudo-val`
- 支持 `test_hard` 导出

**主要输出：**

- `data/splits/train_full.jsonl`
- `data/splits/val_full.jsonl`
- `data/splits/test_full.jsonl`
- `data/splits/train_messages.jsonl`
- `data/splits/val_messages.jsonl`
- `data/splits/test_messages.jsonl`
- `data/splits/train_messages_no_think.jsonl`
- `data/splits/val_messages_no_think.jsonl`
- `data/splits/test_messages_no_think.jsonl`
- `data/splits/test_hard_*.jsonl`
- `data/splits/dataset_profiles.csv`

**运行方式：**

```bash
python -u scripts/data_prep/06_split_and_validate_v2.py 2>&1 | tee data/meta/06_split_and_validate_v2.log
```

如需检查 token 长度：

```bash
python -u scripts/data_prep/06_split_and_validate_v2.py \
  --check-tokens \
  --model-name Qwen/Qwen3-8B \
  2>&1 | tee data/meta/06_split_and_validate_v2_token_check.log
```

---

## 5. 训练

**脚本：**

```text
scripts/train/train_qwen3_swift.sh
```

当前训练使用：

- 基础模型：`Qwen3-4B`
- 微调方式：`LoRA`
- 训练框架：`ms-swift`
- 数据文件：`train_messages_no_think.jsonl`
- 验证文件：`val_messages_no_think.jsonl`

### 当前训练配置特点

- `train_type=lora`
- `torch_dtype=bfloat16`
- `num_train_epochs=8`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `learning_rate=5e-5`
- `lora_rank=8`
- `lora_alpha=32`
- `target_modules=all-linear`
- `max_length=1024`
- `loss_scale=ignore_empty_think`

### 运行方式

```bash
cd /data/projects/shuke/code/singal_cell_annotation
bash scripts/train/train_qwen3_swift.sh 2>&1 | tee data/meta/train_qwen3_swift_v2.log
```

### 说明

训练脚本中对以下内容做了基本检查：

- 基础模型目录是否存在
- train 文件是否存在且非空
- val 文件是否存在且非空

训练输出默认保存在：

```text
output/qwen3_4b_sc_sft_swift_v2/
```

---

## 6. 推理与评估

**脚本：**

```text
scripts/infer/infer_qwen3_swift_batch.py
```

**作用：**

- 加载基础模型与 LoRA adapter
- 读取测试集 `test_messages_no_think.jsonl`
- 逐条生成预测结果
- 解析模型输出 JSON
- 与 gold assistant 内容进行对比
- 输出预测文件与摘要文件

### 当前评估内容

脚本会对以下字段进行比较：

- `cell_type`
- `confidence`
- `need_manual_review`

并统计：

- 解析成功率
- `cell_type_match_accuracy`
- `confidence_match_accuracy`
- `need_manual_review_match_accuracy`

### 主要输出

- `output/infer_qwen3_swift_batch/predictions.jsonl`
- `output/infer_qwen3_swift_batch/predictions.csv`
- `output/infer_qwen3_swift_batch/summary.json`

### 运行方式

```bash
cd /data/projects/shuke/code/singal_cell_annotation
python -u scripts/infer/infer_qwen3_swift_batch.py 2>&1 | tee data/meta/infer_qwen3_swift_batch.log
```

### 注意事项

推理脚本默认需要你手动指定：

- `BASE_MODEL_PATH`
- `ADAPTER_PATH`
- `TEST_FILE`
- `OUTPUT_DIR`

其中 `ADAPTER_PATH` 应指向具体 checkpoint 目录，而不是训练输出根目录。

---

## 7. 快速开始

如果你要从头跑完整流程，推荐顺序如下：

```bash
cd /data/projects/shuke/code/singal_cell_annotation

python -u scripts/data_prep/01_list_candidate_datasets.py 2>&1 | tee data/meta/01_list_candidate_datasets.log
python -u scripts/data_prep/02_export_selected_datasets.py 2>&1 | tee data/meta/02_export_selected_datasets.log
python -u scripts/data_prep/03_clean_and_standardize.py 2>&1 | tee data/meta/03_clean_and_standardize.log
python -u scripts/data_prep/04_make_marker_examples.py 2>&1 | tee data/meta/04_make_marker_examples.log
python -u scripts/data_prep/05_make_sft_jsonl.py 2>&1 | tee data/meta/05_make_sft_jsonl.log
python -u scripts/data_prep/06_split_and_validate_v2.py 2>&1 | tee data/meta/06_split_and_validate_v2.log

bash scripts/train/train_qwen3_swift.sh 2>&1 | tee data/meta/train_qwen3_swift_v2.log
python -u scripts/infer/infer_qwen3_swift_batch.py 2>&1 | tee data/meta/infer_qwen3_swift_batch.log
```

---

## 8. 方法设计说明

### 8.1 为什么优先使用 Census 标准化数据

不同来源的原始 `.h5ad` 在字段命名、元数据组织、数据格式上可能差异很大。  
优先使用 Census 标准化接口，可以显著降低后续工程处理复杂度，提高整个流程的可复现性。

### 8.2 为什么还要再次清洗

即使 Census 已经做过标准化，直接用于监督训练仍然会遇到：

- 缺失标签
- 模糊标签
- 低质量标签
- 标签文本不统一
- 长尾极小标签

因此仍需要进行一次 **训练导向清洗**。

### 8.3 为什么样本单位是 marker record，而不是单细胞

当前任务目标是：

> 给定一个 cluster 的 marker gene 排序列表和上下文，让模型预测最可能的 cell type。

因此，使用 `cell type` 级别的 marker 记录来构造监督样本，更符合当前大模型输入输出形式，也更利于可解释生成。

### 8.4 为什么采用 dataset-level split

如果直接按 record 随机切分，同一个 dataset 的多个样本可能会同时进入 train 和 test。  
这会导致模型记住数据集特征，造成评估泄漏。  
因此项目使用 `dataset_id` 级别切分。

### 8.5 为什么保留 no-think 版本

当前 Qwen3 训练流程中，使用：

- user 末尾加 `/no_think`
- assistant 保留空 `<think></think>`
- 训练时设置 `ignore_empty_think`

这种方式更方便兼容当前训练和推理流程。

---

## 9. 当前项目状态

当前工程链路已经覆盖并打通：

- 候选数据集筛选
- Census 标准化导出
- clean `.h5ad` 构造
- marker 提取
- SFT 数据构造
- dataset-level 切分
- Qwen3-4B + LoRA + ms-swift 训练
- 基础推理与结构化结果比对

这意味着项目已经具备了一个完整的 **单细胞注释任务大模型微调原型**。

---

## 10. 当前局限性

当前版本仍存在一些限制：

1. **训练样本规模仍偏小**  
   尽管已经完成流程搭建，但样本数量仍然有限。

2. **标签统一仍较粗糙**  
   当前主要依赖文本标准化与基础规则映射，尚未做系统 ontology 对齐。

3. **样本形式仍较单一**  
   当前主要使用 marker gene record，尚未加入更多结构化上下文。

4. **评估仍偏工程验证**  
   现阶段更适合验证流程与初步效果，不足以直接作为最终科研结论。

5. **推理评估较基础**  
   当前更多比较结构化字段一致性，尚未引入更完整的 benchmark 评估体系。

---

## 11. 后续优化方向

### 数据层

- 扩大 dataset 覆盖范围
- 增加更多单组织、标签清晰的数据集
- 引入更多正常 / 疾病场景

### 标签层

- 构建统一 label dictionary
- 与 ontology 层级做更系统的映射
- 处理同义标签、父子层级标签和粒度差异

### 样本层

- 引入更多提示模板
- 增加 harder examples
- 构造 manual review 样本
- 加入 differential expression 统计等附加信息

### 训练层

- 比较不同 LoRA 超参数
- 尝试更大模型
- 设计更稳健的验证集与测试集

### 评估层

- 构建标准 benchmark
- 对比 zero-shot / few-shot / SFT
- 评估 JSON 正确率、cell type 准确率与 supporting markers 合理性

---

## 12. 常见问题

### Q1. `04_make_marker_examples.py` 输出的 `Total marker examples` 是不是细胞数？

不是。  
它表示可用于训练的 marker-level 样本数，也就是 `cell type marker records` 的数量。

### Q2. 为什么 `05_make_sft_jsonl.py` 同时输出标准版和 no-think 版？

为了兼容 Qwen3 当前的训练格式。  
项目既保留标准 `messages` 格式，也保留适合 no-think 训练的版本。

### Q3. 为什么 `06_split_and_validate_v2.py` 要坚持 dataset-level split？

因为同一个 dataset 往往会生成多个 marker 样本。  
如果随机切分 record，很容易导致同一个 dataset 同时出现在训练集和测试集中，引入泄漏。

### Q4. 为什么推理时要加载 checkpoint 目录，而不是训练输出根目录？

因为 `PeftModel.from_pretrained(...)` 需要读取 adapter 权重文件，例如：

- `adapter_config.json`
- `adapter_model.safetensors`

这些通常位于具体 checkpoint 目录中。

### Q5. 为什么当前 val 可能为空，或者是 pseudo-val？

因为 dataset 数较少时，严格划分独立 val 会进一步压缩 train 数据。  
因此 v2 版本允许在必要时从 train records 中抽取少量 pseudo-val，用于训练监控。

---

## 13. 致谢

本项目依赖以下数据与工具生态：

- CELLxGENE Census
- 单细胞公开数据集贡献者
- Qwen
- Hugging Face Transformers
- PEFT
- ms-swift
- ModelScope

---

## 14. License

当前项目主要用于研究与实验验证。  
若后续公开发布，请结合以下内容进一步补充具体 license 信息：

- Census 数据使用协议
- 原始单细胞数据集使用条款
- Qwen 模型许可证
- 项目自身代码许可证
