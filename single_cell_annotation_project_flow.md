
# 单细胞注释项目整体流程说明

## 1. 项目定位

本项目是一个围绕 **单细胞 RNA-seq 细胞类型注释** 构建的完整工程流程。它的核心目标并不只是“训练一个模型”，而是搭建一条从 **公开数据获取、清洗标准化、marker 证据提取、SFT 数据构建、数据集切分、知识资源构建，到大模型微调、批量推理和自动评测** 的全链路管线。

从工程视角看，这个项目主要解决了几类问题：

1. 如何从公开单细胞数据库中自动筛选出适合做细胞类型注释的数据集。
2. 如何把原始 h5ad 数据整理成适合大模型学习的结构化监督样本。
3. 如何把单细胞分析中的 marker gene 证据转成模型可学习的指令数据。
4. 如何在本地 Qwen3-4B 上开展 LoRA 微调，得到领域化注释模型。
5. 如何对模型输出进行结构化解析，并从生物学层面做比普通 accuracy 更细的评估。
6. 如何构建 ontology 资源和 marker 知识库，为后续知识增强、检索增强或推理辅助打基础。

因此，这个项目既是一个 **训练数据生产系统**，也是一个 **单细胞注释大模型训练与评测系统**。

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
│   │   └── 08_build_marker_kb.py
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

各目录的作用可以概括为：

- `scripts/data_prep/`：数据准备主流程脚本，负责把公开单细胞数据逐步处理成训练样本。
- `scripts/train/`：模型训练脚本，负责启动 Qwen3 的 SFT。
- `scripts/infer/`：推理与评测脚本，负责加载训练后的模型批量预测并统计结果。
- `data/raw_h5ad/`：从 CELLxGENE Census 导出的原始 h5ad 文件。
- `data/clean_h5ad/`：清洗、标准化之后的 h5ad 文件及其统计产物。
- `data/intermediate/`：中间结果，例如 marker examples。
- `data/meta/`：日志、manifest、run summary 等元信息。
- `data/sft/`：SFT 训练样本。
- `data/splits/`：train / val / test 切分结果。
- `my_models/Qwen/Qwen3-4B/`：本地基础模型。
- `output/`：训练输出、LoRA adapter、推理结果等。

---

## 3. 整体流程总览

整个项目可以拆成三大阶段：

### 第一阶段：数据准备与知识构建
对应脚本：`01` 到 `08`

主要任务：

- 从 CELLxGENE Census 中筛选候选数据集。
- 导出选中的原始 h5ad 数据。
- 对数据做清洗、标签标准化和 ontology 映射。
- 基于差异表达分析提取 marker gene。
- 将 marker 样本转成 Qwen3 可训练的 SFT 对话数据。
- 按 dataset 级别切分 train / val / test。
- 构建本体索引和 marker 知识库。

### 第二阶段：模型训练
对应脚本：训练代码

主要任务：

- 读取切分好的训练集和验证集。
- 将对话样本整理成 prompt-completion 监督形式。
- 在本地 Qwen3-4B 上挂 LoRA adapter 做 SFT。
- 保存训练后的 adapter、tokenizer 和训练状态。

### 第三阶段：批量推理与评估
对应脚本：推理代码

主要任务：

- 加载基础模型和训练好的 adapter。
- 对测试集逐条生成结构化输出。
- 自动抽取和解析模型输出中的 JSON。
- 与 gold 标注做字段级和谱系级比较。
- 输出逐样本结果、CSV 明细和 summary 指标。

一句话概括整个项目：

> **从公开单细胞数据出发，构建面向细胞类型注释的大模型训练数据，微调 Qwen3，并在测试集上做结构化推理与生物学层级评估。**

---

## 4. 数据准备流程详解（01–08）

### 4.1 01_list_candidate_datasets.py：候选数据集筛选

这是整个流程的入口。它的任务不是下载真实表达矩阵，而是先基于 CELLxGENE Census 的元数据，从大量公开数据集中自动筛出适合下游使用的数据集。

#### 主要工作

1. 按 `TARGET_TISSUES` 中定义的目标组织类型逐个查询 Census。
2. 从细胞级 `obs` 中仅读取当前步骤真正需要的列，例如：`dataset_id`、`cell_type`、`tissue`、`tissue_general`、`disease`、`suspension_type`、`is_primary_data`。
3. 对每个组织先做局部聚合，得到 dataset 级统计信息：`cell_count`、`unique_cell_types`、`tissues`、`diseases`。
4. 再把不同组织下的统计信息合并，形成全局 dataset 汇总表。
5. 根据配置阈值进行过滤，例如：dataset 细胞数上下限、最少细胞类型数。
6. 计算辅助选择分数，例如细胞类型丰富度、数据规模、组织混杂度、疾病混杂度。
7. 根据模式输出 `candidate_datasets.csv` 和 `selected_datasets.csv`。

#### 输出产物

- 候选数据集表
- 最终选中数据集表
- 每个组织的预览 CSV
- 各组织和全局统计表
- run summary

#### 这一步的意义

> **把“哪些数据值得下载和进入训练流程”自动化。**

---

### 4.2 02_export_selected_datasets.py：导出原始 h5ad

在 01 已经选好数据集之后，02 负责真正从 Census 导出对应的表达矩阵。

#### 主要工作

1. 读取 `selected_datasets.csv`。
2. 仅保留 `use=1` 的数据集。
3. 对每个 dataset 构造过滤表达式。
4. 调用 `cellxgene_census.get_anndata()` 拉取对应的 AnnData。
5. 保存为原始 `.h5ad` 文件到 `data/raw_h5ad/`。
6. 将来源元信息写入 `adata.uns["source_meta"]` 或 sidecar JSON 文件。
7. 对导出过程进行工程化保护：文件存在则跳过，导出失败时自动重试，中途失败时删除半成品，每完成一个 dataset 就更新 manifest。

#### 输出产物

- `data/raw_h5ad/*.h5ad`
- `*.source_meta.json`
- raw 导出 manifest
- run summary

#### 这一步的意义

> **把已选中的 dataset 真正落地到本地，作为后续处理原料。**

---

### 4.3 03_clean_and_standardize.py：清洗与标准化

02 导出的 raw h5ad 仍然是公开数据原貌，存在标签不统一、字段不完整、基因名不规整、类别稀疏等问题。03 的任务就是把这些原始数据整理成适合训练的 clean h5ad。

#### 主要工作

1. 遍历 `data/raw_h5ad/` 中的所有 h5ad。
2. 对极大文件做预筛选，必要时直接跳过，避免清洗时内存或 I/O 崩溃。
3. 统一基因名：优先使用 `feature_name`，保证 `var_names` 唯一。
4. 标准化 obs 字段：`cell_type`、`tissue_general`、`tissue`、`disease`。
5. 生成两个重要标签字段：
   - `cell_type_gold`：原始标签的清洗版
   - `cell_type_clean`：更适合训练的标准化标签
6. 过滤无效样本：非 primary data、缺失 cell type、含糊标签、过少基因、过少细胞的标签类别。
7. 保留原始计数矩阵到 `layers["counts"]`，便于后续 marker 提取。
8. 尝试加入 ontology 相关字段：`cell_ontology_id`、`cell_ontology_label`、`cell_ontology_parent_label`、`cell_type_status`、`cell_type_level`。
9. 生成附属文件：`label_counts.csv`、`label_counts_gold.csv`、`cell_type_mapping.csv`、`dataset_profile.json`。
10. 支持断点续跑：已经 clean 完整的文件直接跳过；如果主 clean 文件存在但 side outputs 缺失，则自动修复。

#### 输出产物

- `data/clean_h5ad/*.h5ad`
- 标签统计文件
- 标签映射文件
- dataset 画像文件
- clean manifest
- run summary

#### 这一步的意义

> **把原始单细胞数据整理成统一、干净、适合后续 marker 提取和模型训练的标准输入。**

---

### 4.4 04_make_marker_examples.py：提取 marker 样本

经过 03 的清洗之后，每个 h5ad 已经具备比较可靠的标签和表达矩阵。04 的任务是在每个数据集内，围绕细胞类型做差异表达分析，提取 marker gene 证据，并把这些证据写成后续建模使用的样本。

#### 主要工作

1. 读取 clean h5ad。
2. 检查目标分组列（通常是 `cell_type_clean`）是否存在且标签数至少为 2。
3. 对每一类细胞抽样，避免超大类压制计算。
4. 使用 `counts` 层作为 marker 分析输入。
5. 做轻量基因过滤、归一化、`log1p`。
6. 调用 `scanpy.tl.rank_genes_groups()` 做差异表达分析。
7. 针对每个 cell type 提取 top positive markers。
8. 过滤无信息基因、坏前缀基因以及不可靠 marker。
9. 构建 marker example 记录，内容包括 dataset 信息、tissue / disease 背景、`cell_type_clean`、gold 标签主值、`n_cells`、marker 基因列表、logFC / p 值等统计量。
10. 在增强版 v2 中，进一步加入 negative markers、ontology 信息、`marker_quality_score`、`hardness_flags`。

#### 输出产物

- `marker_examples.jsonl`
- `marker_examples_v2.jsonl`
- marker manifest
- v2 marker manifest
- run summary

#### 这一步的意义

> **把“表达矩阵 + 标签”转化为“细胞类型注释证据样本”，也就是后续 SFT 的原材料。**

---

### 4.5 05_make_sft_jsonl.py：构建 SFT 对话数据

04 得到的是 marker 证据样本，但还不是大模型训练能直接使用的格式。05 的作用就是把 marker 记录包装成 **Qwen3 可直接训练的多轮消息样本**。

#### 主要工作

1. 读取 `marker_examples.jsonl` 或 `marker_examples_v2.jsonl`。
2. 为每条样本构建 user prompt，写入 organism、tissue、disease/context、top marker genes 和任务要求。
3. 根据记录中的标签构建 assistant 端标准答案。
4. 标准答案以 JSON 形式输出，字段可能包括 `cell_type`、`supporting_markers`、`confidence`、`need_manual_review`、`rationale`。
5. 同时生成两套训练数据：标准 messages 和 no-think messages。
6. 额外保存 full records、manifest CSV、v2 SFT 数据和 distillation records。

#### 输出产物

- `sft_records_full.jsonl`
- `sft_messages.jsonl`
- `sft_messages_no_think.jsonl`
- v2 对应版本
- distill records
- manifest 与 summary

#### 这一步的意义

> **把 marker 样本真正变成大模型能学的“指令—回答”监督数据。**

---

### 4.6 06_split_and_validate_v2.py：切分 train / val / test

有了完整 SFT 数据后，需要严格切分训练集、验证集和测试集。06 的重点是 **按 dataset 级别切分**，避免同一个 dataset 同时出现在训练和测试中造成数据泄漏。

#### 主要工作

1. 读取 full SFT records。
2. 构建 dataset profile，包括 `dataset_id`、`dataset_title`、主组织 `main_tissue_general`、主疾病 `main_disease`、样本数 `n_records`、细胞类型数 `n_cell_types`。
3. 采用 dataset-level split：优先按主组织分层切分，小数据集场景下做特殊处理。
4. 在 val dataset 不足时，可从 train records 中抽少量样本构造 pseudo-val。
5. 额外导出难样本测试集 `test_hard`。
6. 可选地做 token 长度检查，确认样本长度分布。
7. 对 v2 数据复用相同 dataset 切分。
8. 构造 benchmark 子集，例如 rare subset、ontology-unmapped subset。

#### 输出产物

- `train_full.jsonl` / `val_full.jsonl` / `test_full.jsonl`
- `train_messages*.jsonl` / `val_messages*.jsonl` / `test_messages*.jsonl`
- `test_hard_*`
- v2 split
- benchmark manifest
- dataset profiles
- run summary

#### 这一步的意义

> **保证训练、验证、测试严格隔离，并尽可能让分布合理、评测可靠。**

---

### 4.7 07_build_ontology_resources.py：构建 ontology 资源

在这个项目里，细胞类型并不只是自由文本标签，还希望与 cell ontology 建立映射。07 的作用就是把已有的别名表和组织层级表构建成一个轻量级本地 ontology 索引。

#### 主要工作

1. 读取 `label_aliases.tsv` 和 `organ_hierarchy.tsv`。
2. 基于别名表按 `cell_ontology_id` 聚合。
3. 统一得到每个 ontology entry 的信息：标准标签、同义词、parent label、organ scope、label level。
4. 写出轻量级 JSONL 格式资源，供运行时查表使用。

#### 输出产物

- `ontology_index.jsonl`
- `cell_ontology_min.jsonl`
- ontology manifest

#### 这一步的意义

> **为后续标签规范化、ontology 映射和知识增强提供本地查找资源。**

---

### 4.8 08_build_marker_kb.py：构建 marker 知识库

除了 ontology 资源，这个项目还构建了一个本地 marker KB。它把外部整理的 marker 知识和从训练数据中归纳出来的 marker 证据合并成统一知识库。

#### 主要工作

1. 读取外部 marker KB，例如人工整理或导出的数据库。
2. 读取 `marker_examples_v2.jsonl` 中的训练导出 marker。
3. 将两者统一成统一 schema：`species`、`tissue_general`、`cell_type_label`、`cell_ontology_id`、`parent_label`、`marker_genes`、`weight`、`entry_type`、`evidence_level`。
4. 按 `(species, tissue, label)` 去重合并。
5. 外部知识优先，训练导出知识作为补充。
6. 合并同标签下的 marker gene 列表，限制长度。

#### 输出产物

- `train_marker_kb.jsonl`
- `merged_marker_kb.jsonl`
- marker KB manifest

#### 这一步的意义

> **把“训练数据里的经验”与“外部已有知识”融合，形成可用于推理增强和知识检索的 marker 知识库。**

---

## 5. 训练流程详解

数据准备完成后，就进入模型训练阶段。你的训练脚本本质上是基于 **Transformers + TRL + PEFT**，对本地 `Qwen3-4B` 做 **LoRA 监督微调**。

### 5.1 训练输入

训练脚本直接读取：

- `train_messages_no_think_v2.jsonl`
- `val_messages_no_think_v2.jsonl`

这些数据来自 06 的切分结果，已经是规范的多轮 `messages` 格式。

### 5.2 数据转换逻辑

训练脚本并不是直接把整段 `messages` 原样送进 Trainer，而是先做了一层 **prompt-completion 转换**：

1. 找到一条样本中最后一条 assistant 消息。
2. 将其前面的 system + user 消息作为 prompt。
3. 将最后的 assistant 消息作为 completion。
4. 用 tokenizer 的 `apply_chat_template()` 把 prompt 渲染成模型真正使用的输入格式。

这种设计的意义是明确训练目标：模型不是去复述整段对话，而是学习“根据前文生成最终回答”。

### 5.3 训练前检查

脚本在真正开训前做了不少保护：

- 检查训练/验证文件是否存在、是否为空。
- 逐条检查 `messages` 格式是否合法。
- 跳过无 assistant、内容为空、格式异常的样本。
- 打印数据统计：总样本数、保留数、各类跳过原因数量。
- 打印少量 dataset preview，方便人工检查 prompt 和 completion。

### 5.4 模型加载与 LoRA 配置

训练使用的是本地基础模型：`my_models/Qwen/Qwen3-4B`。

在此基础上挂载 LoRA adapter，典型配置包括：

- `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `target_modules = "all-linear"`

这说明训练策略是对线性层较广泛地加适配器，而不是只对极少数 attention 投影层做修改。

### 5.5 训练超参数与显存优化

训练脚本中显式写出了常见训练参数，例如：

- `MAX_LENGTH = 1024`
- `NUM_TRAIN_EPOCHS = 8`
- `PER_DEVICE_TRAIN_BATCH_SIZE = 1`
- `GRADIENT_ACCUMULATION_STEPS = 4`
- `LEARNING_RATE = 5e-5`
- `LR_SCHEDULER_TYPE = "cosine"`
- `WARMUP_RATIO = 0.05`

同时也做了显存优化：

- 启用 `bf16`
- 关闭 `use_cache`
- 开启 `gradient_checkpointing`

这些组合适合在较有限显存条件下做 4B 模型 LoRA 微调。

### 5.6 训练器与保存逻辑

训练器使用的是 `TRL` 的 `SFTTrainer`，主要特点是：

- 支持 SFT 训练流程
- 按 step 做日志记录
- 按 step 做验证
- 按 step 保存 checkpoint
- 最终根据 `eval_loss` 自动保留 best model

训练结束后，脚本会保存：

- 最终 adapter
- tokenizer
- metrics
- trainer state

### 5.7 训练阶段的本质作用

> **把前面构建好的单细胞注释指令样本真正学进 Qwen3，使其具备面向 marker-gene 证据的结构化细胞类型注释能力。**

---

## 6. 推理与评估流程详解

训练得到 LoRA adapter 之后，就进入推理和测试阶段。你的推理脚本不只是“跑生成”，而是把 **生成、解析、比较、统计** 全部串成了完整评测闭环。

### 6.1 推理输入与模型组成

推理脚本加载：

- 基础模型：`Qwen3-4B`
- 训练好的 adapter：LoRA 输出目录
- 测试集：`test_messages_no_think_v2.jsonl`

也就是说，实际推理模型是：

> **Base Model + LoRA Adapter**

### 6.2 推理流程

对每条测试样本，脚本会做以下步骤：

1. 读取完整 `messages`。
2. 提取其中最后一条 assistant 作为 gold 标注。
3. 去掉 assistant，只保留 system + user 作为推理输入。
4. 使用 `apply_chat_template()` 生成 prompt。
5. 调用 `model.generate()` 得到模型输出。
6. 解码生成文本。

推理参数采用偏保守的设置，例如：

- `do_sample = False`
- `temperature = 1.0`
- `max_new_tokens = 256`

这更偏向稳定、可复现的评测，而不是追求采样多样性。

### 6.3 输出解析：从文本到 JSON

模型输出不一定总是非常干净，因此脚本写了比较鲁棒的解析逻辑：

1. 先尝试把整体文本当作 JSON 解析。
2. 如果失败，先移除可能干扰的 `</think>`。
3. 再用正则匹配第一个 `{...}` 块。
4. 若匹配到合法 JSON，则进一步解析。
5. 如果完全解析失败，则记录 `parse_ok = False` 和错误原因。

同样地，gold assistant 内容也会用同样方式解析，保证比较逻辑统一。

### 6.4 细胞类型标准化与层级比较

这部分是推理脚本最有价值的地方之一。它没有把评估停留在“字符串是否相同”，而是引入了更贴近单细胞注释任务的层级化比较。

#### 主要处理包括

1. **文本归一化**：小写化、去下划线、去连字符、去冗余空格。
2. **cell type 标准化**：把缩写或常见写法映射到较统一的表达，例如 NK、CD4 T cell、CD8 T cell。
3. **谱系推断**：判断标签属于哪类大谱系，例如 T cell、NK、B cell、Monocyte、Dendritic、Macrophage、Erythroid 等。
4. **粒度比较**：判断预测是比 gold 更粗、更细，还是同层级。
5. **重要 subtype 冲突识别**：例如 CD4 vs CD8、naive vs memory、NK bright vs NK dim。

#### 最终比较结果包括

- exact match
- normalized exact match
- same lineage
- granularity relation
- match level
- severe error

其中 `cross_lineage_error` 会被视为较严重的生物学错误。

### 6.5 结构化字段级比较

除了 `cell_type`，脚本还会比较其它结构化字段：

- `confidence_label`
- `confidence_score`
- `need_manual_review`
- `decision`
- `novelty_flag`
- `evidence_support_level`
- `supporting_markers`

对于 supporting markers，会计算 Jaccard 相似度，而不是简单的完全一致。

### 6.6 结果保存

推理结束后，脚本会保存三类主要结果：

#### 1）`predictions.jsonl`
逐样本完整结果，适合程序再处理。

#### 2）`predictions.csv`
更便于人工查看和筛选错误样本。

#### 3）`summary.json`
整体统计指标，包括：

- parse 成功率
- gold parse 成功率
- cell type exact accuracy
- normalized exact accuracy
- same-lineage rate
- severe error rate
- confidence / decision / novelty 等字段匹配率
- 平均 marker Jaccard
- high-risk error 数量
- 各种 match level 的计数

### 6.7 推理阶段的本质作用

> **不只是看模型“答没答对”，而是从结构化字段、谱系层级和风险等级三个层面综合评估单细胞注释模型的表现。**

---

## 7. 整个项目的数据流转关系

从数据流转的角度，这个项目的逻辑可以概括为：

```text
CELLxGENE Census
   ↓
01 选择合适数据集
   ↓
02 导出 raw h5ad
   ↓
03 清洗、标准化、ontology 映射
   ↓
04 提取 marker examples
   ↓
05 构建 SFT messages / full records
   ↓
06 切分 train / val / test
   ↓
训练脚本：Qwen3-4B + LoRA SFT
   ↓
推理脚本：test 集批量生成
   ↓
自动解析 + 字段比较 + 层级评估
```

如果把知识资源也加入其中，则可以补成：

```text
03 clean h5ad ----------------→ 07 ontology resources
04 marker examples -----------→ 08 marker KB
05/06 SFT data --------------→ 训练
08 marker KB / 07 ontology ---→ 后续知识增强潜力
```

---

## 8. 项目设计上的几个关键特点

### 8.1 数据集级切分，避免泄漏
项目没有按样本随机切分，而是按 `dataset_id` 切分，这能有效避免同一个公开数据集既出现在训练又出现在测试中，保证评测更真实。

### 8.2 marker 驱动的任务建模
项目不是直接让模型看完整表达矩阵，而是围绕 marker gene 证据构造任务，更贴近人工注释时“根据 marker 推断细胞类型”的实际思路。

### 8.3 标签规范化与 ontology 映射
项目不是单纯把 cell type 当作自由文本处理，而是尽量与 ontology 体系挂钩，这为后续标准化、迁移和知识增强提供了基础。

### 8.4 结构化输出而非自由文本
模型学习的不是纯自然语言回答，而是带明确字段的 JSON 结构输出。这使得自动评测、误差分析和后续系统集成都更方便。

### 8.5 评测不止 accuracy
项目评测引入了：

- normalized exact match
- same lineage
- severe error
- supporting markers overlap
- decision / confidence 等字段一致性

这让模型评测更贴近生物学任务实际。

### 8.6 兼顾训练数据与知识资源
项目除了构建 SFT 数据，还额外构建 ontology index 和 marker KB，体现出它不仅是一个训练工程，也是在为更强的知识增强型系统打底。

---

## 9. 项目的最终目标可以怎样表述

如果用一句更系统化的话来总结，本项目的最终目标可以写成：

> **构建一个面向单细胞 RNA-seq 细胞类型注释的领域大模型训练与评测框架，使公开单细胞数据能够经过自动化处理后转化为高质量监督样本，并支持基于 Qwen3 的结构化注释预测、层级化结果评估以及知识增强扩展。**

从研究角度看，这个项目至少同时覆盖了四件事：

1. 单细胞数据工程化处理
2. marker 驱动的指令数据构建
3. 领域大模型微调
4. 结构化推理与多层级评测

---

## 10. 一版简洁总结

如果需要在 README 首页用一小段话概括整个项目，可以直接写成下面这样：

> 本项目实现了一条面向单细胞 RNA-seq 细胞类型注释的完整流程：首先从 CELLxGENE Census 自动筛选并导出合适的数据集；随后对原始 h5ad 数据进行清洗、标签标准化和 ontology 映射；再基于差异表达分析提取 marker gene，构建适用于 Qwen3 的 SFT 指令数据，并按 dataset 级别切分训练、验证和测试集；在此基础上，项目使用 LoRA 对本地 Qwen3-4B 进行监督微调，并在测试集上执行结构化批量推理，最终通过细胞类型匹配、谱系一致性、marker 支持度以及风险错误统计等指标对模型进行系统评估。同时，项目还构建了 ontology 资源与 marker 知识库，为后续知识增强和更复杂的注释系统奠定基础。
