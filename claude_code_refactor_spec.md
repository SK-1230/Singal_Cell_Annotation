# Claude Code 重构任务说明（单细胞注释项目）

## 1. 任务背景

当前项目已经完成了以下主链路：

- 从 CELLxGENE Census 筛选候选数据集
- 导出 raw h5ad
- 清洗与标准化
- 提取 marker examples
- 构建 SFT 数据
- 切分 train/val/test
- 对 Qwen3 进行 LoRA 微调
- 在 test 集上做批量推理与自动评测

当前一次典型推理评测结果大致如下：

- `parse_ok_rate = 1.0`
- `cell_type_exact_accuracy ≈ 0.18`
- `cell_type_same_lineage_rate ≈ 0.67`
- `cell_type_severe_error_rate ≈ 0.33`
- `high_risk_error_count = 16`

这说明：

1. **模型格式学习是成功的**：JSON 输出基本稳定，解析率很高。
2. **模型学到了一些粗粒度生物学模式**：same-lineage rate 不低。
3. **模型没有稳定学会统一的细胞类型精确标签映射**：exact accuracy 很低。
4. **存在不少跨谱系严重错误**：说明当前训练目标和数据构造仍然有明显问题。

因此，这次重构的重点不是“单纯改 LoRA”，而是优先修正：

- 数据集选择策略
- 标签 harmonization / ontology 对齐
- marker 构造逻辑
- SFT 目标格式
- split 粒度
- 推理阶段是否真正使用 ontology / marker KB

---

## 2. 对问题原因的判断

### 2.1 当前效果不好的最主要原因

我当前的判断排序如下：

1. **数据与标签体系问题（主因）**
2. **模型容量偏小（次主因，尤其是 Qwen3-4B）**
3. **训练样本过于压缩，每类监督信号偏薄**
4. **推理没有真正利用 ontology / marker KB**
5. **LoRA 本身不是主问题**

### 2.2 为什么不是首先怀疑 LoRA

因为当前结果显示：

- 输出结构是稳定的
- JSON 基本都能被解析
- confidence / decision / novelty 等字段也并非完全失控

这通常说明训练框架本身是跑通的，LoRA 并不是最主要瓶颈。

### 2.3 为什么更怀疑数据与标签构造

现有流程中，虽然前面已经生成了 ontology 相关列，例如：

- `cell_ontology_id`
- `cell_ontology_label`
- `cell_ontology_parent_label`

但后续 SFT 监督目标仍然较大程度上依赖 `cell_type_clean` 或 dataset 自带标签，而不是强制以统一 ontology 目标作为主标签。

这会导致模型学到的是：

- 各个研究/数据集内部自己的命名习惯
- 粒度不一致的 label space
- 同义词、亚型、研究者自定义状态的混合标签

因此模型容易学会“大类判断”，但难以学会“跨研究统一标签映射”。

这与当前结果高度一致：

- exact 低
- same lineage 还可以
- severe error 不低

---

## 3. 本次 Claude Code 改造总目标

请 Claude Code 以“**先修正数据与标签体系，再提升训练与推理闭环**”为原则，完成一轮较系统的代码重构。

### 总体目标

将项目改造成如下风格：

1. **先构建一版更干净的 Phase-A 数据集**
2. **训练目标尽量以 ontology 对齐标签为主**
3. **marker 构造围绕统一 target label，而不是原始 clean label**
4. **切分尽量按 study/collection 级别，而不是仅 dataset_id**
5. **推理阶段接入 07/08 产出的 ontology / marker KB**
6. **允许保留当前 LoRA + TRL + PEFT 主训练框架**
7. **在必要处可新增辅助函数、辅助文件、manifest 字段、summary 输出**

---

## 4. Claude Code 可以自主发挥的范围

以下事情允许 Claude Code 根据代码结构自行判断并优化：

### 允许自主发挥的方面

- 可以对函数名、局部结构、工具函数进行合理重构
- 可以新增辅助文件，例如：
  - `src/sca/data/target_labeling.py`
  - `src/sca/data/retrieval.py`
  - `src/sca/data/curation_rules.py`
- 可以适度补充日志、manifest、summary 字段
- 可以在不破坏现有主目录结构的前提下新增少量中间输出文件
- 可以在 infer 中引入轻量级 retrieval / reranking 逻辑
- 可以对 config 增加新参数
- 可以把一部分硬编码逻辑迁移到 config 中

### 不要过度做的事情

- 不要完全重写整个项目目录结构
- 不要把所有脚本强行合并成一个超大脚本
- 不要无必要地删除已有 v1/v2 分支产物
- 不要先把 LoRA / Trainer 框架彻底换掉
- 不要在未统一标签体系前就优先改成复杂 RL / DPO 训练

---

## 5. 最高优先级修改事项

---

## 5.1 修改 `scripts/data_prep/data_prep_config.py`

### 目标
新增一套“更严格的数据选择 + ontology 目标标签 + study 级 split + retrieval 配置”。

### 希望新增或调整的方向

#### 1）收紧 Phase-A 目标组织
建议下一轮默认只保留：

- `blood`
- `lung`
- `kidney`
- `liver`

建议先移除：

- `brain`
- `intestine`
- `skin`

原因：这些组织当前会显著增加标签复杂度和疾病异质性。

#### 2）加入严格数据集筛选参数
建议增加类似以下配置项（命名可由 Claude Code 优化）：

- `STRICT_NORMAL_ONLY`
- `MAX_ALLOWED_N_TISSUES`
- `MAX_ALLOWED_N_DISEASES`
- `EXCLUDE_DISEASE_KEYWORDS`
- `EXCLUDE_TITLE_KEYWORDS`
- `EXCLUDE_COLLECTION_KEYWORDS`
- `PREFER_REFERENCE_KEYWORDS`

核心思想：

- 优先保留正常样本
- 优先保留 atlas / reference 风格数据集
- 排除 COVID / cancer / tumor / glioblastoma / leukemia / adenoma 等明显高异质数据

#### 3）加入 ontology 目标标签相关参数
建议增加类似：

- `REQUIRE_ONTOLOGY_MAPPED_DATASET`
- `MIN_DATASET_MAPPED_RATIO`
- `REQUIRE_MAPPED_CELL_FOR_TRAIN`
- `TARGET_LABEL_MODE`
- `TARGET_FALLBACK_TO_PARENT`

#### 4）加入 split 分组参数
建议增加类似：

- `SPLIT_GROUP_KEY = "collection_doi"`
- 支持 fallback 到 `collection_name`
- 再 fallback 到 `dataset_id`

#### 5）加入 KB retrieval 参数
建议增加类似：

- `USE_MARKER_KB_RETRIEVAL`
- `KB_RETRIEVAL_TOPK`
- `KB_TISSUE_BONUS`
- `KB_OVERLAP_WEIGHT`

---

## 5.2 修改 `01_list_candidate_datasets.py`

### 当前问题
当前 01 的自动筛选主要依赖：

- `cell_count`
- `unique_cell_types`
- `n_tissues`
- `n_diseases`

但没有真正显式排除：

- COVID
- 肿瘤
- 严重疾病
- 强研究者自定义数据子集
- 高异质 collection

### 修改目标
让 01 的候选集更贴近“训练统一细胞类型注释模型”的真实需求。

### 需要完成的方向

1. 在候选过滤阶段增加 metadata guardrails：
   - 按 disease/title/collection 关键词过滤
   - 对 `normal-only` 模式做支持
   - 对 `n_tissues` / `n_diseases` 增加强约束

2. 在自动打分阶段，增加“reference/atlas 偏好项”

3. 建议支持 `manual_template` 模式下：
   - 生成候选模板
   - 默认 `use=0`
   - 可增加 `recommended=1`
   - 让人工最终确认，而不是直接自动全量下游使用

4. 输出中尽量保留/补充这些字段，便于后续筛选：
   - `collection_name`
   - `collection_doi`
   - `dataset_title`
   - `diseases`
   - `n_tissues`
   - `n_diseases`
   - 任何有助于人工检查的参考评分字段

### 允许 Claude Code 的发挥
- 可以自己封装 `passes_metadata_guardrails()` 之类的函数
- 可以把关键词过滤逻辑抽到辅助文件
- 可以新增更清晰的评分项

---

## 5.3 修改 `03_clean_and_standardize.py`

### 当前问题
03 虽然已经生成 ontology 相关列，但这些列没有真正成为后续训练主标签。

### 修改目标
在 clean h5ad 中明确构建：

- `cell_type_target_id`
- `cell_type_target_label`

使其成为后续 marker / SFT 的主监督目标。

### 需要完成的方向

1. 新增 target label 构造逻辑：
   - 优先使用 `cell_ontology_label`
   - 保留 `cell_ontology_id`
   - 在必要时允许 fallback 到 `cell_ontology_parent_label`

2. 支持 dataset-level ontology 质量控制：
   - 若映射率太低，可直接 skip 整个 dataset

3. 支持 cell-level 过滤：
   - 若训练要求必须是 ontology mapped cell，则过滤掉没有 target label 的 cell

4. clean 输出中应保留并写出：
   - `cell_type_target_id`
   - `cell_type_target_label`

5. manifest / profile 中可增加：
   - target mapping ratio
   - n_target_labels
   - n_target_ids

### 允许 Claude Code 的发挥
- 可以新增辅助函数，例如 `build_target_label_columns()`
- 可以在 sidecar profile 中加入更多 target-label 相关统计

---

## 5.4 修改 `04_make_marker_examples.py`

### 当前问题
当前 marker 提取仍主要围绕 `cell_type_clean`。

### 修改目标
让 marker extraction 以 **统一 target label** 为主，而不是原始 clean label。

### 需要完成的方向

1. 将分组列默认改为：
   - `cell_type_target_label`
   - 或由 config 控制

2. marker record 中建议保留多套标签字段：
   - `cell_type_source_clean`
   - `cell_type_target_label`
   - `cell_type_target_id`
   - `cell_ontology_id`
   - `cell_ontology_label`
   - `cell_ontology_parent_label`

3. record 中尽量透传 study 信息：
   - `collection_name`
   - `collection_doi`

4. marker v2 仍应保留：
   - positive markers
   - negative markers
   - marker quality score
   - hardness flags

### 允许 Claude Code 的发挥
- 可以优化 v1 / v2 record schema
- 可以对 output schema 做适度清理，但要兼顾后续 05/06 使用

---

## 5.5 修改 `05_make_sft_jsonl.py`

### 当前问题
05 的旧逻辑中，assistant 目标较依赖原始 clean label；而且 supporting markers 往往直接取输入 marker 的前几项，容易让模型主要学会“复制 marker 和遵循输出格式”。

### 修改目标
构建一版新的 SFT 样本，使监督目标尽量转向：

- canonical / ontology-aligned label
- ontology id
- parent cell type
- 更合理的 confidence / review 风格

### 需要完成的方向

1. 新增或重构 builder（名称可自由发挥，例如 v3）

2. user prompt 中建议加入：
   - organism
   - tissue
   - disease/context
   - top positive markers
   - 明确要求输出 canonical label
   - 明确要求尽量输出 ontology id
   - 若子类不确定可输出 parent type

3. assistant answer 中建议以以下字段为核心：
   - `cell_type`
   - `cell_ontology_id`
   - `parent_cell_type`
   - `supporting_markers`
   - `confidence_label`
   - `need_manual_review`
   - `decision`
   - `rationale`

4. full records 中建议保留：
   - `collection_name`
   - `collection_doi`
   - `cell_type_target_label`
   - `cell_type_target_id`
   - `cell_ontology_parent_label`

5. 允许保留 no-think 版本，但应确保训练目标比当前更统一

### 允许 Claude Code 的发挥
- 可以直接在现有文件里实现 v3 builder
- 也可以抽出到 `src/sca/data/sft_builder_v3.py`
- 可以自主决定是否保留旧版 builder 作为兼容路径

---

## 5.6 修改 `06_split_and_validate_v2.py`

### 当前问题
当前 split 主要按 `dataset_id` 级别进行。虽然比 record-level split 好，但对同一 collection 拆成多个 dataset 子集的情况仍然不够严格。

### 修改目标
改成“**优先按 study / collection 级别 split**”。

### 需要完成的方向

1. 支持从 record 中解析 group key：
   - 优先 `collection_doi`
   - fallback `collection_name`
   - 再 fallback `dataset_id`

2. profile 构建时，可以改成按 `group_id` 聚合，或者保留 dataset profile 同时新增 `group profile`

3. split 时应确保：
   - train / val / test 在 group 层面不交叉

4. 可继续保留：
   - small dataset special handling
   - pseudo-val（如确实仍需要）
   - hard test 导出

5. benchmark subset 逻辑可保留，但需兼容新分组方式

### 允许 Claude Code 的发挥
- 可以新增 `resolve_group_key()`
- 可以让 summary 同时输出 dataset-level 和 group-level 分配情况

---

## 5.7 修改训练脚本（当前训练 py / sh）

### 当前判断
训练框架本身不是这轮最该重写的重点。

### 修改目标
在不推翻现有 LoRA + TRL + PEFT 框架的前提下，让训练更适配新的数据格式。

### 建议方向

1. 保持当前训练主框架可用
2. 兼容新的 full/messages schema
3. 若 prompt 长度变长，适当放宽：
   - `MAX_LENGTH`
4. 建议降低过密的 eval/save 频率
5. 可以根据资源情况，支持：
   - 继续使用 4B
   - 或可选切到 8B

### 建议原则
- 若数据/标签体系未修好，不要优先大改训练器
- 训练超参可以做温和修正，但不要把这一步作为主战场

---

## 5.8 修改推理脚本（`scripts/infer/...py`）

### 当前问题
推理脚本目前基本是“基础模型 + LoRA adapter 直接生成”，没有真正利用：

- `ontology_index.jsonl`
- `merged_marker_kb.jsonl`

### 修改目标
实现一版轻量级 retrieval-augmented inference。

### 需要完成的方向

1. 推理输入建议优先改成 `test_full_v2.jsonl` 或等价 full schema
2. 在 infer 中加载 marker KB
3. 基于 marker overlap + tissue bonus 做一个简单 top-k 检索
4. 将检索候选拼进 prompt 作为 hint
5. 保持原有 JSON 解析和比较框架
6. 尽量新增 ontology ID 评测指标，例如：
   - `cell_ontology_id_match`

### 允许 Claude Code 的发挥
- 可以自己实现一个轻量检索器
- 可以抽到 `src/sca/data/retrieval.py`
- 可以自主决定 prompt 注入格式
- 只要最终 inference 更像一个“带知识提示的注释流程”即可

---

## 6. 是否允许新增文件

允许，且推荐在必要时新增少量辅助模块。

### 推荐可新增的文件类型

- `src/sca/data/curation_rules.py`
- `src/sca/data/target_labeling.py`
- `src/sca/data/retrieval.py`
- `src/sca/data/sft_builder_v3.py`
- `src/sca/data/split_grouping.py`

### 原则
- 新增文件应服务于“逻辑更清晰、更可维护”
- 不要无节制拆分
- 保持项目主结构稳定

---

## 7. 本次修改的建议实验顺序

Claude Code 在修改完成后，建议按以下顺序组织实验：

### 实验 A：只改数据与标签，不改模型大小
目标：验证主因是否确实是数据与标签体系。

建议：

- 先保持 4B
- 重做 01→06
- 重训
- 重新 infer

### 实验 B：在实验 A 基础上，加 KB retrieval 推理
目标：验证推理阶段知识增强的收益。

建议：

- 不一定重训
- 先只改 infer
- 对比 retrieval 前后指标

### 实验 C：在实验 A/B 基础上，再考虑切 8B
目标：验证模型容量提升的额外收益。

---

## 8. 验收标准

Claude Code 修改完成后，至少应满足以下验收标准：

### 功能性验收

1. 01–08 仍能完整串起来运行
2. 新 config 能被各脚本正确读取
3. 新增字段不会导致 05/06/infer 崩溃
4. split 能在 group/study 级别避免交叉
5. infer 能正常使用 KB 检索（如启用）

### 数据层验收

1. clean h5ad 中存在 target label 字段
2. marker records 存在 target label / ontology 字段
3. SFT records 的主输出目标不再只是旧的 clean label

### 评测层验收

至少输出并可比较以下指标：

- `cell_type_exact_accuracy`
- `cell_type_normalized_exact_accuracy`
- `cell_type_same_lineage_rate`
- `cell_type_severe_error_rate`
- `high_risk_error_count`
- 若实现了 ontology ID 评测，再增加：
  - `cell_ontology_id_accuracy`

### 期望改进方向

本轮修改后，希望看到至少如下趋势：

- exact / normalized exact 提升
- same-lineage 不下降或略升
- severe error 明显下降
- high-risk error 明显下降

---

## 9. 最后的实现原则

请 Claude Code 按以下优先级理解本次任务：

1. **先统一数据与标签体系**
2. **再改 marker / SFT 构造**
3. **再改 split 粒度**
4. **再接入 retrieval 推理**
5. **训练框架只做必要适配，不要先大改**

最核心的一句话是：

> **让模型学习“统一 ontology 目标下的细胞类型注释”，而不是学习“不同研究各自的标签习惯”。**

