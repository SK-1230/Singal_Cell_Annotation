# SCA 项目下一轮实验优先级执行文档（精简版）

## 1. 文档目标

本文件用于指导下一轮实验与代码修改。核心原则不是“立刻大改全部模块”，而是：

1. **先诊断，再修改**  
2. **先验证主瓶颈，再投入大规模重构**
3. **先收缩任务、净化监督，再扩数据、升级模型**

当前项目已经形成了基本闭环：数据处理 → SFT 训练 → KB 检索推理 → 评估。  
但是从现有结果看，当前最优模型仍然存在以下问题：

- Exact Accuracy 偏低
- Severe Error 偏高
- zero-shot 与长尾类型表现差
- ontology_id 质量不稳定
- 当前训练目标过宽，可能稀释了主任务学习信号

因此，下一轮的首要目标不是继续堆功能，而是**先用诊断实验明确：到底是哪一类因素在拖后腿**。

---

## 2. 总体执行原则

下一轮实验按以下顺序推进：

### Phase 0：新增“诊断文件夹”，优先确认主瓶颈
先不要直接重构全项目。  
先在 `scripts/diagnosis/` 下新增若干诊断脚本，回答以下问题：

1. 当前模型主要是被 **任务定义过宽** 拖累，还是被 **数据量不足** 拖累？
2. 当前 KB 检索到底是 **增益** 还是 **噪声源**？
3. ontology_id 是否应该继续作为主训练目标，而不是改为后处理？
4. 当前错误主要来自：
   - zero-shot
   - rare class
   - 组织分布偏移
   - 自蒸馏标签噪声
   - JSON 多字段生成负担

### Phase 1：基于证据，先收缩主任务
如果诊断表明主任务被结构化生成拖累，则优先改成：
- 只预测 `cell_type`
- 可选再预测 `parent_cell_type`

先移除：
- `ontology_id`
- `rationale`
- `confidence`
- `decision`
- `novelty_flag`
- `evidence_support_level`

### Phase 2：净化监督信号
如果诊断表明标签质量明显拖累性能，则优先：
- 清洗或重建训练标签
- 让 `cell_type` 和 `ontology_id` 来自更可靠来源
- 减少模型学习“自蒸馏输出风格”的风险

### Phase 3：扩数据与补覆盖
在任务收缩、监督净化之后，再做：
- 修复失败数据集处理
- 增加 lung / kidney / brain
- 引入高质量公共数据集
- 提升 KB 覆盖

### Phase 4：最后再碰大模型/复杂训练策略
包括：
- 更大底模
- 更复杂 loss
- curriculum
- ensemble
- dense retrieval

---

## 3. 第一优先级：新增诊断模块（必须先做）

新增目录建议如下：

```text
scripts/
├── diagnosis/
│   ├── README.md
│   ├── diag_config.py
│   ├── run_all_diagnosis.py
│   ├── analyze_error_buckets.py
│   ├── ablate_output_schema.py
│   ├── ablate_kb_retrieval.py
│   ├── ablate_ontology_target.py
│   ├── analyze_label_noise.py
│   ├── analyze_data_coverage.py
│   └── generate_diagnosis_report.py
```

如有需要，也可新增：

```text
src/sca/diagnosis/
├── __init__.py
├── metrics.py
├── bucket_analysis.py
├── label_quality.py
└── report_utils.py
```

---

## 4. 诊断模块设计要求

## 4.1 `analyze_error_buckets.py`
### 目标
将当前模型错误拆解为不同来源，明确性能下降主要来自哪里。

### 需要分析的 bucket
至少输出以下分桶统计：

- seen vs zero-shot
- frequent class vs rare class
- mapped ontology vs unmapped ontology
- high marker quality vs low marker quality
- tissue bucket（blood/liver/lung/kidney/brain/...）
- hard subset vs normal subset
- with parent-known vs parent-unknown

### 输出内容
至少输出：

- 每个 bucket 的样本数
- exact accuracy
- ontology compatible
- severe error rate
- same lineage rate

### 验收标准
满足以下条件才算完成：

- 可以自动读取推理结果与 gold 文件
- 自动输出 CSV 报表
- 自动生成 Markdown 汇总
- 能明确指出 top-3 最致命错误来源

---

## 4.2 `ablate_output_schema.py`
### 目标
验证“当前训练目标过宽”是否是主瓶颈。

### 需要支持的模式
至少支持以下训练/推理模式对比：

1. **full_json**  
   当前完整输出字段

2. **cell_type_only**  
   只输出：
   ```json
   {"cell_type": "..."}
   ```

3. **cell_type_parent**
   只输出：
   ```json
   {
     "cell_type": "...",
     "parent_cell_type": "..."
   }
   ```

### 要求
- 复用现有训练脚本，尽量少改主干
- 新增 schema 配置项，而不是复制多套训练代码
- 推理脚本也要适配不同 schema

### 关键比较指标
- exact accuracy
- severe error rate
- parse ok rate
- ontology compatible
- 训练 loss 与任务指标是否背离

### 验收标准
若出现以下任一结果，则认为“任务定义过宽”成立：

- `cell_type_only` 相比 `full_json`，exact 提升 ≥ 5 个百分点
- `cell_type_only` severe error 下降 ≥ 5 个百分点
- `cell_type_only` 在相近 loss 下有明显更好泛化
- `full_json` loss 更低但 exact 更差，说明模型过多拟合结构字段

---

## 4.3 `ablate_kb_retrieval.py`
### 目标
验证 KB 当前到底是帮助还是添乱。

### 需要支持的模式
1. no_kb  
2. jaccard_kb（当前方案）  
3. oracle_parent_hint（实验性，可用 gold parent 模拟上限）  
4. optional: bm25_kb（若实现成本不高）

### 输出内容
对比不同模式下的：

- exact accuracy
- ontology compatible
- severe error
- zero-shot 子集表现
- rare 子集表现

### 验收标准
- 如果 `no_kb >= jaccard_kb`，则当前 KB 视为“未形成稳定增益”
- 如果 `jaccard_kb` 只在 seen class 有帮助、在 zero-shot 上无帮助甚至更差，则说明检索质量不足
- 如果 oracle hint 明显更高，说明“有用的外部知识”理论上可提升，但当前检索实现太弱

---

## 4.4 `ablate_ontology_target.py`
### 目标
验证 ontology_id 是否应该继续作为主训练目标。

### 需要支持的设置
1. 模型直接生成 `cell_type + ontology_id`
2. 模型只生成 `cell_type`，再由规则映射 ontology_id
3. 模型生成 `cell_type + parent_cell_type`，再后处理 ontology_id

### 需要比较
- exact accuracy
- ontology_id accuracy
- ontology compatible
- severe error
- JSON 解析稳定性

### 验收标准
若“只生成 cell_type + 后处理 ontology_id”整体优于直接生成 ontology_id，则下一轮主线应移除 ontology_id 的直接生成监督。

---

## 4.5 `analyze_label_noise.py`
### 目标
评估当前 SFT 标签中哪些字段不可靠，以及噪声的严重程度。

### 分析内容
至少检查：

- `cell_type` 是否与原始 cluster/gold 一致
- `ontology_id` 是否在官方 CL 中存在
- `parent_cell_type` 是否真为 ontology 父类或合法上位类
- `rationale` 是否支持预测 label，而不是与 label 不一致
- 当前样本中由模型自蒸馏生成的字段，有多少存在逻辑冲突

### 推荐输出
为每条训练样本打标签：

- clean
- weak
- noisy
- invalid

并输出总体占比。

### 验收标准
- 能统计出噪声样本比例
- 能列出 top-20 典型问题样本
- 若 noisy + invalid 占比明显，则必须先做监督净化

---

## 4.6 `analyze_data_coverage.py`
### 目标
量化当前数据覆盖缺口，避免“拍脑袋扩数据”。

### 统计维度
- 每个 tissue 的 train/val/test 样本数
- 每个细胞类型的 train count
- zero-shot 类型列表
- rare 类型列表（如 train≤2）
- ontology mapped/unmapped 分布
- marker quality 分布
- 每个数据集贡献的细胞类型数与样本数

### 验收标准
- 自动输出 coverage 报告
- 明确指出最该补的数据类型：
  - 最缺的 tissue
  - 最缺的 lineage
  - 最该优先补的 cell types
- 可作为扩数据清单的依据

---

## 5. 新增诊断阶段的总验收标准

必须达到以下结果，才进入下一阶段修改：

1. 已完成至少 3 组关键消融：
   - schema 消融
   - KB 消融
   - ontology 目标消融

2. 已完成至少 2 组定性分析：
   - error bucket 分析
   - label noise 分析

3. 已生成一份统一诊断报告：
   - `output/diagnosis/<timestamp>/diagnosis_report.md`

4. 报告中必须明确回答以下问题：
   - 当前第一瓶颈是不是任务定义过宽？
   - 当前 KB 是否形成稳定收益？
   - ontology_id 是否应改为后处理？
   - zero-shot/rare/tissue shift 哪个影响最大？
   - 当前自蒸馏标签噪声是否严重到必须先处理？

---

## 6. 第二优先级：根据诊断结果收缩任务

若诊断结果支持“full_json 拖累主任务”，则立即执行。

## 6.1 修改目标
将主训练目标收缩为：

### 最简方案 A
```json
{"cell_type": "..."}
```

### 方案 B
```json
{
  "cell_type": "...",
  "parent_cell_type": "..."
}
```

### 暂时移除字段
- `cell_ontology_id`
- `supporting_markers`
- `confidence_label`
- `confidence_score`
- `need_manual_review`
- `decision`
- `novelty_flag`
- `evidence_support_level`
- `rationale`

## 6.2 推荐代码修改位置
- `src/sca/data/sft_builder.py`
- `scripts/data_prep/05_make_sft_jsonl.py`
- `scripts/train/train_config.yaml`
- `scripts/infer/infer_qwen3_kb_retrieval.py`
- `scripts/data_prep/evaluate_predictions.py`

## 6.3 验收标准
用同一批 train/val/test 做实验，满足以下任一目标即可视为成功：

- Exact Accuracy 提升 ≥ 5 个百分点
- Severe Error 下降 ≥ 5 个百分点
- Parse 稳定且任务指标同步改善
- 训练 loss 与任务指标相关性更一致

---

## 7. 第三优先级：净化监督信号

如果标签噪声分析表明问题显著，则先于扩数据执行。

## 7.1 目标
建立“可信主标签 + 可选辅助字段”的训练集，而不是继续依赖完全自蒸馏输出。

## 7.2 修改方向
### 必做
- `cell_type` 以原始标注/规则校验结果为准
- `ontology_id` 必须来自官方 CL 映射，不允许自造 ID
- 对无法确认 ontology_id 的样本，不作为 ontology 监督样本

### 推荐
- rationale 不再作为第一阶段训练目标
- confidence / review / novelty 暂不参与主训练
- 训练集样本增加 label_quality 字段：`clean / weak / noisy`

## 7.3 代码修改位置建议
- `src/sca/data/ontology_mapping.py`
- `src/sca/data/target_labeling.py`
- `src/sca/data/sft_builder.py`
- `scripts/data_prep/05_make_sft_jsonl.py`
- `scripts/diagnosis/analyze_label_noise.py`

## 7.4 验收标准
- ontology_id 非法样本占比显著下降
- `clean` 样本可单独导出训练
- “只用 clean 子集训练”的试验结果不劣于全量 noisy 数据
- 训练后 severe error 明显下降或 exact 稳定提升

---

## 8. 第四优先级：扩数据与补覆盖

注意：这一步很重要，但应放在诊断和监督净化之后执行。

## 8.1 扩数据目标
- 训练样本：从当前量级提升到 2000+
- 细胞类型：覆盖 300+
- 优先补 lung / kidney / brain
- 降低 zero-shot 比例
- 提高 ontology 映射覆盖率

## 8.2 数据来源优先顺序
1. 修复现有 CELLxGENE 失败数据集处理  
2. 补当前测试相关组织的数据集  
3. 引入高质量专家标注来源（如 Tabula Sapiens / HCA）  
4. 扩展外部 marker KB

## 8.3 推荐代码位置
- `scripts/data_prep/03_clean_and_standardize.py`
- `scripts/data_prep/04_make_marker_examples.py`
- `scripts/data_prep/06_split_and_validate_v2.py`
- `scripts/data_prep/08_build_marker_kb.py`

## 8.4 验收标准
- 训练样本数显著增长
- rare / zero-shot 比例下降
- tissue 分布更加平衡
- 扩数据后在独立测试集上有稳定提升，而不只是 train/val 提升

---

## 9. 第五优先级：改进 KB

只有在诊断表明 KB 理论上有效但当前实现弱时，才投入升级。

## 9.1 修改方向
优先顺序建议：

1. 保留当前 Jaccard，增加 BM25 召回
2. 做 hybrid retrieval（BM25 + Jaccard）
3. 若有条件，再上 dense retrieval
4. 扩外部 KB（CellMarker / PanglaoDB 等）

## 9.2 验收标准
- zero-shot 子集性能明显改善
- rare 子集 severe error 下降
- 相比 no_kb 有稳定正增益
- top-k 检索候选的人工可解释性更好

---

## 10. 第六优先级：模型与训练策略升级

这是后续项，不应抢在前面。

## 10.1 可选方向
- 升级到底模更大的版本
- 调整 LoRA rank
- curriculum learning
- ensemble
- ontology-aware loss

## 10.2 执行前提
只有在以下条件下才建议投入：
- schema 已验证
- 标签已清洗
- 数据已扩充
- KB 是否有效已被证明

## 10.3 验收标准
升级后的收益必须超过简单基线改造收益，否则不视为高 ROI 方案。

---

## 11. 建议的下一轮最小可执行实验包（必须先做）

下一轮先只做以下 6 件事，不要同时做更多：

1. 新增 `scripts/diagnosis/`
2. 完成 `analyze_error_buckets.py`
3. 完成 `ablate_output_schema.py`
4. 完成 `ablate_kb_retrieval.py`
5. 完成 `ablate_ontology_target.py`
6. 生成统一诊断报告

只有这些完成后，再决定是否继续：

- 收缩任务
- 清洗标签
- 扩数据
- 升模型

---

## 12. 最终一句话执行策略

下一轮实验不要再以“继续加模块”为主，而要以：

**先建立诊断层 → 证明确切瓶颈 → 收缩主任务 → 净化监督 → 再扩数据与升级模型**

作为唯一主线。

如果诊断结果显示：
- full_json 是主拖累项，就立刻改成 cell_type-only；
- ontology_id 直接生成拖累明显，就改成后处理；
- KB 当前无稳定增益，就先不要继续在检索上投入太多；
- 标签噪声很高，就优先重做监督数据，而不是先换更大模型。
