# scripts/diagnosis/ — SCA 诊断模块

## 目标

在继续扩数据或升级模型之前，先用诊断实验**明确当前的主瓶颈**，避免盲目投入。

## 脚本列表

| 脚本 | 需要 GPU | 功能 |
|------|---------|------|
| `diag_config.py` | — | 共用路径配置 |
| `analyze_error_buckets.py` | 否 | 错误分桶分析（seen/tissue/hard/marker quality 等）|
| `ablate_output_schema.py` | 是（推理侧）| 验证训练目标过宽是否是主瓶颈 |
| `ablate_kb_retrieval.py` | 是（推理侧）| 验证 KB 是增益还是噪声 |
| `ablate_ontology_target.py` | 否 | 验证 ontology_id 是否应改为后处理 |
| `analyze_label_noise.py` | 否 | 统计自蒸馏标签的噪声程度 |
| `analyze_data_coverage.py` | 否 | 量化数据覆盖缺口和扩充优先级 |
| `generate_diagnosis_report.py` | 否 | 整合所有结果，生成统一诊断报告 |
| `run_all_diagnosis.py` | 可选 | 一键运行所有诊断脚本 |

## 快速开始

```bash
cd /data/projects/shuke/code/singal_cell_annotation

# Step 1: 运行所有无需 GPU 的诊断（~1-2 分钟）
python scripts/diagnosis/run_all_diagnosis.py

# Step 2: 查看统一诊断报告
cat output/diagnosis/final_report/diagnosis_report.md

# Step 3（可选）: 运行需要 GPU 的消融（~30-60 分钟）
CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/run_all_diagnosis.py --with-gpu
```

## 单独运行某个脚本

```bash
# 错误分桶分析（最重要，先跑这个）
python scripts/diagnosis/analyze_error_buckets.py

# 标签噪声分析
python scripts/diagnosis/analyze_label_noise.py

# 数据覆盖分析
python scripts/diagnosis/analyze_data_coverage.py

# Ontology 消融（无需 GPU，基于已有推理结果）
python scripts/diagnosis/ablate_ontology_target.py

# Schema 消融（需 GPU，运行两次推理后对比）
CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/ablate_output_schema.py

# Schema 消融：只生成简化训练数据，不运行推理
python scripts/diagnosis/ablate_output_schema.py --prep-train-data

# KB 消融（需 GPU）
CUDA_VISIBLE_DEVICES=0 python scripts/diagnosis/ablate_kb_retrieval.py --modes no_kb

# 生成统一报告（整合已有结果）
python scripts/diagnosis/generate_diagnosis_report.py
```

## 诊断结果说明

所有输出保存在 `output/diagnosis/` 下：

```
output/diagnosis/
  buckets/
    bucket_report.md         ← 分桶分析报告
    bucket_summary.csv       ← 可用 Excel 查看的数据
    enriched_results.jsonl   ← 带分桶标签的完整结果
  ontology_ablation/
    ontology_comparison.md
    rule_mapped.jsonl
  label_noise/
    label_noise_report.md
    label_quality.csv
    top20_noisy_samples.json ← 人工复核用
  data_coverage/
    data_coverage_report.md
    tissue_coverage.csv
    celltype_coverage.csv
    zero_shot_types.txt      ← 必须补充的细胞类型
    rare_types.txt
  kb_ablation/               （仅 --with-gpu 后存在）
    kb_comparison.md
    no_kb/results.jsonl
    oracle_hint/results.jsonl
  schema_ablation/           （仅 --with-gpu 后存在）
    schema_comparison.md
    cell_type_only/results.jsonl
    cell_type_parent/results.jsonl
  final_report/
    diagnosis_report.md      ← 最终统一报告（核心问答 + 建议）
```

## 5 个核心问题

诊断完成后，`diagnosis_report.md` 会自动回答：

1. 当前第一瓶颈是不是任务定义过宽（full_json schema）？
2. 当前 KB 是否形成稳定收益？
3. ontology_id 是否应改为后处理？
4. zero-shot / rare / tissue shift 哪个影响最大？
5. 当前自蒸馏标签噪声是否严重到必须先处理？

## 依赖

```
共享库：src/sca/diagnosis/（metrics.py, bucket_analysis.py, label_quality.py, report_utils.py）
配置：scripts/diagnosis/diag_config.py
推理：scripts/infer/infer_qwen3_kb_retrieval.py
```
