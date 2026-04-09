"""
train_qwen3_hf_trl.py — Qwen3 单细胞注释模型 SFT 训练脚本

训练方法概述
-----------
SFT（Supervised Fine-Tuning，监督微调）：
  给模型喂大量 (prompt, answer) 对，让它学会对特定格式的输入产生对应的输出。
  训练目标是最小化模型在 answer 部分的交叉熵损失（Cross-Entropy Loss），
  即让模型尽可能准确地"预测"出正确答案的每一个 token。

LoRA（Low-Rank Adaptation）：
  不修改原始模型的全部权重（参数量太大），而是在每个线性层旁边
  新增一对低秩矩阵 A(d×r) 和 B(r×k)，只训练这些小矩阵。
  等效更新：ΔW = A × B，通常 r << d，可训练参数量仅为原来的 0.1-1%。

QLoRA（Quantized LoRA，8B 模式）：
  在 LoRA 的基础上，将基础模型的权重压缩为 4-bit（NF4 格式），
  把显存从 16GB（8B bf16）降至约 4GB，同时 LoRA 部分依然以 bf16 计算。

用法
----
  # 使用默认配置（scripts/train/train_config.yaml）
  CUDA_VISIBLE_DEVICES=0 python -u scripts/train/train_qwen3_hf_trl.py

  # 指定配置文件
  CUDA_VISIBLE_DEVICES=0 python -u scripts/train/train_qwen3_hf_trl.py \\
      --config scripts/train/train_config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# 告诉 PyTorch 使用可扩展内存段，减少显存碎片（对 GPU 内存接近上限时很有用）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 项目根目录（此文件在 scripts/train/ 下，上两级即为根目录）
_PROJECT_DIR = Path(__file__).resolve().parents[2]

# 模型路径映射：model_size → 本地路径
_MODEL_PATHS = {
    "4B": str(_PROJECT_DIR / "my_models/Qwen/Qwen3-4B"),
    "8B": str(_PROJECT_DIR / "my_models/Qwen/Qwen3-8B"),
}


# =============================================================================
# 配置加载
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    读取 YAML 格式的训练配置文件，返回配置字典。

    Args:
        config_path: train_config.yaml 的路径

    Returns:
        包含所有超参数的字典
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    将配置中的相对路径转换为绝对路径（基于项目根目录）。
    同时自动生成带时间戳的 output_dir（若配置中留空）。

    Args:
        cfg: 原始配置字典

    Returns:
        路径已解析为绝对路径的配置字典
    """
    cfg = dict(cfg)  # 浅拷贝，避免修改原字典

    # 将 train_file / val_file 解析为绝对路径
    cfg["train_file"] = str(_PROJECT_DIR / cfg["train_file"])
    cfg["val_file"] = str(_PROJECT_DIR / cfg["val_file"])

    # output_dir 留空时自动生成
    if not cfg.get("output_dir"):
        size_tag = cfg["model_size"].lower()  # "4b" 或 "8b"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["output_dir"] = str(
            _PROJECT_DIR / f"output/qwen3_{size_tag}_sc_sft_hf_trl_v2_{timestamp}"
        )
    else:
        cfg["output_dir"] = str(_PROJECT_DIR / cfg["output_dir"])

    return cfg


def check_file(path: str, name: str) -> None:
    """验证文件是否存在，不存在则立即报错退出。"""
    if not Path(path).exists():
        print(f"[ERROR] {name} 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# 数据处理
# =============================================================================

@dataclass
class ExampleStats:
    """记录数据集预处理时各类样本的数量统计。"""
    total: int = 0
    kept: int = 0
    skipped_empty: int = 0
    skipped_no_assistant: int = 0
    skipped_bad_format: int = 0


def _normalize_message_content(content: Any) -> str:
    """
    将消息内容统一转换为字符串。

    Qwen3 的 content 字段有时是字符串，有时是 list（多模态格式），
    此函数统一处理为纯文本字符串。

    Args:
        content: 原始 content，可能是 str、list 或其他类型

    Returns:
        规范化后的字符串
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多模态格式：[{"type": "text", "text": "..."}, ...]
        # 只取 text 类型的内容
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts)
    return str(content)


def convert_messages_to_prompt_completion(
    records: List[Dict[str, Any]],
    tokenizer,
    source_name: str,
    enable_thinking: bool = False,
) -> Dataset:
    """
    将原始 JSONL 记录转换为 SFTTrainer 所需的 (prompt, completion) 数据集。

    SFTTrainer 的训练格式：
      - prompt：包含 system 和 user 消息的对话上文（用 chat_template 格式化）
      - completion：模型需要输出的 assistant 回复

    训练时，loss 只计算在 completion 部分，prompt 部分不参与梯度更新
    （这是 SFT 的关键——我们只让模型"学"如何回答，不学输入本身）。

    Args:
        records: JSONL 加载后的字典列表，每条有 "messages" 字段
        tokenizer: 用于 apply_chat_template 格式化对话
        source_name: "train" 或 "val"，用于打印统计信息
        enable_thinking: 是否启用 Qwen3 的 thinking 模式

    Returns:
        HuggingFace Dataset，包含 prompt/completion/source/example_id 字段
    """
    stats = ExampleStats(total=len(records))
    rows: List[Dict[str, str]] = []

    for idx, rec in enumerate(records):
        messages = rec.get("messages")

        # 过滤格式异常的记录
        if not isinstance(messages, list) or len(messages) == 0:
            stats.skipped_bad_format += 1
            continue

        # 将每条消息的 content 统一为字符串
        normalized_messages = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = _normalize_message_content(m.get("content"))
            if not role:
                continue
            normalized_messages.append({"role": role, "content": content})

        if not normalized_messages:
            stats.skipped_bad_format += 1
            continue

        # 找到最后一条 assistant 消息（作为训练目标）
        # 注意：一次对话可能有多轮，我们取最后一条 assistant 作为 completion
        last_assistant_idx = None
        for i in range(len(normalized_messages) - 1, -1, -1):
            if normalized_messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            stats.skipped_no_assistant += 1
            continue

        # prompt = assistant 回复之前的所有消息（system + user）
        # completion = assistant 的回复内容
        prompt_messages = normalized_messages[:last_assistant_idx]
        assistant_message = normalized_messages[last_assistant_idx]
        completion = assistant_message["content"].strip()

        if not prompt_messages or not completion:
            stats.skipped_empty += 1
            continue

        # 用 tokenizer 的 chat_template 将 messages 列表格式化为模型输入字符串
        # 例如：<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        # add_generation_prompt=True 会在末尾追加 <|im_start|>assistant\n，提示模型开始生成
        try:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,         # 返回字符串，不做 tokenization
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # 旧版 tokenizer 不支持 enable_thinking 参数时的降级处理
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt = prompt.strip()
        if not prompt:
            stats.skipped_empty += 1
            continue

        rows.append({
            "prompt": prompt,
            "completion": completion,
            "source": source_name,
            "example_id": str(idx),
        })
        stats.kept += 1

    # 打印数据集统计
    print(f"\n===== {source_name} dataset stats =====")
    print(f"total={stats.total}")
    print(f"kept={stats.kept}")
    print(f"skipped_empty={stats.skipped_empty}")
    print(f"skipped_no_assistant={stats.skipped_no_assistant}")
    print(f"skipped_bad_format={stats.skipped_bad_format}")
    print("===================================\n")

    if not rows:
        raise ValueError(f"没有从 {source_name} 中产出任何有效样本，请检查数据格式")

    return Dataset.from_list(rows)


def preview_dataset(dataset: Dataset, n: int = 2) -> None:
    """打印数据集前 n 条样本的 prompt 和 completion 预览，用于快速验证数据格式。"""
    print("===== Dataset preview =====")
    for i in range(min(n, len(dataset))):
        row = dataset[i]
        prompt_preview = row["prompt"][:400].replace("\n", "\\n")
        completion_preview = row["completion"][:200].replace("\n", "\\n")
        print(f"[sample {i}] prompt: {prompt_preview}")
        print(f"[sample {i}] completion: {completion_preview}")
        print("---")
    print("===========================\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件，返回字典列表。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# =============================================================================
# 模型加载
# =============================================================================

def load_model_and_tokenizer(cfg: Dict[str, Any]):
    """
    根据配置加载 tokenizer 和 base model。

    4B 模式（标准 LoRA）：
      - 以 bfloat16 精度加载完整模型权重
      - 显存约 8GB（权重）+ 训练开销 = 共约 20GB

    8B 模式（QLoRA）：
      - 使用 bitsandbytes 将权重量化为 4-bit NF4 格式加载
      - 显存约 4GB（量化权重）+ 训练开销 = 共约 10-12GB
      - 计算时自动反量化为 bfloat16，精度损失极小

    Args:
        cfg: 完整配置字典

    Returns:
        (model, tokenizer) 元组
    """
    model_size = cfg["model_size"]
    model_path = _MODEL_PATHS.get(model_size)
    if model_path is None:
        raise ValueError(f"不支持的 model_size: {model_size}，可选: {list(_MODEL_PATHS)}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=cfg["local_files_only"],
    )
    # 如果 tokenizer 没有 pad_token，用 eos_token 代替
    # 这是 Qwen 系列模型的常见设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({model_size}) from {model_path} ...")

    if model_size == "8B":
        # ====== QLoRA 加载方式 ======
        # 需要 bitsandbytes 库支持 4-bit 量化
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("8B QLoRA 需要 bitsandbytes 库：pip install bitsandbytes")

        # 配置 4-bit 量化参数
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # NF4 量化：专为正态分布权重设计，精度优于普通 fp4
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            # 实际矩阵乘法时用 bf16 精度计算（量化只用于存储，不用于计算）
            bnb_4bit_compute_dtype=torch.bfloat16,
            # 双重量化：对量化参数本身再做一次量化，额外节省约 0.37 bits/param
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            # device_map="auto" 是 bitsandbytes 量化加载的必须参数
            device_map="auto",
            trust_remote_code=cfg["trust_remote_code"],
            local_files_only=cfg["local_files_only"],
        )

        # QLoRA 的必要步骤：为量化模型的梯度计算做准备
        # - 将 LayerNorm 转为 fp32 以保证稳定性
        # - 启用 gradient checkpointing（若配置开启）
        # - 使输入 embedding 能够接收梯度
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg["gradient_checkpointing"],
            )
        except ImportError:
            pass  # 新版 PEFT 可能已自动处理，不影响训练

    else:
        # ====== 标准 4B LoRA 加载方式 ======
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # bfloat16：比 float32 省一半显存，精度接近，RTX 4090 原生支持
            torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float16,
            trust_remote_code=cfg["trust_remote_code"],
            local_files_only=cfg["local_files_only"],
        )

    # 关闭 KV-cache：训练时不需要缓存，开启反而占显存
    model.config.use_cache = False

    return model, tokenizer


# =============================================================================
# LoRA 配置
# =============================================================================

def build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    """
    构建 LoRA 适配器配置。

    LoRA 原理：
      对于原始权重矩阵 W (d×k)，冻结 W 不更新，
      新增两个小矩阵 A(d×r) 和 B(r×k)，r << d。
      前向计算变为：output = x @ W + x @ A @ B × (alpha/r)
      只训练 A 和 B，参数量从 d×k 降至 r×(d+k)。

    Args:
        cfg: 配置字典

    Returns:
        LoraConfig 对象
    """
    target_modules = cfg["lora_target_modules"]
    # 如果配置是逗号分隔的字符串（如 "q_proj,v_proj"），转为列表
    # "all-linear" 保持字符串形式（PEFT 特殊值，会自动找所有线性层）
    if isinstance(target_modules, str) and "," in target_modules:
        target_modules = [m.strip() for m in target_modules.split(",")]

    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=target_modules,
        bias="none",           # 不对 bias 参数做 LoRA（通常不需要）
        task_type="CAUSAL_LM", # 因果语言模型任务（自回归生成）
    )


# =============================================================================
# 训练参数配置
# =============================================================================

def build_training_args(cfg: Dict[str, Any]) -> SFTConfig:
    """
    构建 SFTTrainer 的训练超参数对象。

    SFTConfig 继承自 TrainingArguments，是 TRL 库针对 SFT 的封装。

    Args:
        cfg: 配置字典

    Returns:
        SFTConfig 对象
    """
    return SFTConfig(
        output_dir=cfg["output_dir"],

        # --- 训练规模 ---
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

        # --- 学习率调度 ---
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],

        # --- 日志与保存 ---
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],

        # 每 eval_steps 步评估，按验证 loss 保存最佳 checkpoint
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        metric_for_best_model="eval_loss",  # 用验证集 loss 选最优模型
        greater_is_better=False,            # loss 越低越好
        load_best_model_at_end=True,        # 训练结束时自动加载最佳 checkpoint

        # --- 精度 ---
        bf16=cfg["use_bf16"],
        fp16=cfg["use_fp16"],

        # --- 显存优化 ---
        # gradient_checkpointing：以重计算代替存储激活值，节省 30-50% 显存
        # 8B QLoRA 模式下已通过 prepare_model_for_kbit_training 启用，此处不重复开启
        gradient_checkpointing=(cfg["gradient_checkpointing"] and cfg["model_size"] == "4B"),
        # use_reentrant=False：更现代的 checkpoint 实现，支持更多模型结构
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # --- SFT 特有参数 ---
        max_length=cfg["max_length"],       # 超过此长度的样本截断
        packing=False,                      # 不将多个短样本拼接为一条（保证每条独立）

        # --- 其他 ---
        dataloader_num_workers=cfg["dataloader_num_workers"],
        seed=cfg["seed"],
        report_to=[],                       # 不上报到 wandb/tensorboard（可改为 ["tensorboard"]）
        dataset_kwargs={"skip_prepare_dataset": False},
    )


# =============================================================================
# 主训练流程
# =============================================================================

def main() -> None:
    """
    主函数：按顺序执行完整的 SFT 训练流程。

    流程：
      1. 解析 CLI 参数 → 加载配置文件
      2. 验证数据文件存在
      3. 设置随机种子（保证可复现）
      4. 加载 tokenizer 和 base model
      5. 将 JSONL 数据转换为 HuggingFace Dataset
      6. 构建 LoRA 配置和训练参数
      7. 初始化 SFTTrainer 并启动训练
      8. 保存最终 adapter 权重和 tokenizer
    """
    # ------ 步骤 1：解析 CLI 参数 ------
    parser = argparse.ArgumentParser(description="Qwen3 SCA SFT 训练脚本")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "train_config.yaml"),
        help="训练配置文件路径（默认与脚本同目录的 train_config.yaml）",
    )
    args = parser.parse_args()

    # 加载并解析配置
    print(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg)

    model_size = cfg["model_size"]
    print(f"Model size: {model_size}")
    print(f"Output dir: {cfg['output_dir']}")

    # ------ 步骤 2：验证文件 ------
    check_file(cfg["train_file"], "train_file")
    check_file(cfg["val_file"], "val_file")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(str(_PROJECT_DIR / "data/meta"), exist_ok=True)

    # ------ 步骤 3：设置随机种子 ------
    # 保证每次运行的权重初始化、数据采样顺序一致，实验可复现
    set_seed(cfg["seed"])

    # ------ 步骤 4：加载模型 ------
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ------ 步骤 5：加载并处理数据集 ------
    print("Loading datasets...")
    train_records = load_jsonl(cfg["train_file"])
    val_records = load_jsonl(cfg["val_file"])

    # 将 messages 列表格式转为 (prompt, completion) 格式
    train_dataset = convert_messages_to_prompt_completion(
        train_records, tokenizer, "train", cfg["enable_thinking"]
    )
    val_dataset = convert_messages_to_prompt_completion(
        val_records, tokenizer, "val", cfg["enable_thinking"]
    )

    # 打印样本预览，确认格式正确
    preview_dataset(train_dataset)

    # ------ 步骤 6：构建 LoRA 和训练参数 ------
    # LoRA config：定义在哪些层插入低秩适配器，以及适配器的规模
    peft_config = build_lora_config(cfg)

    # 打印 LoRA 参数量统计（可选，帮助理解训练规模）
    print(f"LoRA config: r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, "
          f"dropout={cfg['lora_dropout']}, targets={cfg['lora_target_modules']}")

    # 训练参数：学习率、batch size、保存策略等
    training_args = build_training_args(cfg)

    # ------ 步骤 7：初始化 SFTTrainer 并训练 ------
    # SFTTrainer 会自动：
    #   - 将 peft_config 应用到 model（插入 LoRA 层，冻结原始权重）
    #   - 处理 prompt/completion 的 loss masking（只对 completion 计算 loss）
    #   - 按 eval_steps 频率做验证并保存 checkpoint
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    train_result = trainer.train()

    # ------ 步骤 8：保存结果 ------
    # save_model 保存的是 LoRA adapter 权重（几十 MB），不是完整模型（几 GB）
    # 推理时需要同时加载 base model + adapter
    print("Saving final adapter...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # 保存训练指标到 all_results.json
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 释放 GPU 显存（如果后续还有操作的话）
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Training finished.")
    print(f"Final output dir: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
