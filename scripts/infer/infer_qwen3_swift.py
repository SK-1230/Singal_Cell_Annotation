import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 可改参数 =====
BASE_MODEL_PATH = "/data/projects/shuke/code/singal_cell_annotation/my_models/Qwen/Qwen3-4B"
ADAPTER_PATH = "/data/projects/shuke/code/singal_cell_annotation/output/qwen3_4b_sc_sft_swift_v2/v0-20260326-181619/checkpoint-176"
TEST_FILE = "/data/projects/shuke/code/singal_cell_annotation/data/splits/test_messages_no_think.jsonl"
MAX_NEW_TOKENS = 256


def load_one_example(test_file: str):
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"Test file is empty: {test_file}")

    rec = json.loads(first_line)
    return rec["messages"]


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    messages = load_one_example(TEST_FILE)

    # 推理时不喂 assistant 答案，只保留 system + user
    infer_messages = [m for m in messages if m["role"] != "assistant"]

    text = tokenizer.apply_chat_template(
        infer_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\n===== Prompt =====\n")
    print(text)

    print("\n===== Model Output =====\n")
    print(generated)


if __name__ == "__main__":
    main()


# python infer_qwen3_swift.py 2>&1 | tee data/meta/infer_qwen3_swift.log