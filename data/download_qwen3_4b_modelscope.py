from pathlib import Path
from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen3-8B"
CACHE_DIR = Path("./my_models")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading model from ModelScope: {MODEL_ID}")
local_model_path = snapshot_download(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
)

print(f"Download finished.")
print(f"Local model path: {local_model_path}")