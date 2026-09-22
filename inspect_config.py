from transformers import AutoConfig
from pathlib import Path
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_PATH = Path("./.model/OLMo-2-0425-1B-Instruct")

print(f"Model path: {MODEL_PATH.resolve()}")
print(f"Path exists: {MODEL_PATH.exists()}")
print(f"Is directory: {MODEL_PATH.is_dir()}")

model_config = AutoConfig.from_pretrained(
    MODEL_PATH, 
    local_files_only=True,
)

print(type(model_config))
print(model_config)