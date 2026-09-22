from pathlib import Path
from transformers import AutoModelForCausalLM


MODEL_PATH = Path("./.model/Llama-3.2-1B-Instruct")


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model.eval()
target_keywords = [
    "embed_tokens.weight",
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "lm_head.weight",
]

# for name, parameter in model.named_parameters():
#     if any(keyword in name for keyword in target_keywords):
#         print(name)
#         print(f"  shape: {tuple(parameter.shape)}")
#         print(f"  dtype: {parameter.dtype}")
#         print(f"  device: {parameter.device}")
#         print(f"  requires_grad: {parameter.requires_grad}")



for name, parameter in model.named_parameters():
    if (
        name == "model.embed_tokens.weight"
        or name.startswith("model.layers.0.")
        or name == "lm_head.weight"
    ):
        print(name, tuple(parameter.shape), parameter.dtype,
        parameter.device)