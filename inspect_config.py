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

fields = [
    "model_type",
    "architectures",
    "hidden_size",          # token vector hidden size
    "intermediate_size",    # attention block forward layer size
    "num_hidden_layers",    # num of transformer blocks 
    "num_attention_heads",  # num of query head
    "num_key_value_heads",  # num of K/V head
    "head_dim",             # each head query dim [hidden_size // num_attetntion_heads]
    "hidden_act",           # the active function used in MLP
    "vocab_size",           # as name it is
    "tie_word_embeddings",  # embedding vocab weight is same as the logits layer weight 
    "rope_theta",           # rope freq
    "rope_scaling",         
    "rms_norm_eps",         # as name it is 
    "attention_bias",       # T/F if attention have bias?
    "torch_dtype",          # params type
    "dtype",                # same as top
]

for field in fields:
    value = getattr(model_config, field, None)
    print(f"{field}: {value}")




