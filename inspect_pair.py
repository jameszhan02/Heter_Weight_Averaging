from pathlib import Path
from transformers import AutoConfig

MODEL_A_PATH = Path(
    "./.model/OLMo-2-0425-1B-Instruct"
)

MODEL_B_PATH = Path(
    "./.model/Llama-3.2-1B-Instruct"
)

model_a_config = AutoConfig.from_pretrained(
    MODEL_A_PATH,
    local_files_only=True,
)
model_b_config = AutoConfig.from_pretrained(
    MODEL_B_PATH,
    local_files_only=True,
)





# aiming to diagonse gap between two models
def compare_equal(
    field_name,
    model_a_value,
    model_b_value,
):
    values_match = model_a_value == model_b_value

    if values_match:
        status = "PASS"
    else:
        status = "FAIL"

    print(
        f"[{status}] {field_name}: "
        f"A={model_a_value!r}, "
        f"B={model_b_value!r}"
    )

    return values_match


required_equal_fields = [
    "model_type",
    "vocab_size",
    "hidden_act",
    "rope_theta",
    "rope_scaling",
    "rms_norm_eps",
    "tie_word_embeddings",
]

required_checks_passed = True
print("Required equality checks:")

for field_name in required_equal_fields:
    model_a_value = getattr(
        model_a_config,
        field_name,
        None,
    )
    model_b_value = getattr(
        model_b_config,
        field_name,
        None,
    )
    field_passed = compare_equal(
        field_name,
        model_a_value,
        model_b_value,
    )
    if not field_passed:
        required_checks_passed = False



print()
print("Final result:")
if required_checks_passed:
    print(
        "Direct paper-style merge may be supported."
    )
else:
    print(
        "Direct paper-style merge is not supported."
    )