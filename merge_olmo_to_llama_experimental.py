from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from model_merge import merge_state_dict_experimental_target_anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an experimental OLMo-to-Llama target-anchored merge."
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=Path("./.model/OLMo-2-0425-1B-Instruct"),
        help="Path to the source OLMo checkpoint.",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        default=Path("./.model/Llama-3.2-1B-Instruct"),
        help="Path to the target Llama checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/olmo_to_llama_experimental_alpha_0.001"),
        help="Directory where the merged Hugging Face checkpoint will be saved.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Amount of source weights to inject into the target checkpoint.",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Keep target attention weights unchanged.",
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Keep target MLP weights unchanged.",
    )
    parser.add_argument(
        "--skip-norm",
        action="store_true",
        help="Keep target normalization weights unchanged.",
    )
    parser.add_argument(
        "--merge-vocab",
        action="store_true",
        help="Merge embed_tokens/lm_head when source and target vocab weights have identical shapes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow saving into an existing output directory.",
    )
    return parser.parse_args()


def copy_target_tokenizer_files(target_model_dir: Path, output_dir: Path) -> None:
    tokenizer_file_names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]

    for file_name in tokenizer_file_names:
        source_path = target_model_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / file_name)


def copy_target_config_files(target_model_dir: Path, output_dir: Path) -> None:
    # Transformers versions can serialize the same Llama RoPE config with
    # different field names, for example rope_scaling/rope_theta in older
    # configs versus rope_parameters in newer configs. This project is
    # target-anchored, so keep the target config files byte-for-byte compatible
    # with downstream eval loaders that expect the original schema.
    config_file_names = [
        "config.json",
        "generation_config.json",
    ]

    for file_name in config_file_names:
        source_path = target_model_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / file_name)


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory already exists and is not empty: {args.output_dir}"
        )

    print(f"Source model: {args.source_model.resolve()}")
    print(f"Target model: {args.target_model.resolve()}")
    print(f"Output dir:   {args.output_dir.resolve()}")
    print(f"Alpha:        {args.alpha}")

    source_config = AutoConfig.from_pretrained(
        args.source_model,
        local_files_only=True,
    )
    target_config = AutoConfig.from_pretrained(
        args.target_model,
        local_files_only=True,
    )

    print("Loading source model...")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        local_files_only=True,
    )
    print("Loading target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        local_files_only=True,
    )

    print("Merging state_dict...")
    merged_state = merge_state_dict_experimental_target_anchor(
        source_state=source_model.state_dict(),
        target_state=target_model.state_dict(),
        source_config=source_config,
        target_config=target_config,
        alpha=args.alpha,
        merge_attention=not args.skip_attention,
        merge_mlp=not args.skip_mlp,
        merge_norm=not args.skip_norm,
        merge_vocab=args.merge_vocab,
    )

    print("Validating merged tensors...")
    for name, tensor in merged_state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"merged tensor contains NaN or Inf: {name}")

    print("Loading merged state into target architecture...")
    missing_keys, unexpected_keys = target_model.load_state_dict(
        merged_state,
        strict=False,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"load_state_dict mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
        )

    target_model.tie_weights()
    target_model.eval()

    print("Saving merged model...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )

    print("Copying target config files...")
    copy_target_config_files(args.target_model, args.output_dir)

    print("Copying target tokenizer files...")
    copy_target_tokenizer_files(args.target_model, args.output_dir)

    print("Done.")
    print(f"Merged checkpoint saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
