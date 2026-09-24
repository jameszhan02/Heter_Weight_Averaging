[https://arxiv.org/pdf/2607.18026]

-- to fire the averaging

# homo case

```bash
uv run python merge_olmo_to_llama_experimental.py \
  --source-model /path/to/source \
  --target-model /path/to/target \
  --output-dir ./outputs/homogeneous_merge_alpha_0.3 \
  --alpha 0.3 \
  --merge-vocab
```

```bash
uv run python merge_olmo_to_llama_experimental.py \
  --source-model /data/shared_ckpt/opd_teacher \
  --target-model /data/shared_ckpt/Llama-3.2-1B-Instruct \
  --output-dir ./outputs/custom_merge_alpha_0.001 \
  --alpha 0.001
  --overwrite
```
