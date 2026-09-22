# Heterogeneous Model Merging 学习 Walkthrough

> 这是一份本地中文学习记录，已被 `.gitignore` 精确忽略，不属于正式仓库文档。
>
> 学习方式：学习者亲手编写代码；Agent 作为 TA，负责拆解任务、解释概念、review 和提供递进提示。

## 1. 最终目标

复现论文 2607.18026 的异构模型合并 baseline，并最终完成：

```text
small checkpoint + large checkpoint
                  |
                  +-- Intersection: truncate large into small space
                  |
                  +-- Union: expand small into large space
                  |
                  +-- weighted interpolation
                  |
                  +-- save, reload, generate, and evaluate
```

当前阶段不写合并算法。我们先学习如何安全地读取真实模型的配置和权重结构。

## 2. 当前工作环境

```text
Host OS: Windows
Python environment: WSL
Repository on Windows: D:\git_projects\Heter_Weight_Averaging
Repository in WSL: /mnt/d/git_projects/Heter_Weight_Averaging
GPU: NVIDIA GeForce RTX 3070 Ti, 8 GB VRAM
Environment manager: uv inside WSL
```

原则：

- Python、PyTorch 和 Transformers 命令全部在 WSL 中运行。
- Windows 不需要单独安装 Python。
- 源代码可以保存在当前 `/mnt/d/...` 仓库。
- 大型模型缓存更适合放在 WSL Linux 文件系统中，以减少 `/mnt/d` 的文件访问开销。
- 本地 `.model/` 目录已被 Git 忽略，禁止提交 checkpoint。

## 3. 已建立的论文方法地图

设 small model 参数是 $\theta_s$，large model 参数是 $\theta_t$。

### 3.1 Union

先把 small model 扩展到 large parameter space：

$$
\widetilde\theta_s=\mathcal E(\theta_s).
$$

然后插值：

$$
\theta^{\cup}(\lambda)
=(1-\lambda)\widetilde\theta_s
+\lambda\theta_t.
$$

端点含义：

```text
lambda = 0 -> expanded small model
lambda = 1 -> original large model
```

### 3.2 Intersection

先把 large model 截断到 small parameter space：

$$
\widetilde\theta_t=\mathcal T(\theta_t).
$$

然后插值：

$$
\theta^{\cap}(\mu)
=(1-\mu)\theta_s
+\mu\widetilde\theta_t.
$$

端点含义：

```text
mu = 0 -> original small model
mu = 1 -> truncated large model, not the original large model
```

### 3.3 当前最重要的认识

- Union 不是对所有模型都有效的普通 zero padding。
- Whole-head mapping 假设 source 和 target 的 attention 架构足够兼容。
- `head_dim`、GQA grouping、RoPE 和模块语义不兼容时，应当报告 unsupported，而不是强行映射。
- 参数坐标虽然可以移动，但所有相连层必须使用一致的坐标映射，原计算线路才可能保持。
- 论文描述的是 expansion largely preserves the source function，不是对任意架构严格保证完全等价。

更详细的数学解释见本地笔记：

```text
notes/union_math_zh.md
```

## 4. 已学习的 Python 概念

### 4.1 `transformers`

`transformers` 是 Hugging Face 提供的第三方 Python 库。它提供：

```text
AutoConfig
AutoTokenizer
AutoModelForCausalLM
model implementations
checkpoint loading
generation utilities
```

### 4.2 `AutoConfig`

Config 是模型结构说明书，不是模型权重。

```text
config.json       -> architecture metadata
model code        -> implementation of the architecture
model.safetensors -> trained tensor values
tokenizer files   -> text/token conversion
```

`AutoConfig` 会读取 `config.json` 中的 `model_type`，然后选择对应的具体 config class。

例如：

```text
AutoConfig
    -> reads model_type
    -> selects Olmo2Config, Qwen2Config, LlamaConfig, etc.
```

### 4.3 `Path`

```python
Path("./.model/OLMo-2-0425-1B-Instruct")
```

这是相对路径。它相对于 Python 进程的 current working directory，而不是自动相对于脚本文件。

在仓库根目录执行：

```bash
python inspect_config.py
```

时，它应解析为：

```text
/mnt/d/git_projects/Heter_Weight_Averaging/.model/OLMo-2-0425-1B-Instruct
```

常用检查：

```python
print(Path.cwd())
print(MODEL_PATH.resolve())
print(MODEL_PATH.exists())
print(MODEL_PATH.is_dir())
```

### 4.4 `expanduser()`

`expanduser()` 只负责将路径开头的 `~` 展开为用户 home directory。

```text
~/models/qwen
    ->
/home/<user>/models/qwen
```

明确使用 `./.model/...` 或 `/mnt/d/...` 时不需要 `expanduser()`。

## 5. 当前代码检查点

当前 `inspect_config.py` 已经完成：

- 使用本地相对路径。
- 打印解析后的绝对路径。
- 检查路径是否存在以及是否为目录。
- 使用 `local_files_only=True`，禁止缺少文件时静默联网下载。
- 加载并打印 config 类型和完整内容。

当前代码：

```python
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
```

代码风格上的下一次整理：

- Python 标准库 `pathlib` 放在第三方库 `transformers` 前面。
- 两组 import 之间留一个空行。
- 当前阶段不需要保留未使用的 `MODEL_ID` 注释。

这些是风格建议，不影响当前代码功能。

## 6. 当前任务：读懂 OLMo Config

### 6.1 第一步：打印关键字段

接下来由学习者在 `inspect_config.py` 中加入关键字段检查：

```text
model_type
architectures
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
hidden_act
vocab_size
tie_word_embeddings
rope_theta
rope_scaling
rms_norm_eps
attention_bias
torch_dtype
```

建议学习并使用：

```python
getattr(object, attribute_name, default_value)
```

原因：不同模型的 config 不一定定义完全相同的字段。

### 6.2 第二步：推导 Attention 结构

需要自己计算：

```text
head_dim
queries_per_kv_head
```

在常见配置中：

$$
\text{head\_dim}
=
\frac{\text{hidden\_size}}
{\text{num\_attention\_heads}}.
$$

$$
\text{queries\_per\_kv\_head}
=
\frac{\text{num\_attention\_heads}}
{\text{num\_key\_value\_heads}}.
$$

计算前必须检查能否整除：

```text
hidden_size % num_attention_heads == 0
num_attention_heads % num_key_value_heads == 0
```

不能永远假设所有模型都满足：

```text
head_dim = hidden_size / num_attention_heads
```

如果 config 显式提供 `head_dim`，还需要将显式值与计算值进行比较。

### 6.3 当前验收问题

完成配置检查后，应能回答：

1. `type(model_config)` 是什么？
2. OLMo 有多少 Transformer layers？
3. `hidden_size` 是多少？
4. `intermediate_size` 是多少？
5. 有多少 query heads？
6. 有多少 KV heads？
7. 一个 head 有多少维？
8. 每个 KV head 被多少 query heads 共享？
9. `tie_word_embeddings` 是什么？
10. 使用什么 activation function？
11. 使用什么 normalization 配置？

只有能够解释这些问题后，才进入真实权重加载。

## 7. 下一阶段预告：只读检查真实权重

下一阶段会学习：

```text
AutoModelForCausalLM.from_pretrained
model.eval()
model.named_parameters()
model.state_dict()
parameter name
tensor shape
dtype
device
requires_grad
tied embeddings
```

计划寻找这些模块或 OLMo 中功能对应的模块：

```text
embedding
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
normalization
lm_head
```

这一阶段仍然只读，不修改 tensor。

## 8. 暂时不要做的事

- 不同时加载 small 和 large 两个模型。
- 不加载 14B 或 32B checkpoint。
- 不使用 `.data` 修改 parameter。
- 不对原 tensor 做 in-place 操作。
- 不保存修改后的 checkpoint。
- 不开始训练。
- 不实现完整 `truncate_tensor` 或 `expand_tensor`。
- 不把 Qwen 的参数名和架构假设直接套在 OLMo 上。

OLMo 当前只用于学习真实 config、模型对象和 `state_dict`。论文 baseline 最终仍需要使用经过兼容性检查的目标模型对。

## 9. 学习路线

```text
[Current] Load and understand local config
    |
    v
Load one manageable real checkpoint
    |
    v
Read parameter names, shapes, dtypes, and devices
    |
    v
Understand state_dict and storage sharing
    |
    v
Safely copy and slice one real tensor
    |
    v
Compare the configs of one small/large model pair
    |
    v
Design compatibility checks
    |
    v
Implement Intersection first
    |
    v
Validate, interpolate, save, and reload
    |
    v
Implement Union with architecture-aware mappings
```

## 10. Session Log

### 2026-09-22

Completed:

- Established WSL as the Python execution environment.
- Identified the Windows-to-WSL repository path mapping.
- Learned the difference between config, model code, checkpoint, and tokenizer.
- Learned `AutoConfig.from_pretrained` and `local_files_only=True`.
- Learned relative paths, `Path.resolve()`, and `expanduser()`.
- Created the first local-config loading script.

Current task:

- Print and interpret important OLMo config fields.
- Derive `head_dim` and `queries_per_kv_head`.

Next review input:

- Output of `inspect_config.py`.
- The learner's answers to the eleven config questions.
