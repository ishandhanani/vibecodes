# sglcap

Offline capacity calculator for SGLang. Estimates KV cache capacity, max tokens, and request limits without requiring GPUs.

> **Note**: Linux only. Requires SGLang CPU build.

## How It Works

sglcap calls **actual SGLang functions** with a mocked environment:

```
CLI args → ServerArgs → ModelConfig → SGLang's profile_max_num_token() → Capacity Report
```

The key insight is that SGLang's capacity calculations only need:

- Model config (from HuggingFace)
- GPU memory (mocked)
- Distributed state (mocked)

We mock these dependencies so SGLang's real `ModelRunner.profile_max_num_token()` runs without actual GPUs or distributed setup.

## Installation

```bash
# 1. Install SGLang CPU build
make install-sglang

# 2. Activate the venv
source .venv/bin/activate

# 3. Install sglcap
uv pip install -e .
```

## Usage

```bash
# Basic usage
sglcap --model-path meta-llama/Llama-3.1-8B --gpu-type h100

# With parallelism
sglcap --model-path deepseek-ai/DeepSeek-V3 \
    --gpu-type gb200 \
    --tp-size 48 --dp-size 8 --enable-dp-attention \
    --kv-cache-dtype fp8_e4m3
```

## Supported GPUs

| GPU   | Memory | GPUs/Node |
| ----- | ------ | --------- |
| L40S  | 48 GB  | 8         |
| H100  | 80 GB  | 8         |
| H200  | 141 GB | 8         |
| GB200 | 192 GB | 4         |
| GB300 | 288 GB | 4         |

## What's Mocked

| Component         | How                                                                 |
| ----------------- | ------------------------------------------------------------------- |
| GPU memory        | `get_available_gpu_memory()` returns configured value               |
| Distributed state | `get_attention_tp_size()`, `get_world_group()` return mocked values |
| ModelRunner       | Minimal mock with actual `profile_max_num_token()` method bound     |
| Triton/sgl_kernel | Intercepted via meta path finders                                   |

## Output

sglcap outputs a detailed report matching SGLang's log format:

- Cell size calculation (MHA/MLA)
- Memory budget breakdown
- Max tokens calculation
- req_to_token_pool sizing
- max_running_requests limit
- max_req_len limit
