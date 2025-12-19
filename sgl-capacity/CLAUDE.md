# sglcap Development Notes

## Environment Setup

Use the test venv and set CPU mode:
```bash
source /home/ubuntu/test/.venv/bin/activate
export SGLANG_USE_CPU_ENGINE=1
```

Run tests:
```bash
pytest tests/ -v
```

## Architecture

sglcap estimates SGLang capacity without requiring GPUs by:

1. **Using ONLY SGLang's functions** - `ServerArgs`, `ModelConfig`, `compute_dp_attention_world_info()` from sglang directly
2. **Mocking GPU detection** - `_compat_setup.py` mocks triton, sgl_kernel, vllm imports
3. **Single source of truth** - All calculations in `capacity.py` use SGLang's `ModelConfig` properties directly

## Key Files

- `sglcap/capacity.py` - Single source of truth for all calculations using SGLang
- `sglcap/_compat_setup.py` - SGLang CPU mode setup and GPU simulation
- `sglcap/calculator.py` - Main entry point, builds CapacityReport
- `sglcap/model_config.py` - ModelArchitecture dataclass for clean field extraction
- `sglcap/gpu_db.py` - GPU specs (memory, compute capability, GPUs per node)

## Key SGLang Integrations

From `capacity.py`:
- `ModelConfig.attention_arch` - MLA vs MHA detection
- `ModelConfig.get_num_kv_heads(tp_size)` - KV head division for TP
- `ModelConfig.kv_lora_rank`, `qk_rope_head_dim` - MLA parameters
- `ModelConfig.head_dim`, `v_head_dim` - Attention dimensions
- `compute_dp_attention_world_info()` - Attention TP size for DP attention

## Flow

```
CLI args → ServerArgs → ModelConfig → capacity.py calculations → CapacityReport
```

## GPU Simulation

Simulate different GPUs using `GPUSimulation`:
```python
from sglcap import GPUSimulation, get_gpu_spec

gpu_spec = get_gpu_spec("gb200")  # 192GB Blackwell
with GPUSimulation(gpu_spec):
    # Code runs with GB200 simulation
    pass
```

Supported GPUs: L40S (48GB), H100 (80GB), H200 (141GB), GB200 (192GB), GB300 (288GB)
