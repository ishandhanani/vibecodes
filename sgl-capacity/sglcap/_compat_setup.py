"""
SGLang compatibility setup - standalone patching module.

This module contains ONLY the patching logic with NO imports from sglcap
to avoid circular import issues. It must be importable before any other
sglcap module.
"""

import sys
import os
import importlib.util
import importlib.abc
from types import ModuleType
from unittest.mock import MagicMock

_setup_done = False


class _MockModule(ModuleType):
    """A mock module that returns MagicMock for missing attributes."""

    def __init__(self, name: str, with_spec: bool = False, is_package: bool = True):
        super().__init__(name)
        if with_spec:
            self.__spec__ = importlib.util.spec_from_loader(name, None)
            if is_package:
                # Make it look like a package so submodule imports work
                self.__path__ = []
                self.__package__ = name
        # Provide common dunder attributes
        self.__version__ = "0.0.0"  # Fake version for compatibility checks

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return MagicMock()


class _TritonRuntimeMock(_MockModule):
    """Special mock for triton.runtime with driver that raises on .active."""

    def __init__(self):
        super().__init__("triton.runtime", with_spec=True)

        class _DriverMock:
            @property
            def active(self):
                raise RuntimeError("No triton driver available")

        self.driver = _DriverMock()


class _TritonMetaPathFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that intercepts all triton imports."""

    def __init__(self):
        self._modules = {}

        # Pre-create the triton module with runtime already attached
        self._runtime_mock = _TritonRuntimeMock()
        self._triton_root = _MockModule("triton", with_spec=True)
        self._triton_root.runtime = self._runtime_mock

        self._modules["triton"] = self._triton_root
        self._modules["triton.runtime"] = self._runtime_mock

    def find_spec(self, fullname, path, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        name = spec.name
        if name in self._modules:
            return self._modules[name]

        mod = _MockModule(name, with_spec=True)
        self._modules[name] = mod
        return mod

    def exec_module(self, module):
        # Wire up parent-child relationships
        parts = module.__name__.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            # Ensure parent exists
            if parent_name not in self._modules:
                self._modules[parent_name] = _MockModule(parent_name, with_spec=True)
            setattr(self._modules[parent_name], parts[-1], module)

        # Also register in sys.modules
        sys.modules[module.__name__] = module


class _SglKernelMetaPathFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that intercepts all sgl_kernel imports."""

    def __init__(self):
        self._modules = {}

    def find_spec(self, fullname, path, target=None):
        if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        name = spec.name
        if name in self._modules:
            return self._modules[name]

        mod = _MockModule(name, with_spec=True)
        self._modules[name] = mod
        return mod

    def exec_module(self, module):
        # Wire up parent-child relationships
        parts = module.__name__.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            if parent_name not in self._modules:
                self._modules[parent_name] = _MockModule(parent_name, with_spec=True)
            setattr(self._modules[parent_name], parts[-1], module)

        # Also register in sys.modules
        sys.modules[module.__name__] = module


def _patch_torch_library_for_sgl_kernel():
    """
    Patch torch.library.register_fake to ignore sgl_kernel operators.

    SGLang's fp8_kernel.py uses @torch.library.register_fake decorators
    which fail because sgl_kernel operators don't actually exist.
    """
    import torch.library

    original_register_fake = torch.library.register_fake

    def patched_register_fake(op_name, *args, **kwargs):
        # If the op is from sgl_kernel, return a no-op decorator
        if isinstance(op_name, str) and "sgl_kernel" in op_name:
            def noop_decorator(fn):
                return fn
            return noop_decorator
        # Otherwise use the original
        return original_register_fake(op_name, *args, **kwargs)

    torch.library.register_fake = patched_register_fake


def setup_sglang_compat():
    """Set up mocks for SGLang CPU-only usage. Safe to call multiple times."""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    # Set CPU engine mode
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

    # Install triton meta path finder FIRST (before any imports can happen)
    if not any(isinstance(f, _TritonMetaPathFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _TritonMetaPathFinder())

    # Mock vllm entirely - it's not needed for ModelConfig calculations
    if "vllm" not in sys.modules:
        vllm_mock = _MockModule("vllm", with_spec=True)
        sys.modules["vllm"] = vllm_mock

        # Mock common vllm submodules that sglang might import
        vllm_submodules = [
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.activation",
            "vllm.config",
            "vllm.distributed",
            "vllm.utils",
        ]
        for submod in vllm_submodules:
            sys.modules[submod] = _MockModule(submod, with_spec=True)

    # Mock sgl_kernel for CPU-only usage using meta path finder
    # This catches ALL sgl_kernel.* imports dynamically
    if not any(isinstance(f, _SglKernelMetaPathFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _SglKernelMetaPathFinder())

        # Mock torch.library.register_fake for sgl_kernel ops
        # The fp8_kernel.py file tries to register fake implementations
        # We need to intercept these registrations
        _patch_torch_library_for_sgl_kernel()

    # DON'T mock flashinfer - let importlib.util.find_spec return None naturally
    # so is_flashinfer_available() returns False


# ============================================================================
# GPU Memory Mocking
# ============================================================================

# Global storage for mocked GPU memory values
_mocked_gpu_memory_gb = None
_mocked_available_gpu_memory_gb = None


def set_mock_gpu_memory(total_gb: float, available_gb: float = None):
    """
    Set mocked GPU memory values.

    Args:
        total_gb: Total GPU memory in GB
        available_gb: Available GPU memory in GB (defaults to total_gb - small overhead)
    """
    global _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb
    _mocked_gpu_memory_gb = total_gb
    _mocked_available_gpu_memory_gb = available_gb if available_gb is not None else total_gb


def clear_mock_gpu_memory():
    """Clear mocked GPU memory values."""
    global _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb
    _mocked_gpu_memory_gb = None
    _mocked_available_gpu_memory_gb = None


def get_mocked_gpu_memory():
    """Get current mocked GPU memory values."""
    return _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb


_patched = False

def patch_gpu_memory_functions():
    """
    Patch SGLang's GPU memory functions to use our mocked values.

    This must be called AFTER sglang modules are imported.
    Patches both the source module and any modules that imported it.
    """
    global _patched
    if _patched:
        return
    _patched = True

    try:
        from sglang.srt.utils import common as sglang_common
        from sglang.srt.model_executor import model_runner as sglang_model_runner

        # Create patched function
        def patched_get_available_gpu_memory(device, gpu_id, distributed=False, empty_cache=True, cpu_group=None):
            global _mocked_available_gpu_memory_gb
            if _mocked_available_gpu_memory_gb is not None:
                return _mocked_available_gpu_memory_gb
            # For CPU mode without mock, return system memory / numa nodes
            import psutil
            total_free = psutil.virtual_memory().available
            return total_free / (1 << 30)

        # Patch in the source module
        sglang_common.get_available_gpu_memory = patched_get_available_gpu_memory

        # Patch in model_runner where it's imported
        if hasattr(sglang_model_runner, 'get_available_gpu_memory'):
            sglang_model_runner.get_available_gpu_memory = patched_get_available_gpu_memory

    except ImportError as e:
        print(f"Warning: Could not patch GPU memory functions: {e}")


class MockGPUMemory:
    """
    Context manager for mocking GPU memory.

    Usage:
        with MockGPUMemory(total_gb=80, available_gb=72):
            # Code that uses GPU memory functions
            pass
    """

    def __init__(self, total_gb: float, available_gb: float = None):
        self.total_gb = total_gb
        self.available_gb = available_gb if available_gb is not None else total_gb
        self._old_total = None
        self._old_available = None

    def __enter__(self):
        global _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb
        self._old_total = _mocked_gpu_memory_gb
        self._old_available = _mocked_available_gpu_memory_gb
        set_mock_gpu_memory(self.total_gb, self.available_gb)
        patch_gpu_memory_functions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb
        _mocked_gpu_memory_gb = self._old_total
        _mocked_available_gpu_memory_gb = self._old_available
        return False


# ============================================================================
# Distributed State Mocking
# ============================================================================

# Global storage for mocked distributed state
_mocked_attention_tp_size = None
_mocked_world_group = None
_mocked_tp_size = None
_mocked_dp_size = None
_mocked_enable_dp_attention = None


class MockGroupCoordinator:
    """
    Mock GroupCoordinator for single-process capacity calculation.

    Provides the minimal interface needed by SGLang functions like
    get_available_gpu_memory() which check world_size and cpu_group.
    """

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = 0
        self.rank_in_group = rank
        self.ranks = list(range(world_size))
        self.cpu_group = None  # Not needed for capacity calculation
        self.device_group = None


def compute_attention_tp_size(tp_size: int, dp_size: int, enable_dp_attention: bool) -> int:
    """
    Compute attention TP size from parallelism settings.

    This matches SGLang's logic in compute_dp_attention_world_info().
    """
    if enable_dp_attention and dp_size > 1:
        return tp_size // dp_size
    return tp_size


def set_mock_distributed_state(
    tp_size: int = 1,
    dp_size: int = 1,
    enable_dp_attention: bool = False,
):
    """
    Set mocked distributed state for capacity calculation.

    Computes attention_tp_size from the parallelism settings, matching
    SGLang's actual logic.

    Args:
        tp_size: Tensor parallel size
        dp_size: Data parallel size
        enable_dp_attention: Whether DP attention is enabled
    """
    global _mocked_attention_tp_size, _mocked_world_group
    global _mocked_tp_size, _mocked_dp_size, _mocked_enable_dp_attention

    _mocked_tp_size = tp_size
    _mocked_dp_size = dp_size
    _mocked_enable_dp_attention = enable_dp_attention
    _mocked_attention_tp_size = compute_attention_tp_size(tp_size, dp_size, enable_dp_attention)
    _mocked_world_group = MockGroupCoordinator(world_size=1)

    # Also update the sglang module globals directly
    _update_sglang_globals()


def _update_sglang_globals():
    """Update SGLang module globals with mocked values."""
    global _mocked_attention_tp_size

    if _mocked_attention_tp_size is None:
        return

    try:
        from sglang.srt.layers import dp_attention as sglang_dp_attention
        sglang_dp_attention._ATTN_TP_SIZE = _mocked_attention_tp_size
    except ImportError:
        pass


def clear_mock_distributed_state():
    """Clear mocked distributed state."""
    global _mocked_attention_tp_size, _mocked_world_group
    global _mocked_tp_size, _mocked_dp_size, _mocked_enable_dp_attention
    _mocked_attention_tp_size = None
    _mocked_world_group = None
    _mocked_tp_size = None
    _mocked_dp_size = None
    _mocked_enable_dp_attention = None


def get_mock_attention_tp_size():
    """Get mocked attention TP size."""
    return _mocked_attention_tp_size


def get_mock_world_group():
    """Get mocked world group."""
    return _mocked_world_group


_distributed_patched = False


def patch_distributed_functions():
    """
    Patch SGLang's distributed functions to use our mocked values.

    This allows capacity calculation without actual torch.distributed initialization.
    Must be called AFTER sglang modules are imported.
    """
    global _distributed_patched

    # Always update the globals even if already patched (values may have changed)
    try:
        from sglang.srt.layers import dp_attention as sglang_dp_attention
        if _mocked_attention_tp_size is not None:
            sglang_dp_attention._ATTN_TP_SIZE = _mocked_attention_tp_size
    except ImportError:
        pass

    try:
        from sglang.srt.distributed import parallel_state as sglang_parallel_state
        if _mocked_world_group is not None:
            # Set the _WORLD global directly so get_world_group() works
            sglang_parallel_state._WORLD = _mocked_world_group
    except ImportError:
        pass

    if _distributed_patched:
        return
    _distributed_patched = True

    try:
        # Patch dp_attention module
        from sglang.srt.layers import dp_attention as sglang_dp_attention

        def patched_get_attention_tp_size():
            if _mocked_attention_tp_size is not None:
                return _mocked_attention_tp_size
            # Fallback: return the global if set
            if sglang_dp_attention._ATTN_TP_SIZE is not None:
                return sglang_dp_attention._ATTN_TP_SIZE
            raise RuntimeError("attention tp size not initialized and no mock set")

        sglang_dp_attention.get_attention_tp_size = patched_get_attention_tp_size

    except ImportError as e:
        print(f"Warning: Could not patch dp_attention functions: {e}")

    try:
        # Patch parallel_state module
        from sglang.srt.distributed import parallel_state as sglang_parallel_state

        def patched_get_world_group():
            if _mocked_world_group is not None:
                return _mocked_world_group
            if sglang_parallel_state._WORLD is not None:
                return sglang_parallel_state._WORLD
            raise RuntimeError("world group not initialized and no mock set")

        sglang_parallel_state.get_world_group = patched_get_world_group

    except ImportError as e:
        print(f"Warning: Could not patch parallel_state functions: {e}")


class MockDistributedState:
    """
    Context manager for mocking distributed state.

    Usage:
        with MockDistributedState(tp_size=4, dp_size=4, enable_dp_attention=True):
            # Code that uses get_attention_tp_size(), get_world_group()
            # attention_tp_size will be 4 // 4 = 1
            pass
    """

    def __init__(
        self,
        tp_size: int = 1,
        dp_size: int = 1,
        enable_dp_attention: bool = False,
    ):
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.enable_dp_attention = enable_dp_attention
        self._old_attention_tp_size = None
        self._old_world_group = None

    def __enter__(self):
        global _mocked_attention_tp_size, _mocked_world_group
        self._old_attention_tp_size = _mocked_attention_tp_size
        self._old_world_group = _mocked_world_group
        set_mock_distributed_state(
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            enable_dp_attention=self.enable_dp_attention,
        )
        patch_distributed_functions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _mocked_attention_tp_size, _mocked_world_group
        _mocked_attention_tp_size = self._old_attention_tp_size
        _mocked_world_group = self._old_world_group
        return False


# ============================================================================
# ModelRunner Mocking - Call actual SGLang functions
# ============================================================================

def _detect_mambaish_config(hf_config):
    """
    Detect Mamba/hybrid model configs from hf_config.

    Returns the config if it's a Mamba-like model, None otherwise.
    """
    try:
        from sglang.srt.model_executor.model_runner import (
            FalconH1Config,
            NemotronHConfig,
            NemotronH_Nano_VL_V2_Config,
            Qwen3NextConfig,
            JetNemotronConfig,
            JetVLMConfig,
        )

        # Mamba2 configs
        if isinstance(hf_config, (FalconH1Config, NemotronHConfig)):
            return hf_config
        if isinstance(hf_config, NemotronH_Nano_VL_V2_Config):
            return hf_config.llm_config

        # Hybrid GDN configs (Qwen3Next, JetNemotron, JetVLM)
        if isinstance(hf_config, (Qwen3NextConfig, JetNemotronConfig, JetVLMConfig)):
            return hf_config

    except ImportError:
        pass

    return None


def _create_mock_spec_algorithm(server_args):
    """
    Create a mock SpeculativeAlgorithm based on server_args.
    """
    try:
        from sglang.srt.dllm.algorithm.base import SpeculativeAlgorithm
        return SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm,
            server_args.speculative_num_draft_tokens,
        )
    except (ImportError, AttributeError):
        # Fallback: create a mock that returns False for all checks
        class MockSpecAlgorithm:
            def is_none(self): return True
            def is_eagle(self): return False
            def is_eagle3(self): return False
            def is_standalone(self): return False
            def is_ngram(self): return False
        return MockSpecAlgorithm()


def create_mock_model_runner(
    model_config,
    server_args,
    kv_cache_dtype,
    gpu_memory_gb: float,
):
    """
    Create a minimal mock ModelRunner that can call profile_max_num_token().

    This avoids the heavy __init__ (model loading, CUDA init, distributed setup)
    while still allowing us to call the actual SGLang calculation functions.

    Supports:
    - Standard transformer models (MHA/GQA/MLA)
    - Mamba/hybrid models (FalconH1, NemotronH, Qwen3Next, etc.)
    - Speculative decoding (EAGLE, standalone, ngram)
    - FP4/FP8 quantization
    - NSA models (DeepSeek V3.2)

    Args:
        model_config: SGLang ModelConfig
        server_args: SGLang ServerArgs
        kv_cache_dtype: torch.dtype for KV cache
        gpu_memory_gb: GPU memory in GB (for mocking)

    Returns:
        A mock object that has profile_max_num_token() method bound to it
    """
    import types
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.model_executor.model_runner import ModelRunner

    class MockModelRunner:
        pass

    mock = MockModelRunner()

    # Ensure model_config has required attributes
    if not hasattr(model_config, 'full_attention_layer_ids'):
        model_config.full_attention_layer_ids = None
    if not hasattr(model_config, 'v_head_dim'):
        model_config.v_head_dim = model_config.head_dim
    if not hasattr(model_config, 'swa_attention_layer_ids'):
        model_config.swa_attention_layer_ids = None

    # Basic attributes
    mock.device = "cpu"
    mock.gpu_id = 0
    mock.model_config = model_config
    mock.server_args = server_args
    mock.kv_cache_dtype = kv_cache_dtype
    mock.mem_fraction_static = server_args.mem_fraction_static
    mock.kv_cache_memory = 0
    mock.pp_size = 1  # No pipeline parallelism in capacity calculation

    # MLA detection
    mock.use_mla_backend = model_config.attention_arch == AttentionArch.MLA

    # Speculative decoding
    mock.spec_algorithm = _create_mock_spec_algorithm(server_args)
    mock.is_draft_worker = getattr(server_args, 'is_draft_worker', False)

    # Mamba/hybrid model detection
    mock.mambaish_config = _detect_mambaish_config(model_config.hf_config)

    # Hybrid SWA detection (models with both full attention and sliding window layers)
    mock.is_hybrid_swa = (
        model_config.full_attention_layer_ids is not None and
        model_config.swa_attention_layer_ids is not None
    )

    # num_effective_layers - depends on model type
    if mock.mambaish_config is not None and hasattr(mock.mambaish_config, 'full_attention_layer_ids'):
        mock.num_effective_layers = len(mock.mambaish_config.full_attention_layer_ids)
    elif model_config.full_attention_layer_ids:
        mock.num_effective_layers = len(model_config.full_attention_layer_ids)
    else:
        mock.num_effective_layers = model_config.num_hidden_layers

    # Set up distributed mocking
    set_mock_distributed_state(
        tp_size=server_args.tp_size,
        dp_size=server_args.dp_size,
        enable_dp_attention=server_args.enable_dp_attention,
    )
    patch_distributed_functions()

    # Set up GPU memory mocking
    set_mock_gpu_memory(gpu_memory_gb, gpu_memory_gb)
    patch_gpu_memory_functions()

    # Bind actual SGLang methods to our mock
    mock.profile_max_num_token = types.MethodType(
        ModelRunner.profile_max_num_token, mock
    )
    mock.handle_max_mamba_cache = types.MethodType(
        ModelRunner.handle_max_mamba_cache, mock
    )

    return mock


def call_profile_max_num_token(
    model_config,
    server_args,
    kv_cache_dtype,
    gpu_memory_gb: float,
) -> int:
    """
    Call SGLang's actual ModelRunner.profile_max_num_token() with mocked environment.

    This is the main entry point for getting max_num_tokens using SGLang's
    actual calculation, not a reimplementation.

    Args:
        model_config: SGLang ModelConfig
        server_args: SGLang ServerArgs
        kv_cache_dtype: torch.dtype for KV cache
        gpu_memory_gb: GPU memory in GB

    Returns:
        max_num_token calculated by SGLang's actual function
    """
    mock_runner = create_mock_model_runner(
        model_config=model_config,
        server_args=server_args,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_gb=gpu_memory_gb,
    )

    # Call the actual SGLang function
    # Note: total_gpu_memory is in GB (same as get_available_gpu_memory returns)
    max_num_token = mock_runner.profile_max_num_token(gpu_memory_gb)

    return max_num_token


# ============================================================================
# GPU Simulation from GPUSpec
# ============================================================================

# Global storage for simulated GPU compute capability
_mocked_compute_capability = None


def setup_gpu_simulation(gpu_memory_gb: float, compute_capability: tuple = None):
    """
    Configure GPU simulation for a specific GPU type.

    Args:
        gpu_memory_gb: GPU memory in GB
        compute_capability: Optional (major, minor) tuple, e.g. (9, 0) for Hopper
    """
    global _mocked_compute_capability
    set_mock_gpu_memory(gpu_memory_gb, gpu_memory_gb * 0.98)  # 98% available
    patch_gpu_memory_functions()
    if compute_capability:
        _mocked_compute_capability = compute_capability


def get_mocked_compute_capability():
    """Get mocked compute capability (major, minor) or None."""
    return _mocked_compute_capability


class GPUSimulation:
    """
    Context manager for simulating a specific GPU type.

    Usage:
        from sglcap.gpu_db import get_gpu_spec

        gpu_spec = get_gpu_spec("gb200")
        with GPUSimulation(gpu_spec):
            # Code runs with GB200 192GB simulation
            pass

    Or with explicit values:
        with GPUSimulation(memory_gb=192, compute_capability=(10, 0)):
            # Code runs with custom GPU simulation
            pass
    """

    def __init__(self, gpu_spec=None, memory_gb: float = None, compute_capability: tuple = None):
        """
        Initialize GPU simulation.

        Args:
            gpu_spec: GPUSpec object (from gpu_db)
            memory_gb: Override memory in GB (or use gpu_spec.memory_gb)
            compute_capability: Override compute capability (or use gpu_spec.compute_capability)
        """
        if gpu_spec is not None:
            self.memory_gb = memory_gb if memory_gb is not None else gpu_spec.memory_gb
            self.compute_capability = compute_capability if compute_capability is not None else gpu_spec.compute_capability
        else:
            self.memory_gb = memory_gb or 80  # Default to H100
            self.compute_capability = compute_capability or (9, 0)

        self._old_total = None
        self._old_available = None
        self._old_compute_capability = None

    def __enter__(self):
        global _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb, _mocked_compute_capability
        self._old_total = _mocked_gpu_memory_gb
        self._old_available = _mocked_available_gpu_memory_gb
        self._old_compute_capability = _mocked_compute_capability

        setup_gpu_simulation(self.memory_gb, self.compute_capability)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _mocked_gpu_memory_gb, _mocked_available_gpu_memory_gb, _mocked_compute_capability
        _mocked_gpu_memory_gb = self._old_total
        _mocked_available_gpu_memory_gb = self._old_available
        _mocked_compute_capability = self._old_compute_capability
        return False
