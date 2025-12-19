"""
Tests for openai/gpt-oss-20b model.

Reference SGLang run:
    python -m sglang.launch_server \
        --model-path openai/gpt-oss-20b \
        --tp-size 4 --dp-size 4 --enable-dp-attention \
        --kv-cache-dtype fp8_e4m3 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 1000000

Key SGLang log lines:
    KV cache: MHA(heads=8, dim=64) layers=24 dtype=torch.float8_e4m3fn 24.00KB/tok
    req_to_token_pool: slots=4096
    max_running_requests: effective=4096
    max_req_len: limited by context_len (131072 - 1 = 131071)
    max_prefill_tokens=16384
"""

import pytest
from tests.conftest import run_sglcap, assert_report_matches, ExpectedValues


class TestGptOss20b:
    """Tests for gpt-oss-20b with DP attention."""

    MODEL_PATH = "openai/gpt-oss-20b"

    def test_cell_size_fp8(self):
        """Test cell size calculation with FP8 KV cache."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            max_total_tokens=1000000,
            attention_backend="triton",  # Required for GptOssForCausalLM in CPU mode
        )

        # Cell size should be 24.00 KB/token
        # Formula: kv_heads × head_dim × layers × 2 × dtype_bytes
        # = 8 × 64 × 24 × 2 × 1 = 24,576 bytes = 24.00 KB
        assert report.cell_size_kb == 24.00

    def test_attention_type(self):
        """Test attention type detection."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            attention_backend="triton",
        )

        # Should be MHA (SGLang doesn't distinguish GQA in logs)
        assert report.attention_type == "MHA"
        # But detail should show GQA since num_kv_heads < num_attention_heads
        assert report.attention_type_detail == "GQA"

    def test_kv_cache_params(self):
        """Test KV cache parameters."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            attention_backend="triton",
        )

        # With DP attention: attention_tp_size = tp_size // dp_size = 4 // 4 = 1
        # So kv_heads_per_tp = 8 // 1 = 8
        assert report.num_kv_heads == 8
        assert report.head_dim == 64
        assert report.num_hidden_layers == 24

    def test_req_pool_slots(self):
        """Test req_to_token_pool slot calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            max_total_tokens=1000000,
            attention_backend="triton",
        )

        # Slots should be clamped to 4096
        # heuristic = profiled_max_tokens / context_len * 512
        # Even though we set max_total_tokens=1M, pool uses profiled value
        assert report.req_to_token_pool_size == 4096

    def test_max_running_requests(self):
        """Test max_running_requests calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            max_total_tokens=1000000,
            attention_backend="triton",
        )

        # effective = min(token_limit, pool_slots)
        # token_limit = max_total_tokens // 2 = 500,000
        # pool_slots = 4096
        # So effective = 4096
        assert report.max_running_requests == 4096

    def test_max_req_len(self):
        """Test max_req_len calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            max_total_tokens=1000000,
            attention_backend="triton",
        )

        # limited by context_len: 131072 - 1 = 131071
        assert report.max_req_len == 131071
        assert report.context_len == 131072

    def test_full_match(self):
        """Test all values match SGLang output."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            dp_size=4,
            enable_dp_attention=True,
            kv_cache_dtype="fp8_e4m3",
            mem_fraction_static=0.85,
            max_total_tokens=1000000,
            attention_backend="triton",
        )

        expected = ExpectedValues(
            cell_size_kb=24.00,
            attention_type="MHA",
            num_kv_heads=8,
            head_dim=64,
            num_layers=24,
            context_len=131072,
            req_pool_slots=4096,
            max_running_requests=4096,
            max_req_len=131071,
            max_prefill_tokens=16384,
        )

        assert_report_matches(report, expected)
