"""
Tests for Qwen/Qwen3-0.6B model.

Reference SGLang run:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-0.6B \
        --tp 4 \
        --mem-fraction-static 0.85 \
        --max-prefill-tokens 2839

Key SGLang log lines:
    KV cache: MHA(heads=2, dim=128) layers=28 dtype=torch.bfloat16 28.00KB/tok
    req_to_token_pool: slots=4096, context_len=40960
    max_running_requests: effective=4096
    max_req_len: limited by context_len (40960 - 1 = 40959)
    max_prefill_tokens=2839
"""

import pytest
from tests.conftest import run_sglcap, assert_report_matches, ExpectedValues


class TestQwen06B:
    """Tests for Qwen3-0.6B model."""

    MODEL_PATH = "Qwen/Qwen3-0.6B"

    def test_cell_size(self):
        """Test cell size calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Cell size should be 28.00 KB/token
        # Formula: kv_heads × head_dim × layers × 2 × dtype_bytes
        # = 2 × 128 × 28 × 2 × 2 = 28,672 bytes = 28.00 KB
        assert report.cell_size_kb == 28.00

    def test_attention_type(self):
        """Test attention type detection."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Should be MHA (matches SGLang log format)
        assert report.attention_type == "MHA"

    def test_kv_cache_params(self):
        """Test KV cache parameters."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Qwen3-0.6B has 8 KV heads total, with TP=4: 8 // 4 = 2 per rank
        assert report.num_kv_heads == 2
        assert report.head_dim == 128
        assert report.num_hidden_layers == 28

    def test_context_length(self):
        """Test context length detection."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Qwen3-0.6B has 40960 context length
        assert report.context_len == 40960

    def test_not_moe(self):
        """Test that model is not detected as MoE."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Qwen3-0.6B is not MoE
        assert report.is_moe == False

    def test_req_pool_slots(self):
        """Test req_to_token_pool slot calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Slots should be clamped to 4096
        assert report.req_to_token_pool_size == 4096

    def test_max_running_requests(self):
        """Test max_running_requests calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Should be clamped by pool_slots
        assert report.max_running_requests == 4096

    def test_max_req_len_context_limited(self):
        """Test max_req_len when limited by context_len."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # limited by context_len: 40960 - 1 = 40959
        assert report.max_req_len == 40959

    def test_custom_max_prefill_tokens(self):
        """Test custom max_prefill_tokens setting."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
            max_prefill_tokens=2839,
        )

        # Should respect the custom value
        assert report.max_prefill_tokens == 2839

    def test_full_match(self):
        """Test all values match SGLang output."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
            max_prefill_tokens=2839,
        )

        expected = ExpectedValues(
            cell_size_kb=28.00,
            attention_type="MHA",
            num_kv_heads=2,
            head_dim=128,
            num_layers=28,
            context_len=40960,
            req_pool_slots=4096,
            max_running_requests=4096,
            max_req_len=40959,
            max_prefill_tokens=2839,
        )

        assert_report_matches(report, expected)
