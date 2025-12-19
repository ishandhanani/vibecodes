"""
Tests for silence09/DeepSeek-R1-Small-2layers model (MLA attention).

Reference SGLang run:
    python -m sglang.launch_server \
        --model-path silence09/DeepSeek-R1-Small-2layers \
        --tp 4 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 60000

Key SGLang log lines:
    KV cache: MLA(kv_lora=512, rope=64) layers=2 dtype=torch.bfloat16 2.25KB/tok
    req_to_token_pool: slots=4096
    context_len=163840
    max_prefill_tokens=16384
"""

import pytest
from tests.conftest import run_sglcap, assert_report_matches, ExpectedValues


class TestDeepSeek2Layer:
    """Tests for DeepSeek 2-layer model with MLA attention."""

    MODEL_PATH = "silence09/DeepSeek-R1-Small-2layers"

    def test_cell_size_mla(self):
        """Test cell size calculation for MLA attention."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
            max_total_tokens=60000,
        )

        # Cell size should be 2.25 KB/token
        # Formula: (kv_lora_rank + qk_rope_head_dim) × layers × dtype_bytes
        # = (512 + 64) × 2 × 2 = 2,304 bytes = 2.25 KB
        assert report.cell_size_kb == 2.25

    def test_attention_type_mla(self):
        """Test MLA attention type detection."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Should be MLA
        assert report.attention_type == "MLA"
        assert report.attention_type_detail == "MLA"

    def test_mla_params(self):
        """Test MLA-specific parameters."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # MLA params
        assert report.kv_lora_rank == 512
        assert report.qk_rope_head_dim == 64
        assert report.num_hidden_layers == 2

    def test_context_length(self):
        """Test context length detection."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # DeepSeek R1 has 163840 context length
        assert report.context_len == 163840

    def test_moe_detection(self):
        """Test MoE detection."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
        )

        # Should detect MoE
        assert report.is_moe == True
        assert report.num_experts == 64
        assert report.num_experts_per_tok == 4

    def test_req_pool_slots(self):
        """Test req_to_token_pool slot calculation."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
            max_total_tokens=60000,
        )

        # Slots should be clamped to 4096
        assert report.req_to_token_pool_size == 4096

    def test_max_req_len_memory_limited(self):
        """Test max_req_len when limited by memory (max_total_tokens)."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
            max_total_tokens=60000,
        )

        # limited by memory: 60000 - 1 = 59999
        # (since max_total_tokens < context_len)
        assert report.max_req_len == 59999

    def test_full_match(self):
        """Test all values match SGLang output."""
        report = run_sglcap(
            model_path=self.MODEL_PATH,
            gpu_type="l40s",
            tp_size=4,
            mem_fraction_static=0.85,
            max_total_tokens=60000,
        )

        expected = ExpectedValues(
            cell_size_kb=2.25,
            attention_type="MLA",
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_layers=2,
            context_len=163840,
            req_pool_slots=4096,
            max_running_requests=4096,
            max_req_len=59999,
            max_prefill_tokens=16384,
        )

        assert_report_matches(report, expected)
