"""
Report formatting for capacity calculations.

Output is styled to match SGLang's initialization logs from PR #14071.
"""

from sglcap.calculator import CapacityReport


def _memory_bar(weights_pct: float, kv_pct: float, width: int = 40) -> str:
    """Create a simple ASCII memory bar."""
    weights_chars = max(1, int((weights_pct / 100) * width)) if weights_pct > 0 else 0
    kv_chars = max(1, int((kv_pct / 100) * width)) if kv_pct > 0 else 0

    if weights_chars + kv_chars > width:
        scale = width / (weights_chars + kv_chars)
        weights_chars = max(1, int(weights_chars * scale))
        kv_chars = width - weights_chars

    avail_chars = width - weights_chars - kv_chars

    return f"{'█' * weights_chars}{'▓' * kv_chars}{'░' * avail_chars}"


def format_report(report: CapacityReport, verbose: bool = False) -> str:
    """Format capacity report to match SGLang log output style."""
    r = report
    lines = []

    # Calculate percentages for memory bar
    total = r.gpu_memory_gb
    weights_pct = (r.weight_memory_gb / total) * 100 if total > 0 else 0
    kv_pct = (max(0, r.kv_cache_budget_gb) / total) * 100 if total > 0 else 0

    # Header
    lines.extend([
        "",
        "┌" + "─" * 78 + "┐",
        "│" + " sglcap - SGLang Capacity Calculator".center(78) + "│",
        "└" + "─" * 78 + "┘",
        "",
    ])

    # Model info
    lines.append(f"  Model: {r.model_path}")

    model_details = f"  {r.architecture} | {r.attention_type_detail} | {r.dtype_str}"
    if r.quantization:
        model_details += f" | {r.quantization}"
    lines.append(model_details)

    if r.is_moe:
        lines.append(f"  MoE: {r.num_experts} experts, {r.num_experts_per_tok} active/token")

    # Parallelism and cluster topology
    dp_attn_str = " (DP_attn)" if r.enable_dp_attention else ""
    lines.append(f"  TP={r.tp_size} DP={r.dp_size}{dp_attn_str} | {r.gpu_memory_gb:.0f}GB/GPU ({r.gpu_memory_source})")

    # Cluster info
    if r.num_nodes > 1:
        lines.append(f"  Cluster: {r.total_gpus} GPUs across {r.num_nodes} nodes ({r.gpus_per_node} GPUs/node)")
    else:
        lines.append(f"  Cluster: {r.total_gpus} GPUs on 1 node")

    # Perspective note
    lines.append(f"  ℹ All calculations below are per-Node")

    if r.sliding_window:
        lines.append(f"  ⚠ Sliding window={r.sliding_window:,} (SGLang does not optimize KV for SWA)")

    # Memory visualization
    bar = _memory_bar(weights_pct, kv_pct)
    lines.extend([
        "",
        "  ┌─ GPU Memory " + "─" * 32 + "┐",
        f"  │ [{bar}] │",
        "  └" + "─" * 44 + "┘",
        f"    █ Weights: {r.weight_memory_gb:.1f}GB ({weights_pct:.0f}%)  "
        f"▓ KV: {max(0, r.kv_cache_budget_gb):.1f}GB ({kv_pct:.0f}%)  "
        f"░ Overhead",
    ])

    # =========================================================================
    # model_runner.py logs - KV cache profiling (STEP BY STEP)
    # =========================================================================
    lines.extend([
        "",
        "─" * 80,
        "  [model_runner] KV cache profiling",
        "─" * 80,
    ])

    # Step 1: Cell size calculation
    if r.attention_type == "MLA":
        layout_detail = f"kv_lora={r.kv_lora_rank}+rope={r.qk_rope_head_dim}"
    else:
        layout_detail = f"{r.num_kv_heads}kv×{r.head_dim}d"

    lines.extend([
        "",
        f"  Step 1: Cell size ({r.attention_type})",
        f"          {r.cell_size_formula}",
        f"          = {r.cell_size_breakdown}",
        f"          = {r.cell_size:,} bytes/token ({r.cell_size_kb:.2f} KB/token)",
    ])

    # Step 2: Memory budget
    lines.extend([
        "",
        f"  Step 2: Memory budget",
        f"          gpu_memory          = {r.gpu_memory_gb:.1f} GB",
        f"          weight_memory       = {r.weight_memory_gb:.2f} GB/GPU ({r.weight_memory_source})",
        f"          available_memory    = gpu - weights = {r.gpu_memory_gb:.1f} - {r.weight_memory_gb:.2f} = {r.available_memory_gb:.2f} GB",
        f"          mem_fraction_static = {r.mem_fraction_static:.3f}",
        f"          overhead            = gpu × (1 - frac) = {r.gpu_memory_gb:.1f} × {1-r.mem_fraction_static:.3f} = {r.gpu_memory_gb * (1-r.mem_fraction_static):.2f} GB",
        f"          rest_memory (KV)    = available - overhead = {max(0, r.kv_cache_budget_gb):.2f} GB",
    ])

    # Step 3: Max tokens calculation
    lines.extend([
        "",
        f"  Step 3: Max tokens",
        f"          profiled = rest_memory / cell_size",
        f"                   = {max(0, r.kv_cache_budget_gb):.2f} GB / {r.cell_size:,} bytes",
        f"                   = {r.max_total_num_tokens_profiled:,} tokens",
    ])

    # Show user override if applicable
    if r.max_total_num_tokens_arg is not None:
        lines.extend([
            f"          --max-total-tokens = {r.max_total_num_tokens_arg:,}",
            f"          effective = min(arg, profiled) = {r.max_total_num_tokens:,} ({r.max_total_num_tokens_source})",
        ])

    lines.append(f"          page_aligned = (tokens // {r.page_size}) × {r.page_size} = {r.max_total_num_tokens_padded:,}")

    # KV cache summary line (matching PR format)
    lines.extend([
        "",
        f"  → KV cache: {r.attention_type}({layout_detail}) layers={r.num_hidden_layers} "
        f"dtype={r.kv_cache_dtype_str} {r.cell_size_kb:.2f}KB/tok",
        f"    mem={r.available_memory_gb:.1f}/{r.gpu_memory_gb:.0f}GB rest={max(0, r.kv_cache_budget_gb):.1f}GB "
        f"max_tokens={r.max_total_num_tokens_padded:,}",
    ])

    # =========================================================================
    # model_runner.py logs - req_to_token_pool creation
    # =========================================================================
    lines.extend([
        "",
        "─" * 80,
        "  [model_runner] req_to_token_pool",
        "─" * 80,
        "",
        f"  heuristic = profiled_max_tokens / context_len × 512",
        f"            = {r.max_total_num_tokens_profiled:,} / {r.context_len:,} × 512",
        f"            = {r.req_to_token_pool_heuristic:,}",
        f"  slots     = clamp(heuristic, 2048, 4096) = {r.req_to_token_pool_size:,}",
    ])

    pool_mem_gb = r.req_to_token_pool_size * r.context_len * 4 / 1e9
    lines.append(f"  memory    = slots × context_len × 4 bytes = {pool_mem_gb:.2f} GB")

    # =========================================================================
    # tp_worker.py logs - max_running_requests
    # =========================================================================
    token_limit = r.max_total_num_tokens_padded // 2

    lines.extend([
        "",
        "─" * 80,
        "  [tp_worker] max_running_requests",
        "─" * 80,
        "",
    ])

    if r.max_running_requests_source == "arg":
        lines.append(f"  --max-running-requests = {r.max_running_requests:,} (user specified)")
    else:
        lines.extend([
            f"  token_limit = max_tokens // 2 = {r.max_total_num_tokens_padded:,} // 2 = {token_limit:,}",
            f"  effective   = min(token_limit, pool_slots)",
            f"              = min({token_limit:,}, {r.req_to_token_pool_size:,})",
            f"              = {r.max_running_requests:,}",
        ])

    # =========================================================================
    # tp_worker.py logs - max_req_len
    # =========================================================================
    context_limit = r.context_len - 1
    memory_limit = r.max_total_num_tokens_padded - 1

    lines.extend([
        "",
        "─" * 80,
        "  [tp_worker] max_req_len",
        "─" * 80,
        "",
    ])

    if context_limit <= memory_limit:
        lines.append(f"  limited by context_len: {r.context_len:,} - 1 = {r.max_req_len:,}")
    else:
        lines.append(f"  limited by memory: {r.max_total_num_tokens_padded:,} - 1 = {r.max_req_len:,}")

    lines.extend([
        f"  max_req_input_len  = {r.max_req_input_len:,}",
        f"  max_prefill_tokens = {r.max_prefill_tokens:,}",
    ])

    if r.max_queued_requests is not None:
        lines.append(f"  max_queued_requests = {r.max_queued_requests:,}")

    # Notes (if any)
    if r.notes:
        lines.extend([
            "",
            "─" * 80,
        ])
        for note in r.notes:
            lines.append(f"  ⚠ {note}")

    lines.append("")

    return "\n".join(lines)


__all__ = ["format_report"]
