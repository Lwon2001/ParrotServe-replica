import argparse
import time
from dataclasses import dataclass

import torch
from vllm.model_executor.layers.attention import PagedAttention
from vllm.config import ModelConfig

@dataclass
class InputMetadata:
    """Metadata for input sequences."""
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    num_valid_tokens: int = 0
    block_tables: torch.Tensor = None
    context_lens: torch.Tensor = None
    max_context_len: int = 0
    slot_mapping: torch.Tensor = None
    attn_bias: list = None

def get_torch_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

def benchmark_kernel(
    batch_size: int,
    num_queries_per_kv: int,
    max_context_len: int,
    block_size: int,
    num_blocks: int,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    seed: int,
):
    # Create dummy input tensors.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    query = torch.randn(
        batch_size, num_heads, head_size, dtype=dtype, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size // x,
        block_size,
        x,
        dtype=dtype,
        device="cuda",
    )
    value_cache = torch.randn(
        num_blocks, num_kv_heads, head_size, block_size, dtype=dtype, device="cuda"
    )

    # Create dummy KV cache slots.
    block_mapping = torch.randint(
        0, num_blocks, size=(batch_size, max_context_len // block_size), device="cuda"
    )
    context_lens = torch.randint(
        block_size,
        max_context_len + 1,
        size=(batch_size,),
        dtype=torch.int,
        device="cuda",
    )
    # prepare block tables
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        size=(batch_size, max_num_blocks_per_seq),
        dtype=torch.int,
        device="cuda",
    )
    # Set attention parameters.
    attn = PagedAttention(
        num_heads, head_size, scale=head_size**-0.5, num_kv_heads=num_kv_heads
    )

    # Create input metadata
    input_metadata = InputMetadata(
        num_valid_tokens=batch_size,
        num_generation_tokens=batch_size,
        block_tables=block_tables,
        context_lens=context_lens,
        max_context_len=max_context_len,
    )

    # Run benchmark.
    num_iterations = 10
    warmup_iterations = 3

    start_time = time.perf_counter()
    for _ in range(num_iterations + warmup_iterations):
        output = attn.forward(
            query,
            key_cache,
            value_cache,
            key_cache,
            value_cache,
            input_metadata,
            None,  # cache_event
        )
    end_time = time.perf_counter()
    latency = (end_time - start_time) / num_iterations

    # Print benchmark results.
    print(
        f"Batch size: {batch_size}, "
        f"Num queries per kv: {num_queries_per_kv}, "
        f"Max context len: {max_context_len}, "
        f"Block size: {block_size}, "
        f"Num blocks: {num_blocks}, "
        f"Head size: {head_size}, "
        f"Num heads: {num_heads}, "
        f"Num KV heads: {num_kv_heads}, "
        f"Data type: {dtype}, "
        f"Seed: {seed} "
        f"=> Latency: {latency * 1000:.3f} ms"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the attention kernel without using Parrot."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size"
    )
    parser.add_argument(
        "--num-queries-per-kv",
        type=int,
        default=1,
        help="Number of queries per key/value head.",
    )
    parser.add_argument(
        "--max-context-len", type=int, default=4096, help="maximum context length"
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="token block size"
    )
    parser.add_argument(
        "--num-blocks", type=int, default=256, help="number of blocks"
    )
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 128, 256],
        default=128,
        help="head size",
    )
    parser.add_argument(
        "--num-heads", type=int, default=None, help="number of heads"
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=None, help="number of KV heads"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="data type",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    if args.num_heads is None:
      args.num_heads = 8

    if args.num_kv_heads is None:
        args.num_kv_heads = args.num_heads

    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark_kernel(
        batch_size=args.batch_size,
        num_queries_per_kv=args.num_queries_per_kv,
        max_context_len=args.max_context_len,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        head_size=args.head_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dtype=get_torch_dtype(args.dtype),
        seed=args.seed,
    )