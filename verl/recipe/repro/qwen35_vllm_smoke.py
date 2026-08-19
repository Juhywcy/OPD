"""Minimal text-generation smoke test for a Qwen3.5 vLLM runtime."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    outputs = model.generate(
        ["What is 2+2? Answer briefly."],
        SamplingParams(max_tokens=32, temperature=0),
    )
    print(f"Q35_OUTPUT: {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    main()
