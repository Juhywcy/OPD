#!/usr/bin/env python3
"""Measure gradient agreement between correctness and length objectives.

This is a diagnostic, not a training program: it samples one batch, replays
the sampled tokens through the policy, forms two REINFORCE losses, and reports
the cosine similarity of their gradients.  It never calls ``optimizer.step``.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local/HF model path.")
    parser.add_argument("--data", required=True, help="VERL-format parquet file.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts.")
    parser.add_argument("--num-responses", type=int, default=1, help="Samples per prompt.")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--advantage-mode",
        choices=("raw", "centered", "zscore"),
        default="raw",
        help="Transform each scalar reward before REINFORCE (default: raw).",
    )
    parser.add_argument(
        "--loss-aggregation",
        choices=("sequence-mean", "token-mean"),
        default="sequence-mean",
        help="Sequence REINFORCE loss or response-token mean loss.",
    )
    parser.add_argument(
        "--gradient-parameter-regex",
        default=".*",
        help="Only include parameters whose names match this regex. Use e.g. 'lm_head' for a cheap probe.",
    )
    parser.add_argument(
        "--gradient-chunk-size",
        type=int,
        default=64,
        help="Number of parameter tensors per autograd call; lower reduces peak gradient memory.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output", default=None, help="Optional JSON result path.")
    return parser


def _advantages(rewards, mode: str):
    """Transform a vector of scalar rewards into the requested advantages."""
    if mode == "raw":
        return rewards
    centered = rewards - rewards.mean()
    if mode == "centered":
        return centered
    return centered / centered.std(unbiased=False).clamp_min(1e-6)


def _format_prompt(tokenizer, prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def _ground_truth(row: dict[str, Any]) -> Any:
    reward_model = row.get("reward_model") or {}
    return reward_model.get("ground_truth", row.get("ground_truth"))


def _is_correct(response: str, ground_truth: Any, data_source: str) -> bool:
    # This is the same high-recall boxed-answer evaluator used by the OPD
    # training recipe. Its return dictionary contains ``acc``.
    from verl.utils.reward_score.ttrl_math import reward_func

    result = reward_func(data_source, response, ground_truth)
    return bool(result["acc"] if isinstance(result, dict) else result > 0)


def _response_mask(response_ids, eos_token_id: Optional[int], pad_token_id: int):
    """Return a mask that includes EOS and excludes trailing generation pads."""
    import torch

    mask = torch.ones_like(response_ids, dtype=torch.bool)
    if eos_token_id is not None:
        eos_hits = response_ids.eq(eos_token_id)
        first_eos = eos_hits.float().cumsum(dim=-1).le(1)
        # Retain positions through the first EOS, then discard the suffix.
        mask = first_eos
    if eos_token_id != pad_token_id:
        mask &= response_ids.ne(pad_token_id)
    return mask


def _reinforce_loss(action_log_probs, response_mask, advantages, aggregation: str):
    weighted_logp = action_log_probs * response_mask.to(action_log_probs.dtype)
    if aggregation == "sequence-mean":
        return -(advantages * weighted_logp.sum(dim=-1)).mean()
    return -(advantages.unsqueeze(-1) * weighted_logp).sum() / response_mask.sum().clamp_min(1)


def _gradient_cosine(loss_a, loss_b, parameters, chunk_size: int):
    """Compute an exact global cosine without retaining two full gradient copies."""
    import torch

    dot = torch.zeros((), device=loss_a.device, dtype=torch.float64)
    norm_a_sq = torch.zeros_like(dot)
    norm_b_sq = torch.zeros_like(dot)
    used_numel = 0
    chunks = [parameters[i : i + chunk_size] for i in range(0, len(parameters), chunk_size)]
    if not chunks:
        raise ValueError("No trainable parameters matched --gradient-parameter-regex.")

    for chunk_index, chunk in enumerate(chunks):
        keep_graph = True
        grads_a = torch.autograd.grad(loss_a, chunk, retain_graph=keep_graph, allow_unused=True)
        grads_b = torch.autograd.grad(
            loss_b,
            chunk,
            retain_graph=chunk_index != len(chunks) - 1,
            allow_unused=True,
        )
        for grad_a, grad_b in zip(grads_a, grads_b, strict=True):
            if grad_a is None or grad_b is None:
                continue
            a = grad_a.detach().to(dtype=torch.float64)
            b = grad_b.detach().to(dtype=torch.float64)
            dot += (a * b).sum()
            norm_a_sq += (a * a).sum()
            norm_b_sq += (b * b).sum()
            used_numel += a.numel()

    norm_a = norm_a_sq.sqrt()
    norm_b = norm_b_sq.sqrt()
    if bool((norm_a == 0).item()) or bool((norm_b == 0).item()):
        return None, float(norm_a.cpu()), float(norm_b.cpu()), used_numel
    cosine = dot / (norm_a * norm_b)
    return float(cosine.cpu()), float(norm_a.cpu()), float(norm_b.cpu()), used_numel


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_responses <= 0 or args.gradient_chunk_size <= 0:
        raise ValueError("batch-size, num-responses, and gradient-chunk-size must be positive")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires one CUDA GPU.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.train()

    dataset = load_dataset("parquet", data_files=str(Path(args.data)), split="train")
    if len(dataset) < args.batch_size:
        raise ValueError(f"Dataset has {len(dataset)} rows, smaller than batch-size={args.batch_size}")
    rows = [dataset[i] for i in range(args.batch_size)]
    prompt_texts = [_format_prompt(tokenizer, row["prompt"]) for row in rows]
    encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    ).to(device)
    prompt_width = encoded.input_ids.shape[1]

    model.eval()
    with torch.inference_mode():
        sequences = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_responses,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # ``inference_mode`` returns inference tensors, which cannot be saved by
    # autograd when they later index ``gather`` in the log-prob replay.
    # Materialize a normal tensor after leaving that context.
    sequences = sequences.clone()
    model.train()

    response_ids = sequences[:, prompt_width:]
    response_mask = _response_mask(response_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    response_lengths = response_mask.sum(dim=-1)
    decoded_responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    repeated_rows = [row for row in rows for _ in range(args.num_responses)]
    correctness = torch.tensor(
        [
            float(_is_correct(text, _ground_truth(row), str(row.get("data_source", "math_dapo"))))
            for text, row in zip(decoded_responses, repeated_rows, strict=True)
        ],
        device=device,
        dtype=torch.float32,
    )
    length_rewards = -response_lengths.to(dtype=torch.float32)
    accuracy_advantages = _advantages(correctness, args.advantage_mode).to(dtype=dtype)
    length_advantages = _advantages(length_rewards, args.advantage_mode).to(dtype=dtype)

    repeated_input_ids = encoded.input_ids.repeat_interleave(args.num_responses, dim=0)
    repeated_attention = encoded.attention_mask.repeat_interleave(args.num_responses, dim=0)
    full_input_ids = torch.cat((repeated_input_ids, response_ids), dim=-1)
    full_attention = torch.cat((repeated_attention, response_mask.to(repeated_attention.dtype)), dim=-1)
    # Reconstruct the left-padding-aware positions used during generation.
    position_ids = full_attention.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(full_attention.eq(0), 0)
    logits = model(
        input_ids=full_input_ids,
        attention_mask=full_attention,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    response_logits = logits[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1], :]
    action_log_probs = response_logits.log_softmax(dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    accuracy_loss = _reinforce_loss(action_log_probs, response_mask, accuracy_advantages, args.loss_aggregation)
    length_loss = _reinforce_loss(action_log_probs, response_mask, length_advantages, args.loss_aggregation)
    pattern = re.compile(args.gradient_parameter_regex)
    parameters = [param for name, param in model.named_parameters() if param.requires_grad and pattern.search(name)]
    cosine, accuracy_norm, length_norm, parameter_numel = _gradient_cosine(
        accuracy_loss, length_loss, parameters, args.gradient_chunk_size
    )

    result = {
        "gradient_cosine_similarity": cosine,
        "cosine_defined": cosine is not None,
        "accuracy_gradient_norm": accuracy_norm,
        "length_gradient_norm": length_norm,
        "gradient_parameter_numel": parameter_numel,
        "batch_prompts": args.batch_size,
        "responses": len(decoded_responses),
        "num_correct": int(correctness.sum().item()),
        "accuracy_reward_mean": float(correctness.mean().item()),
        "length_reward_mean": float(length_rewards.mean().item()),
        "response_length_mean": float(response_lengths.float().mean().item()),
        "advantage_mode": args.advantage_mode,
        "loss_aggregation": args.loss_aggregation,
        "gradient_parameter_regex": args.gradient_parameter_regex,
        "per_response": [
            {
                "correctness_reward": int(correct),
                "length_reward": -int(length),
                "response_length": int(length),
            }
            for correct, length in zip(correctness.cpu().tolist(), response_lengths.cpu().tolist(), strict=True)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
