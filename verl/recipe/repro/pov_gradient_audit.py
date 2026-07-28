#!/usr/bin/env python3
"""Independent-reference gradient and raw-OPD prefix-window audit.

The program freezes a student checkpoint, independently samples rollout
groups A and B, and never creates an optimizer.  Group A supplies Outcome-PG,
Raw OPD, POV-OPD, and per-window Raw OPD gradients.  Group B supplies the
independent outcome-gradient reference.  Each torchrun rank handles complete
audit batches independently, so eight GPUs run eight audit batches in parallel.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from recipe.repro.pov_audit_utils import bootstrap_mean_ci


METHODS = ("outcome_pg", "raw_opd", "pov_opd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Frozen final POV checkpoint in Hugging Face format.")
    parser.add_argument("--teacher-model", required=True, help="Teacher model path.")
    parser.add_argument("--data", required=True, help="Held-out VERL-format parquet file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-prompts", type=int, default=1, help="Prompts in each audit batch j.")
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--num-responses", type=int, default=4, help="Responses in each of A and B, per prompt.")
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--score-micro-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--outcome-advantage-mode",
        choices=("raw", "centered", "zscore"),
        default="zscore",
        help="Per-prompt outcome advantage transformation; zscore matches GRPO normalization.",
    )
    parser.add_argument(
        "--loss-aggregation",
        choices=("token-mean", "seq-mean-token-sum", "seq-mean-token-mean"),
        default="token-mean",
    )
    parser.add_argument(
        "--gradient-parameter-regex",
        default="lm_head",
        help="Parameter subspace to audit. The default is an lm_head gradient-subspace audit.",
    )
    parser.add_argument("--reference-norm-threshold", type=float, default=1e-10)
    parser.add_argument("--candidate-norm-threshold", type=float, default=1e-10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--prefix-window-size", type=int, default=128)
    parser.add_argument("--prefix-baseline-blocks", type=int, default=2)
    parser.add_argument("--save-responses", action="store_true")
    return parser


def _format_prompt(tokenizer, prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def _ground_truth(row: dict[str, Any]) -> Any:
    reward_model = row.get("reward_model") or {}
    return reward_model.get("ground_truth", row.get("ground_truth"))


def _prompt_id(row: dict[str, Any], dataset_index: int) -> str:
    extra_info = row.get("extra_info") or {}
    for mapping in (row, extra_info):
        for key in ("prompt_id", "id", "index", "uid"):
            value = mapping.get(key) if hasattr(mapping, "get") else None
            if value is not None:
                return str(value)
    return str(dataset_index)


def _is_correct(response: str, row: dict[str, Any]) -> bool:
    from verl.utils.reward_score.ttrl_math import reward_func

    result = reward_func(
        str(row.get("data_source", "math_dapo")),
        response,
        _ground_truth(row),
        row.get("extra_info"),
    )
    if isinstance(result, dict):
        return bool(result.get("acc", result.get("score", 0.0)))
    return bool(float(result) > 0.0)


def _response_mask(response_ids, eos_token_id: Optional[int], pad_token_id: int):
    import torch

    mask = torch.ones_like(response_ids, dtype=torch.bool)
    if eos_token_id is not None:
        # Generated padding is normally EOS for these Qwen checkpoints.  The
        # cumulative count retains the first EOS and rejects subsequent pads.
        mask &= response_ids.eq(eos_token_id).cumsum(dim=-1).le(1)
    if eos_token_id != pad_token_id:
        mask &= response_ids.ne(pad_token_id)
    return mask


def _group_advantages(outcomes, prompts: int, responses: int, mode: str):
    grouped = outcomes.reshape(prompts, responses)
    if mode == "raw":
        return grouped.reshape(-1)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    if mode == "centered":
        return centered.reshape(-1)
    scale = grouped.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (centered / scale).reshape(-1)


def _policy_loss(action_log_probs, mask, advantages, aggregation: str):
    weighted = -action_log_probs * advantages.detach() * mask.to(action_log_probs.dtype)
    if aggregation == "token-mean":
        return weighted.sum() / mask.sum().clamp_min(1)
    sequence_sum = weighted.sum(dim=-1)
    valid_sequence = mask.sum(dim=-1).gt(0)
    if aggregation == "seq-mean-token-sum":
        return sequence_sum[valid_sequence].mean()
    sequence_mean = sequence_sum / mask.sum(dim=-1).clamp_min(1)
    return sequence_mean[valid_sequence].mean()


def _full_inputs(encoded, response_ids, response_mask, repeats: int):
    import torch

    prompt_ids = encoded.input_ids.repeat_interleave(repeats, dim=0)
    prompt_mask = encoded.attention_mask.repeat_interleave(repeats, dim=0)
    full_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    full_mask = torch.cat((prompt_mask, response_mask.to(prompt_mask.dtype)), dim=-1)
    position_ids = full_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(full_mask.eq(0), 0)
    return full_ids, full_mask, position_ids


def _action_log_probs(
    model,
    full_ids,
    full_mask,
    position_ids,
    prompt_width: int,
    response_ids,
    temperature: float = 1.0,
):
    logits = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    response_logits = logits[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1], :]
    if temperature != 1.0:
        response_logits = response_logits / temperature
    return response_logits.log_softmax(dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)


def _teacher_scores(
    teacher,
    full_ids,
    full_mask,
    position_ids,
    prompt_width: int,
    response_ids,
    *,
    temperature: float,
    micro_batch_size: int,
):
    import torch

    log_prob_parts = []
    entropy_parts = []
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, full_ids.shape[0], micro_batch_size):
            end = start + micro_batch_size
            logits = teacher(
                input_ids=full_ids[start:end],
                attention_mask=full_mask[start:end],
                position_ids=position_ids[start:end],
                use_cache=False,
            ).logits[:, prompt_width - 1 : prompt_width - 1 + response_ids.shape[1], :]
            logits = logits.float().div_(temperature)
            log_z = torch.logsumexp(logits, dim=-1)
            selected_logits = logits.gather(-1, response_ids[start:end].unsqueeze(-1)).squeeze(-1)
            log_prob_parts.append((selected_logits - log_z).to(dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=-1)
            entropy_parts.append((log_z - (probabilities * logits).sum(dim=-1)).to(dtype=torch.float32))
            del logits, probabilities
    return torch.cat(log_prob_parts, dim=0), torch.cat(entropy_parts, dim=0)


def _generate_group(args, model, encoded, tokenizer, *, seed: int):
    import torch

    torch.manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        sequences = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_responses,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = encoded.input_ids.shape[1]
    response_ids = sequences[:, prompt_width:].clone()
    response_mask = _response_mask(response_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    return response_ids, response_mask


def _reference_gradients(loss, parameters):
    import torch

    gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    detached = [gradient.detach() if gradient is not None else None for gradient in gradients]
    norm_sq = torch.zeros((), device=loss.device, dtype=torch.float64)
    for gradient in detached:
        if gradient is not None:
            norm_sq += torch.sum(gradient.float() * gradient.float(), dtype=torch.float64)
    return detached, float(norm_sq.sqrt().cpu())


def _gradient_stats(loss, parameters, reference_gradients, reference_norm: float, near_zero_threshold: float):
    import torch

    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    dot = torch.zeros((), device=loss.device, dtype=torch.float64)
    norm_sq = torch.zeros_like(dot)
    for gradient, reference in zip(gradients, reference_gradients, strict=True):
        if gradient is None:
            continue
        gradient_float = gradient.detach().float()
        norm_sq += torch.sum(gradient_float * gradient_float, dtype=torch.float64)
        if reference is not None:
            dot += torch.sum(gradient_float * reference.float(), dtype=torch.float64)
    norm = float(norm_sq.sqrt().cpu())
    dot_value = float(dot.cpu())
    cosine = dot_value / (norm * reference_norm) if norm > 0.0 and reference_norm > 0.0 else None
    return {
        "cosine": cosine,
        "dot": dot_value,
        "gradient_norm": norm,
        "relative_gradient_norm": norm / reference_norm if reference_norm > 0.0 else None,
        "near_zero": norm <= near_zero_threshold,
        "negative_cosine": cosine < 0.0 if cosine is not None else None,
    }


def _pov_advantages(args, raw_opd, response_mask, outcomes, teacher_log_probs, teacher_entropy, group_ids):
    from verl.trainer.ppo.core_algos_pt_oal import compute_outcome_aligned_logit_opd_advantage

    config = {
        "pt_oal": {
            "enabled": True,
            "outcome_validation_enabled": True,
            "prefix_trust_enabled": True,
            "prefix_window_size": args.prefix_window_size,
            "prefix_baseline_blocks": args.prefix_baseline_blocks,
        }
    }
    advantages, _, extras = compute_outcome_aligned_logit_opd_advantage(
        token_level_rewards=raw_opd,
        response_mask=response_mask,
        config=config,
        index=group_ids,
        true_reward_score=outcomes,
        teacher_sampled_log_probs=teacher_log_probs,
        teacher_entropy=teacher_entropy,
        logit_delta_scores=raw_opd,
    )
    return advantages, extras


def _run_batch(args, model, teacher, tokenizer, dataset, batch_index: int, device, parameters):
    import torch

    row_start = args.dataset_offset + batch_index * args.batch_prompts
    rows = [dataset[index] for index in range(row_start, row_start + args.batch_prompts)]
    prompt_ids = [_prompt_id(row, row_start + offset) for offset, row in enumerate(rows)]
    prompt_texts = [_format_prompt(tokenizer, row["prompt"]) for row in rows]
    encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    ).to(device)
    prompt_width = encoded.input_ids.shape[1]

    response_a, mask_a = _generate_group(
        args, model, encoded, tokenizer, seed=args.seed + batch_index * 1009 + 17
    )
    response_b, mask_b = _generate_group(
        args, model, encoded, tokenizer, seed=args.seed + batch_index * 1009 + 7919
    )
    repeated_rows = [row for row in rows for _ in range(args.num_responses)]
    decoded_a = tokenizer.batch_decode(response_a, skip_special_tokens=True)
    decoded_b = tokenizer.batch_decode(response_b, skip_special_tokens=True)
    outcomes_a = torch.tensor(
        [float(_is_correct(text, row)) for text, row in zip(decoded_a, repeated_rows, strict=True)],
        device=device,
        dtype=torch.float32,
    )
    outcomes_b = torch.tensor(
        [float(_is_correct(text, row)) for text, row in zip(decoded_b, repeated_rows, strict=True)],
        device=device,
        dtype=torch.float32,
    )
    outcome_adv_a = _group_advantages(
        outcomes_a, args.batch_prompts, args.num_responses, args.outcome_advantage_mode
    ).unsqueeze(-1).expand_as(mask_a)
    outcome_adv_b = _group_advantages(
        outcomes_b, args.batch_prompts, args.num_responses, args.outcome_advantage_mode
    ).unsqueeze(-1).expand_as(mask_b)

    full_a = _full_inputs(encoded, response_a, mask_a, args.num_responses)
    full_b = _full_inputs(encoded, response_b, mask_b, args.num_responses)
    teacher_logp_a, teacher_entropy_a = _teacher_scores(
        teacher,
        *full_a,
        prompt_width,
        response_a,
        temperature=args.teacher_temperature,
        micro_batch_size=args.score_micro_batch_size,
    )

    model.train()
    logp_b = _action_log_probs(
        model, *full_b, prompt_width, response_b, temperature=args.temperature
    )
    reference_loss = _policy_loss(logp_b, mask_b, outcome_adv_b, args.loss_aggregation)
    reference_gradients, reference_norm = _reference_gradients(reference_loss, parameters)
    reference_valid = reference_norm > args.reference_norm_threshold
    del logp_b, reference_loss
    gc.collect()
    torch.cuda.empty_cache()

    logp_a = _action_log_probs(
        model, *full_a, prompt_width, response_a, temperature=args.temperature
    )
    raw_opd = (teacher_logp_a - logp_a.detach().float()) * mask_a
    repeated_prompt_ids = [prompt_id for prompt_id in prompt_ids for _ in range(args.num_responses)]
    outcome_group_ids = [
        prompt_index
        for prompt_index in range(args.batch_prompts)
        for _ in range(args.num_responses)
    ]
    pov_advantage, pov_extras = _pov_advantages(
        args,
        raw_opd,
        mask_a,
        outcomes_a,
        teacher_logp_a,
        teacher_entropy_a,
        outcome_group_ids,
    )
    candidate_advantages = {
        "outcome_pg": outcome_adv_a,
        "raw_opd": raw_opd,
        "pov_opd": pov_advantage,
    }
    method_records = {}
    for method in METHODS:
        loss = _policy_loss(logp_a, mask_a, candidate_advantages[method], args.loss_aggregation)
        method_records[method] = _gradient_stats(
            loss, parameters, reference_gradients, reference_norm, args.candidate_norm_threshold
        )

    trajectories = []
    valid_lengths = mask_a.sum(dim=-1).tolist()
    horizons = pov_extras["pt_oal_horizon"].long().tolist()
    triggered = pov_extras["pt_oal_triggered"].bool().tolist()
    prefix_weights = pov_extras["pt_oal_prefix_weights"]
    prefix_cusum = pov_extras["pt_oal_cusum"]
    window_support = pov_extras["pt_oal_window_support"]
    for trajectory_index, (prompt_id, outcome, valid_length, horizon, did_trigger) in enumerate(
        zip(repeated_prompt_ids, outcomes_a.tolist(), valid_lengths, horizons, triggered, strict=True)
    ):
        windows = []
        for window_index, start in enumerate(range(0, int(valid_length), args.prefix_window_size)):
            end = min(start + args.prefix_window_size, int(valid_length))
            window_mask = torch.zeros_like(mask_a)
            window_mask[trajectory_index, start:end] = mask_a[trajectory_index, start:end]
            window_loss = _policy_loss(logp_a, window_mask, raw_opd, "token-mean")
            stats = _gradient_stats(
                window_loss, parameters, reference_gradients, reference_norm, args.candidate_norm_threshold
            )
            windows.append(
                {
                    "window_index": window_index,
                    "start": start,
                    "end": end,
                    "normalized_start": start / max(1, int(valid_length)),
                    "cosine": stats["cosine"],
                    "dot": stats["dot"],
                    "gradient_norm": stats["gradient_norm"],
                    "near_zero": stats["near_zero"],
                    "conflict": stats["dot"] < 0.0,
                    "prefix_weight": float(prefix_weights[trajectory_index, start].cpu()),
                    "prefix_cusum": float(prefix_cusum[trajectory_index, start].cpu()),
                    "teacher_support": float(window_support[trajectory_index, start].cpu()),
                }
            )
        record = {
            "trajectory_id": f"b{batch_index}:a{trajectory_index}",
            "prompt_id": prompt_id,
            "outcome_class": int(outcome > 0.5),
            "response_length": int(valid_length),
            "triggered": bool(did_trigger),
            "half_trust_reached": bool(did_trigger),
            "horizon_token": int(horizon),
            "horizon_normalized": int(horizon) / max(1, int(valid_length)),
            "minimum_prefix_weight": min(
                (float(window["prefix_weight"]) for window in windows),
                default=1.0,
            ),
            "windows": windows,
        }
        if args.save_responses:
            record["response"] = decoded_a[trajectory_index]
        trajectories.append(record)

    result = {
        "batch_index": batch_index,
        "dataset_row_start": row_start,
        "prompt_ids": prompt_ids,
        "reference_norm": reference_norm,
        "reference_valid": reference_valid,
        "outcome_a_mean": float(outcomes_a.mean().cpu()),
        "outcome_b_mean": float(outcomes_b.mean().cpu()),
        "methods": method_records,
        "trajectories": trajectories,
    }
    del logp_a, raw_opd, pov_advantage, pov_extras, reference_gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _summarize(args, records: list[dict[str, Any]]) -> dict[str, Any]:
    shared = [record for record in records if bool(record["reference_valid"])]
    methods: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        method_records = [record["methods"][method] for record in shared]
        defined = [record for record in method_records if record["cosine"] is not None]
        methods[method] = {
            "mean_cosine": bootstrap_mean_ci(
                (record["cosine"] for record in defined),
                samples=args.bootstrap_samples,
                seed=args.seed + method_index,
            ),
            "negative_cosine_rate": (
                sum(bool(record["negative_cosine"]) for record in defined) / len(defined) if defined else None
            ),
            "near_zero_candidate_rate": (
                sum(bool(record["near_zero"]) for record in method_records) / len(method_records)
                if method_records
                else None
            ),
            "relative_gradient_norm": bootstrap_mean_ci(
                (record["relative_gradient_norm"] for record in method_records),
                samples=args.bootstrap_samples,
                seed=args.seed + 100 + method_index,
            ),
            "candidate_cosine_valid_batches": len(defined),
            "reference_valid_batches": len(shared),
        }
    return {
        "total_batches": len(records),
        "reference_valid_batches": len(shared),
        "reference_norm_threshold": args.reference_norm_threshold,
        "candidate_norm_threshold": args.candidate_norm_threshold,
        "gradient_parameter_regex": args.gradient_parameter_regex,
        "gradient_scope": "subspace" if args.gradient_parameter_regex != ".*" else "full_model",
        "methods": methods,
    }


def _validate_args(args) -> None:
    positive = {
        "batch-prompts": args.batch_prompts,
        "num-batches": args.num_batches,
        "num-responses": args.num_responses,
        "max-new-tokens": args.max_new_tokens,
        "score-micro-batch-size": args.score_micro_batch_size,
        "prefix-window-size": args.prefix_window_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.temperature <= 0 or args.teacher_temperature <= 0:
        raise ValueError("temperatures must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)

    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("The gradient audit requires CUDA.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if distributed:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pattern = re.compile(args.gradient_parameter_regex)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if pattern.search(name)]
    if not named_parameters:
        sample_names = [name for name, _ in list(model.named_parameters())[-10:]]
        raise ValueError(
            f"No parameters matched {args.gradient_parameter_regex!r}. Last model parameter names: {sample_names}"
        )
    selected_ids = {id(parameter) for _, parameter in named_parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    parameters = [parameter for _, parameter in named_parameters]
    parameter_numel = sum(parameter.numel() for parameter in parameters)
    if rank == 0:
        print(
            f"[pov-audit] gradient scope matched {len(parameters)} tensors / {parameter_numel:,} parameters",
            flush=True,
        )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    teacher.config.use_cache = False
    teacher.requires_grad_(False)
    if model.config.vocab_size != teacher.config.vocab_size:
        raise ValueError(
            "Student and teacher vocab sizes differ; sampled-token OPD requires a shared tokenizer/vocabulary."
        )

    dataset = load_dataset("parquet", data_files=str(Path(args.data)), split="train")
    required_rows = args.dataset_offset + args.batch_prompts * args.num_batches
    if len(dataset) < required_rows:
        raise ValueError(f"Dataset has {len(dataset)} rows, but this run requires at least {required_rows}.")

    rank_dir = shard_dir / f"rank_{rank:03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(rank, args.num_batches, world_size):
        record = _run_batch(args, model, teacher, tokenizer, dataset, batch_index, device, parameters)
        record["gradient_parameter_numel"] = parameter_numel
        (rank_dir / f"batch_{batch_index:06d}.json").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"[pov-audit][rank={rank}] batch={batch_index} "
            f"reference_norm={record['reference_norm']:.6g} valid={record['reference_valid']}",
            flush=True,
        )

    if distributed:
        dist.barrier()
    if rank == 0:
        batch_paths = sorted(shard_dir.glob("rank_*/batch_*.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in batch_paths]
        records.sort(key=lambda record: int(record["batch_index"]))
        if len(records) != args.num_batches:
            raise RuntimeError(f"Expected {args.num_batches} batch records, found {len(records)}.")
        with (output_dir / "batches.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = _summarize(args, records)
        summary["gradient_parameter_numel"] = parameter_numel
        summary["gradient_parameter_names"] = [name for name, _ in named_parameters]
        (output_dir / "gradient_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
