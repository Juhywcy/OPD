#!/usr/bin/env python3
"""Unit tests for the standalone GPQA validation helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpqa_reward import extract_gpqa_choice, reward_func
from prepare_model_for_eval import ModelPreparationError, resolve_model_path, validate_hf_model
from summarize_gpqa_avg16_best16 import summarize_rows


class GPQARewardTest(unittest.TestCase):
    def test_boxed_letter_and_latex_wrapper(self):
        self.assertEqual(extract_gpqa_choice(r"Reasoning. \boxed{D}"), ("D", "boxed"))
        self.assertEqual(extract_gpqa_choice(r"Reasoning. \boxed{\text{c}}"), ("C", "boxed"))

    def test_last_valid_box_wins(self):
        self.assertEqual(extract_gpqa_choice(r"Try \boxed{A}; final \boxed{B}."), ("B", "boxed"))

    def test_explicit_answer_fallback(self):
        self.assertEqual(extract_gpqa_choice("Therefore, the final answer is (b)."), ("B", "explicit"))

    def test_ordinary_option_discussion_is_not_an_answer(self):
        self.assertEqual(extract_gpqa_choice("Option D is inconsistent with the premise."), ("", "none"))

    def test_reward_fields_are_complete(self):
        result = reward_func(
            data_source="GPQA",
            solution_str=r"Conclusion: \boxed{A}",
            ground_truth="A",
            extra_info={"index": "GPQA-7"},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["format_score"], 1.0)
        self.assertEqual(result["pred"], "A")
        self.assertEqual(result["sample_id"], "GPQA-7")
        self.assertTrue(all(value is not None for value in result.values()))


class GPQASummaryTest(unittest.TestCase):
    def test_run_best_and_question_pass_are_both_reported(self):
        rows = [
            {"sample_id": "q1", "acc": 1.0, "response_tokens": 10},
            {"sample_id": "q1", "acc": 0.0, "response_tokens": 20},
            {"sample_id": "q2", "acc": 1.0, "response_tokens": 30},
            {"sample_id": "q2", "acc": 0.0, "response_tokens": 40},
        ]
        summary = summarize_rows(rows, n_responses=2, expected_questions=2)
        self.assertEqual(summary["run_accuracies"], [1.0, 0.0])
        self.assertEqual(summary["avg_at_n"], 0.5)
        self.assertEqual(summary["best_at_n"], 1.0)
        self.assertEqual(summary["worst_at_n"], 0.0)
        self.assertEqual(summary["pass_at_n"], 1.0)
        self.assertEqual(summary["mean_response_tokens"], 25.0)

    def test_incomplete_question_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly 2 responses"):
            summarize_rows([{"sample_id": "q1", "score": 1.0}], n_responses=2)


class ModelPreparationTest(unittest.TestCase):
    @staticmethod
    def _write_metadata(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    def test_hub_id_passes_through(self):
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.assertEqual(resolve_model_path(model_id, dry_run=True), model_id)

    def test_missing_conventional_local_path_is_not_treated_as_hub_id(self):
        with self.assertRaisesRegex(ModelPreparationError, "local model/checkpoint path does not exist"):
            resolve_model_path("model/DeepSeek-R1-Distill-Qwen-1.5B", dry_run=True)

    def test_complete_local_hf_model_passes_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            self._write_metadata(model_dir)
            (model_dir / "model.safetensors").write_bytes(b"weights")

            valid, _ = validate_hf_model(model_dir)
            self.assertTrue(valid)
            self.assertEqual(resolve_model_path(str(model_dir), dry_run=True), str(model_dir.resolve()))

    def test_actor_step_and_config_only_hf_child_resolve_same_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            actor = Path(temp_dir) / "run" / "global_step_279" / "actor"
            metadata = actor / "huggingface"
            self._write_metadata(metadata)
            actor.mkdir(parents=True, exist_ok=True)
            (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 2}), encoding="utf-8")
            for rank in range(2):
                (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(f"rank-{rank}".encode())

            resolved_paths = {
                Path(resolve_model_path(str(input_path), dry_run=True))
                for input_path in (actor, actor.parent, metadata)
            }
            self.assertEqual(len(resolved_paths), 1)
            resolved = resolved_paths.pop()
            self.assertIn(".eval_cache/fsdp_hf/global_step_279", resolved.as_posix())
            self.assertFalse(resolved.exists(), "dry-run must not create or merge the cache")

    def test_merge_uses_cpu_cache_and_preserves_actor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actor = root / "run" / "global_step_7" / "actor"
            self._write_metadata(actor / "huggingface")
            (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 2}), encoding="utf-8")
            for rank in range(2):
                (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(f"rank-{rank}".encode())
                (actor / f"optim_world_size_2_rank_{rank}.pt").write_bytes(f"optim-{rank}".encode())

            observed: dict[str, object] = {}

            def fake_merge(command, **kwargs):
                observed["command"] = command
                observed["env"] = kwargs["env"]
                target = Path(command[command.index("--target_dir") + 1])
                self._write_metadata(target)
                (target / "model.safetensors").write_bytes(b"merged-weights")

            cache_dir = root / "cache"
            with patch("prepare_model_for_eval.subprocess.run", side_effect=fake_merge) as mock_run:
                resolved = Path(resolve_model_path(str(actor), cache_dir=str(cache_dir)))
                resolved_again = Path(resolve_model_path(str(actor), cache_dir=str(cache_dir)))

            self.assertEqual(resolved, resolved_again)
            self.assertEqual(mock_run.call_count, 1, "a validated fingerprint cache must be reused")
            self.assertTrue((resolved / ".merge_complete.json").is_file())
            self.assertEqual(observed["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(observed["command"][:6], [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
            ])
            for rank in range(2):
                self.assertEqual(
                    (actor / f"model_world_size_2_rank_{rank}.pt").read_bytes(),
                    f"rank-{rank}".encode(),
                )
                self.assertEqual(
                    (actor / f"optim_world_size_2_rank_{rank}.pt").read_bytes(),
                    f"optim-{rank}".encode(),
                )

    def test_force_remerge_does_not_bypass_actor_shards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            actor = Path(temp_dir) / "run" / "global_step_8" / "actor"
            metadata = actor / "huggingface"
            self._write_metadata(metadata)
            (metadata / "model.safetensors").write_bytes(b"checkpoint-hf-weights")
            (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 1}), encoding="utf-8")
            (actor / "model_world_size_1_rank_0.pt").write_bytes(b"rank-0")

            self.assertEqual(resolve_model_path(str(actor), dry_run=True), str(metadata.resolve()))
            forced = Path(resolve_model_path(str(actor), force_remerge=True, dry_run=True))
            self.assertNotEqual(forced, metadata.resolve())
            self.assertIn(".eval_cache/fsdp_hf/global_step_8", forced.as_posix())

    def test_missing_fsdp_rank_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            actor = Path(temp_dir) / "global_step_1" / "actor"
            self._write_metadata(actor / "huggingface")
            (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 2}), encoding="utf-8")
            (actor / "model_world_size_2_rank_0.pt").write_bytes(b"rank-0")

            with self.assertRaisesRegex(ModelPreparationError, "missing/empty FSDP model shards"):
                resolve_model_path(str(actor), dry_run=True)

    def test_partial_hf_weight_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "partial"
            self._write_metadata(model_dir)
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )

            valid, reason = validate_hf_model(model_dir)
            self.assertFalse(valid)
            self.assertIn("missing/empty indexed weight shards", reason)

    def test_tokenizer_config_without_vocabulary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "partial"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")

            valid, reason = validate_hf_model(model_dir)
            self.assertFalse(valid)
            self.assertIn("tokenizer vocabulary/model asset", reason)

    def test_adapter_weights_are_not_mistaken_for_full_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "adapter"
            self._write_metadata(model_dir)
            (model_dir / "adapter_model.safetensors").write_bytes(b"adapter")

            valid, reason = validate_hf_model(model_dir)
            self.assertFalse(valid)
            self.assertIn("no non-empty Hugging Face model weights", reason)


if __name__ == "__main__":
    unittest.main()
