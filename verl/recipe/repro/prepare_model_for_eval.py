#!/usr/bin/env python3
"""Resolve an evaluation model, safely merging a verl FSDP actor if needed.

Accepted inputs:

* a Hugging Face Hub model id;
* a complete local Hugging Face model directory;
* ``global_step_N/actor``;
* ``global_step_N`` (its ``actor`` child is used);
* a run directory containing ``latest_checkpointed_iteration.txt``.

Raw actor checkpoints are never modified or deleted.  FSDP model shards are
merged into a fingerprinted cache through a temporary directory and atomic
rename.  Diagnostics go to stderr; stdout contains only the resolved model
path so a Bash caller can safely use command substitution.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_HUB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MODEL_SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_UNSHARDED_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_TOKENIZER_ASSET_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
)
_MANIFEST_NAME = ".merge_complete.json"


class ModelPreparationError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(f"[model-prepare] {message}", file=sys.stderr, flush=True)


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelPreparationError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelPreparationError(f"expected a JSON object in {path}")
    return value


def validate_hf_model(path: Path) -> tuple[bool, str]:
    """Check config, tokenizer metadata, and every referenced weight shard."""

    if not path.is_dir():
        return False, "directory does not exist"

    config_path = path / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        return False, "missing or empty config.json"
    try:
        _load_json(config_path)
    except ModelPreparationError as exc:
        return False, str(exc)

    if not any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in _TOKENIZER_ASSET_FILES
    ):
        return False, "missing tokenizer vocabulary/model asset"

    for index_name in _INDEX_FILES:
        index_path = path / index_name
        if not index_path.exists():
            continue
        try:
            index = _load_json(index_path)
        except ModelPreparationError as exc:
            return False, str(exc)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False, f"missing weight_map in {index_name}"
        referenced = sorted({str(name) for name in weight_map.values()})
        missing = [name for name in referenced if not (path / name).is_file() or (path / name).stat().st_size == 0]
        if missing:
            return False, f"missing/empty indexed weight shards: {missing[:5]}"
        return True, f"complete indexed Hugging Face model ({len(referenced)} shards)"

    weights = [path / name for name in _UNSHARDED_WEIGHT_FILES]
    weights = [weight for weight in weights if weight.is_file() and weight.stat().st_size > 0]
    if not weights:
        return False, "no non-empty Hugging Face model weights"
    return True, f"complete Hugging Face model ({len(weights)} weight file(s))"


def _actor_from_existing_path(path: Path) -> Path | None:
    if (path / "fsdp_config.json").is_file():
        return path
    if path.name == "huggingface" and (path.parent / "fsdp_config.json").is_file():
        return path.parent
    if (path / "actor" / "fsdp_config.json").is_file():
        return path / "actor"

    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        raw_step = tracker.read_text(encoding="utf-8").strip()
        if not raw_step.isdigit():
            raise ModelPreparationError(f"invalid checkpoint tracker {tracker}: {raw_step!r}")
        actor = path / f"global_step_{int(raw_step)}" / "actor"
        if not actor.is_dir():
            raise ModelPreparationError(f"tracker points to a missing actor directory: {actor}")
        return actor
    return None


def _validate_fsdp_actor(actor: Path) -> tuple[int, list[Path]]:
    fsdp_config_path = actor / "fsdp_config.json"
    if not fsdp_config_path.is_file():
        raise ModelPreparationError(
            f"{actor} is not a supported current-format FSDP actor: missing fsdp_config.json. "
            "Use verl/scripts/legacy_model_merger.py manually for an older checkpoint."
        )

    fsdp_config = _load_json(fsdp_config_path)
    world_size = fsdp_config.get("world_size")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ModelPreparationError(f"invalid world_size in {fsdp_config_path}: {world_size!r}")

    expected = [actor / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ModelPreparationError(f"missing/empty FSDP model shards: {missing[:8]}")

    discovered: list[tuple[int, int, Path]] = []
    for path in actor.glob("model_world_size_*_rank_*.pt"):
        match = _MODEL_SHARD_RE.fullmatch(path.name)
        if match:
            discovered.append((int(match.group(1)), int(match.group(2)), path))
    unexpected = [str(path) for size, rank, path in discovered if size != world_size or rank >= world_size]
    if unexpected:
        raise ModelPreparationError(
            "actor contains model shards from a different/incompatible world size: " + ", ".join(unexpected[:8])
        )

    metadata_dir = actor / "huggingface"
    config_path = metadata_dir / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        raise ModelPreparationError(f"missing actor Hugging Face metadata: {config_path}")
    _load_json(config_path)
    if not any(
        (metadata_dir / name).is_file() and (metadata_dir / name).stat().st_size > 0
        for name in _TOKENIZER_ASSET_FILES
    ):
        raise ModelPreparationError(f"missing tokenizer vocabulary/model asset under {metadata_dir}")

    return world_size, expected


def _merger_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for module_name in (
        "verl.model_merger.base_model_merger",
        "verl.model_merger.fsdp_model_merger",
    ):
        try:
            spec = importlib.util.find_spec(module_name)
            origin = Path(spec.origin) if spec and spec.origin else None
            if origin and origin.is_file():
                hashes[module_name] = _sha256_file(origin)
        except Exception as exc:
            hashes[module_name] = f"unavailable:{type(exc).__name__}"
    return hashes


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("torch", "transformers", "accelerate", "safetensors", "verl"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed-as-distribution"
    return versions


def _source_snapshot(
    actor: Path,
    shards: list[Path],
    trust_remote_code: bool,
    strict_shard_hash: bool,
) -> tuple[str, dict[str, Any]]:
    def stat_record(path: Path) -> dict[str, Any]:
        stat = path.stat()
        record: dict[str, Any] = {
            "path": str(path.relative_to(actor)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
        }
        if strict_shard_hash:
            record["sha256"] = _sha256_file(path)
        return record

    metadata_records = []
    for path in sorted((actor / "huggingface").rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        record = {
            "path": str(path.relative_to(actor)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        metadata_records.append(record)

    payload: dict[str, Any] = {
        "actor": str(actor.resolve()),
        "fsdp_config_sha256": _sha256_file(actor / "fsdp_config.json"),
        "model_shards": [stat_record(path) for path in shards],
        "huggingface_metadata": metadata_records,
        "merger_sources": _merger_source_hashes(),
        "runtime_versions": _runtime_versions(),
        "trust_remote_code": trust_remote_code,
        "strict_shard_hash": strict_shard_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _default_cache_root(actor: Path) -> Path:
    step_dir = actor.parent
    run_dir = step_dir.parent if step_dir.name.startswith("global_step_") else actor.parent
    return run_dir / ".eval_cache" / "fsdp_hf"


def _read_manifest(target: Path) -> dict[str, Any] | None:
    manifest = target / _MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        return _load_json(manifest)
    except ModelPreparationError:
        return None


def _cache_is_usable(target: Path, fingerprint: str) -> tuple[bool, str]:
    valid, reason = validate_hf_model(target)
    if not valid:
        return False, reason
    manifest = _read_manifest(target)
    if not manifest or manifest.get("source_fingerprint") != fingerprint:
        return False, "missing or mismatched merge manifest"
    return True, reason


def _replace_directory_atomically(staging: Path, target: Path) -> None:
    if target.is_symlink():
        raise ModelPreparationError(f"refusing to replace symlink cache target: {target}")

    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.stale.{socket.gethostname()}.{os.getpid()}")
        if backup.exists():
            raise ModelPreparationError(f"stale backup path already exists: {backup}")
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _looks_like_missing_local_path(raw_path: str) -> bool:
    path = Path(raw_path).expanduser()
    return (
        path.is_absolute()
        or raw_path.startswith(("./", "../", "~"))
        or "global_step_" in raw_path
        or raw_path.startswith(
            (
                "checkpoint/",
                "checkpoints/",
                "ckpt/",
                "model/",
                "models/",
                "output/",
                "outputs/",
            )
        )
    )


def resolve_model_path(
    requested_path: str,
    *,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
    force_remerge: bool = False,
    strict_shard_hash: bool = False,
    dry_run: bool = False,
) -> str:
    expanded = Path(requested_path).expanduser()

    if not expanded.exists():
        if _HUB_ID_RE.fullmatch(requested_path) and not _looks_like_missing_local_path(requested_path):
            _log(f"Hugging Face Hub id detected; no conversion required: {requested_path}")
            return requested_path
        raise ModelPreparationError(f"local model/checkpoint path does not exist: {expanded}")

    if not expanded.is_dir():
        raise ModelPreparationError(f"model path must be a directory or Hub id: {expanded}")
    expanded = expanded.resolve()

    is_hf, hf_reason = validate_hf_model(expanded)
    if is_hf:
        _log(f"complete local Hugging Face model detected ({hf_reason}): {expanded}")
        return str(expanded)

    actor = _actor_from_existing_path(expanded)
    if actor is None:
        raise ModelPreparationError(
            f"directory is neither a complete Hugging Face model nor a supported verl actor: {expanded} "
            f"(HF check: {hf_reason})"
        )
    actor = actor.resolve()

    world_size, shards = _validate_fsdp_actor(actor)
    in_checkpoint_hf = actor / "huggingface"
    in_checkpoint_valid, in_checkpoint_reason = validate_hf_model(in_checkpoint_hf)
    if in_checkpoint_valid and not force_remerge:
        _log(f"checkpoint already contains a full Hugging Face model ({in_checkpoint_reason}): {in_checkpoint_hf}")
        return str(in_checkpoint_hf)
    if in_checkpoint_valid:
        _log(f"force re-merge requested; ignoring checkpoint-contained Hugging Face weights: {in_checkpoint_hf}")

    fingerprint, snapshot = _source_snapshot(actor, shards, trust_remote_code, strict_shard_hash)
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else _default_cache_root(actor).resolve()
    target = cache_root / actor.parent.name / fingerprint[:24]
    usable, reason = _cache_is_usable(target, fingerprint)
    if usable and not force_remerge:
        _log(f"reusing merged FSDP cache ({reason}): {target}")
        return str(target)

    if dry_run:
        action = "would force re-merge" if force_remerge else "would merge"
        _log(
            f"{action} verl FSDP actor (world_size={world_size}) into cache: "
            f"{actor} -> {target}"
        )
        return str(target)

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_dir = cache_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    actor_lock_id = hashlib.sha256(str(actor).encode("utf-8")).hexdigest()[:24]
    lock_path = lock_dir / f"{actor_lock_id}.lock"

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        _log(f"waiting for merge lock: {lock_path}")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

        # Re-read everything after acquiring the lock to avoid a stale decision.
        world_size, shards = _validate_fsdp_actor(actor)
        fingerprint, snapshot = _source_snapshot(actor, shards, trust_remote_code, strict_shard_hash)
        target = cache_root / actor.parent.name / fingerprint[:24]
        usable, reason = _cache_is_usable(target, fingerprint)
        if usable and not force_remerge:
            _log(f"another process prepared the cache; reusing it ({reason}): {target}")
            return str(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(target.parent)
        shard_bytes = sum(path.stat().st_size for path in shards)
        if disk.free < 2 * 1024**3:
            raise ModelPreparationError(
                f"less than 2 GiB free on merge cache filesystem: {target.parent}"
            )
        recommended_free = int(1.25 * shard_bytes) + 2 * 1024**3
        if disk.free < recommended_free:
            _log(
                "warning: free disk is below the conservative recommendation "
                f"({disk.free / 1024**3:.1f} GiB free; {recommended_free / 1024**3:.1f} GiB recommended)"
            )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{fingerprint[:12]}.tmp.{socket.gethostname()}.",
                dir=str(target.parent),
            )
        )
        try:
            command = [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor),
                "--target_dir",
                str(staging),
            ]
            if trust_remote_code:
                command.append("--trust-remote-code")

            _log(
                f"merging {world_size} FSDP model shards on CPU; source checkpoint is preserved: "
                f"{actor} -> {staging}"
            )
            merge_env = os.environ.copy()
            # The official merger loads and combines shards on CPU.  Hiding all
            # GPUs also prevents it from reserving accelerator memory before
            # Ray/vLLM starts.
            merge_env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(command, check=True, env=merge_env, stdout=sys.stderr, stderr=sys.stderr)

            valid, output_reason = validate_hf_model(staging)
            if not valid:
                raise ModelPreparationError(f"merged output validation failed: {output_reason}")

            after_fingerprint, _ = _source_snapshot(actor, shards, trust_remote_code, strict_shard_hash)
            if after_fingerprint != fingerprint:
                raise ModelPreparationError(
                    "actor checkpoint changed while it was being merged; retry after training/checkpoint rotation stops"
                )

            manifest = {
                "source_actor": str(actor),
                "source_fingerprint": fingerprint,
                "world_size": world_size,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.executable,
                "output_validation": output_reason,
                "source_snapshot": snapshot,
            }
            (staging / _MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _replace_directory_atomically(staging, target)
            _log(f"FSDP actor merge completed: {target}")
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model id/path or verl checkpoint path")
    parser.add_argument("--cache-dir", help="override the default per-run .eval_cache/fsdp_hf directory")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-remerge", action="store_true")
    parser.add_argument(
        "--strict-shard-hash",
        action="store_true",
        help="SHA256 every large model shard; safer but reads all shard bytes before/after merge",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate without writing/merging")
    args = parser.parse_args()

    try:
        resolved = resolve_model_path(
            args.model_path,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            force_remerge=args.force_remerge,
            strict_shard_hash=args.strict_shard_hash,
            dry_run=args.dry_run,
        )
    except ModelPreparationError as exc:
        _log(f"ERROR: {exc}")
        return 2

    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
