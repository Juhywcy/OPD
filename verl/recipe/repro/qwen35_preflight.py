"""Fail-fast checks for Qwen3.5 OPD training in verl."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _check_model(path: Path, expected_arch: str) -> dict[str, object]:
    from transformers import AutoConfig, AutoModelForImageTextToText

    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json: {path}")

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    architectures = list(getattr(config, "architectures", []) or [])
    if expected_arch not in architectures:
        raise RuntimeError(f"{path} has architectures={architectures}, expected {expected_arch}")
    if type(config) not in AutoModelForImageTextToText._model_mapping.keys():
        raise RuntimeError(
            "installed transformers does not register Qwen3.5 under "
            "AutoModelForImageTextToText"
        )

    weight_files = list(path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights found in {path}")
    return {
        "path": str(path),
        "architectures": architectures,
        "weight_files": len(weight_files),
    }


def _check_vllm(expected_arch: str) -> None:
    from vllm.model_executor.models import registry as registry_module

    registered_models = getattr(registry_module, "_VLLM_MODELS", None)
    if isinstance(registered_models, dict):
        supported = expected_arch in registered_models
        if not supported:
            raise RuntimeError(
                f"vLLM {_version('vllm')} does not support {expected_arch}; "
                "Qwen3.5 requires a recent vLLM release"
            )
        return

    try:
        from vllm import ModelRegistry
    except ImportError:
        from vllm.model_executor.models import ModelRegistry

    if hasattr(ModelRegistry, "is_model_supported"):
        supported = bool(ModelRegistry.is_model_supported(expected_arch))
    elif hasattr(ModelRegistry, "get_supported_archs"):
        supported = expected_arch in set(ModelRegistry.get_supported_archs())
    else:
        raise RuntimeError("cannot query the installed vLLM model registry")
    if not supported:
        raise RuntimeError(
            f"vLLM {_version('vllm')} does not support {expected_arch}; "
            "Qwen3.5 requires a recent vLLM release"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    args = parser.parse_args()

    expected_arch = "Qwen3_5ForConditionalGeneration"
    report = {
        "transformers": _version("transformers"),
        "vllm": _version("vllm"),
        "student": _check_model(args.student, expected_arch),
        "teacher": _check_model(args.teacher, expected_arch),
        "datasets": [],
    }
    _check_vllm(expected_arch)

    for dataset in args.dataset:
        if not dataset.is_file():
            raise FileNotFoundError(f"validation dataset does not exist: {dataset}")
        report["datasets"].append(str(dataset))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
