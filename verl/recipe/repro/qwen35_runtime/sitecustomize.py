"""Keep incompatible system FlashAttention out of the Qwen3.5 runtime.

The base training image ships a FlashAttention extension built against its
original PyTorch.  The isolated Qwen3.5 vLLM runtime uses a newer PyTorch, so
loading that extension fails with an undefined C++ symbol.  vLLM has a native
fallback when FlashAttention is not discoverable; hide only that optional
package while leaving every other import untouched.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
from typing import Any


_original_find_spec = importlib.util.find_spec


def _qwen35_find_spec(name: str, *args: Any, **kwargs: Any) -> ModuleSpec | None:
    if name == "flash_attn" or name.startswith("flash_attn."):
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _qwen35_find_spec
