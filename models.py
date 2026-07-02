from __future__ import annotations

from typing import Callable

import torch.nn as nn

from model.internal_runtime import AssembledRuntimeA, AssembledRuntimeB


def build_public_runtime_params(variant: str, recursion: str = "adaptive") -> dict:
    params = {
        "max_steps": 5,
        "min_steps": 2,
        "process_mode": "double_evidence_halt_patience",
        "halt_patience": 2,
        "halt_threshold": 0.6,
        "local_branch": "off",
    }
    if str(variant) == "secondary":
        params.update({"M": 3, "N": 21, "dropout_flag": False})
    else:
        params.update({"in_channels": 1, "num_classes": 2, "widths": [32, 64, 128, 256], "depths": [1, 1, 2, 2]})
    if str(recursion) == "fixed_t5":
        params.update({"min_steps": 5, "process_mode": "fixed_T", "halt_patience": 6, "halt_threshold": 1.1})
    return params


def _build_runtime_primary(**kwargs) -> nn.Module:
    return AssembledRuntimeA(**dict(kwargs))


def _build_runtime_secondary(**kwargs) -> nn.Module:
    params = dict(kwargs)
    params.pop("in_channels", None)
    return AssembledRuntimeB(**params)


_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    'refinement_primary': _build_runtime_primary,
    'refinement_secondary': _build_runtime_secondary,
}


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def get_model(name: str, **kwargs) -> nn.Module:
    key = str(name)
    builder = _MODEL_REGISTRY.get(key)
    if builder is None:
        raise ValueError(f"Unsupported model '{name}'. This release exposes only: {', '.join(available_models())}")
    return builder(**kwargs)
