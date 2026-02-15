from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np

from .constants import SUPPORTED_MODEL_NAMES


_TRANSFORM_EPS = 1e-12

_PARAMETER_SPECS: dict[str, list[dict[str, object]]] = {
    "cont_threshold": [
        {"name": "thr_b1", "lower": 0.10, "upper": 15.00},
        {"name": "thr_b2", "lower": 0.10, "upper": 15.00},
        {"name": "thr_b3", "lower": 0.10, "upper": 15.00},
        {"name": "thr_b4", "lower": 0.10, "upper": 15.00},
        {"name": "t0", "lower": 0.00, "upper": 1000.00},
        {"name": "g", "lower": 0.20, "upper": 8.00},
    ],
    "cont_asymptote": [
        {"name": "asy_b1", "lower": 0.10, "upper": 15.00},
        {"name": "asy_b2", "lower": 0.10, "upper": 15.00},
        {"name": "asy_b3", "lower": 0.10, "upper": 15.00},
        {"name": "asy_b4", "lower": 0.10, "upper": 15.00},
        {"name": "t0", "lower": 0.00, "upper": 1000.00},
        {"name": "g", "lower": 0.20, "upper": 8.00},
    ],
    "ddm_dnm": [
        {"name": "a", "lower": 0.20, "upper": 4.00},
        {"name": "t0", "lower": 0.00, "upper": 1000.00},
        {"name": "k_v", "lower": -6.00, "upper": 6.00},
        {"name": "k_z", "lower": -6.00, "upper": 6.00},
    ],
}


def _validate_model_name(model_name: str) -> str:
    model_name_str = str(model_name)
    if model_name_str not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{model_name_str}'. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    if model_name_str not in _PARAMETER_SPECS:
        raise ValueError(f"No parameter specification registered for '{model_name_str}'.")
    return model_name_str


def _spec_bounds(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = get_parameter_spec(model_name)
    lower = np.asarray([float(param["lower"]) for param in spec], dtype=float)
    upper = np.asarray([float(param["upper"]) for param in spec], dtype=float)
    return lower, upper


def _as_1d_float_vector(values: np.ndarray | Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector; found shape {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _validate_block_ids(block_ids: Iterable[int], *, expected_count: int = 4) -> tuple[int, ...]:
    normalized = tuple(int(block_id) for block_id in block_ids)
    if len(normalized) != expected_count:
        raise ValueError(
            f"block_ids must contain {expected_count} entries, got {len(normalized)}."
        )
    if len(set(normalized)) != expected_count:
        raise ValueError(f"block_ids must be unique, got {normalized}.")
    return normalized


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit_stable(p: np.ndarray, eps: float = _TRANSFORM_EPS) -> np.ndarray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def get_parameter_spec(model_name: str) -> list[dict[str, object]]:
    """Return the ordered parameter specification for a model."""
    model_name_str = _validate_model_name(model_name)
    return deepcopy(_PARAMETER_SPECS[model_name_str])


def eta_to_theta(model_name: str, eta: np.ndarray) -> np.ndarray:
    """Map unconstrained `eta` vector to bounded `theta` with a logistic transform."""
    model_name_str = _validate_model_name(model_name)
    eta_vector = _as_1d_float_vector(eta, name="eta")
    lower, upper = _spec_bounds(model_name_str)
    if eta_vector.size != lower.size:
        raise ValueError(
            f"eta length mismatch for model '{model_name_str}': "
            f"expected {lower.size}, got {eta_vector.size}."
        )
    p = _sigmoid_stable(eta_vector)
    theta = lower + (upper - lower) * p
    return np.clip(theta, lower, upper)


def theta_to_eta(model_name: str, theta: np.ndarray) -> np.ndarray:
    """Map bounded `theta` vector to unconstrained `eta` via logit."""
    model_name_str = _validate_model_name(model_name)
    theta_vector = _as_1d_float_vector(theta, name="theta")
    lower, upper = _spec_bounds(model_name_str)
    if theta_vector.size != lower.size:
        raise ValueError(
            f"theta length mismatch for model '{model_name_str}': "
            f"expected {lower.size}, got {theta_vector.size}."
        )
    in_range_mask = (theta_vector >= lower) & (theta_vector <= upper)
    if not bool(np.all(in_range_mask)):
        invalid_indices = np.where(~in_range_mask)[0].tolist()
        raise ValueError(
            f"theta contains out-of-bounds values at indices {invalid_indices} "
            f"for model '{model_name_str}'."
        )
    p = (theta_vector - lower) / (upper - lower)
    return _logit_stable(p)


def theta_to_named_params(
    model_name: str,
    theta: np.ndarray,
    block_ids: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> dict[str, float]:
    """Convert ordered `theta` values to named scalar parameters."""
    model_name_str = _validate_model_name(model_name)
    theta_vector = _as_1d_float_vector(theta, name="theta")
    spec = get_parameter_spec(model_name_str)
    if theta_vector.size != len(spec):
        raise ValueError(
            f"theta length mismatch for model '{model_name_str}': "
            f"expected {len(spec)}, got {theta_vector.size}."
        )

    named = {
        str(spec_entry["name"]): float(value)
        for spec_entry, value in zip(spec, theta_vector, strict=True)
    }

    if model_name_str in ("cont_threshold", "cont_asymptote"):
        _validate_block_ids(block_ids, expected_count=4)

    return named


def theta_to_scoring_model_params(
    model_name: str,
    theta: np.ndarray,
    block_ids: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> dict[str, object]:
    """Build scorer-compatible `model_params` plus A/B block sidecar metadata."""
    model_name_str = _validate_model_name(model_name)
    named = theta_to_named_params(model_name_str, theta, block_ids=block_ids)
    normalized_blocks = _validate_block_ids(block_ids, expected_count=4)

    if model_name_str == "cont_threshold":
        block_param_order = tuple(f"thr_b{i}" for i in range(1, 5))
        threshold_by_block = {
            int(block_id): float(named[param_name])
            for block_id, param_name in zip(
                normalized_blocks,
                block_param_order,
                strict=True,
            )
        }
        return {
            "decision_time_ms": float(named["t0"]),
            "noise_gain": float(named["g"]),
            "threshold_mode": "participant_block_mean_abs_belief",
            "threshold_by_block_sidecar": threshold_by_block,
            "block_ids_sidecar": list(normalized_blocks),
            "block_param_order_sidecar": list(block_param_order),
            "block_params_used_in_scoring": False,
        }

    if model_name_str == "cont_asymptote":
        block_param_order = tuple(f"asy_b{i}" for i in range(1, 5))
        asymptote_by_block = {
            int(block_id): float(named[param_name])
            for block_id, param_name in zip(
                normalized_blocks,
                block_param_order,
                strict=True,
            )
        }
        return {
            "decision_time_ms": float(named["t0"]),
            "noise_gain": float(named["g"]),
            "threshold_mode": "participant_block_mean_abs_belief",
            "asymptote_by_block_sidecar": asymptote_by_block,
            "block_ids_sidecar": list(normalized_blocks),
            "block_param_order_sidecar": list(block_param_order),
            "block_params_used_in_scoring": False,
        }

    return {
        "boundary_a": float(named["a"]),
        "non_decision_time_ms": float(named["t0"]),
        "llr_to_drift_scale": float(named["k_v"]),
        "start_k": float(named["k_z"]),
        "diffusion_sigma": 1.0,
    }
