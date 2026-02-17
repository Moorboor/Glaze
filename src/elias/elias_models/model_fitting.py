"""Parameter-space transforms and model fitting for Elias workflow scoring.

Function Inventory:
- `fit_model_parameters`: Multi-start optimizer over unconstrained eta space; called by `core_workflow.fit_models_train_split`.
- `get_parameter_spec`: Ordered parameter spec per model; called by `fit_model_parameters` and parameter helpers.
- `eta_to_theta`: Logistic transform from eta to bounded theta; called by `_score_eta_candidate`.
- `theta_to_eta`: Inverse transform for recovery/debug; internal utility for consistency.
- `theta_to_named_params`: Map theta vector to readable parameter names; called by `fit_model_parameters`.
- `theta_to_scoring_model_params`: Convert theta to scorer-ready parameter dict; called by `_score_eta_candidate`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .model_scoring import score_model_simulation_likelihood


SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "cont_threshold",
    "cont_asymptote",
    "ddm_dnm",
)
_TRANSFORM_EPS = 1e-12

# Single source of truth for parameter dimensionality and bounds.
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

_DEFAULT_FIT_CONFIG: dict[str, object] = {
    "n_starts": 8,
    "n_iterations": 20,
    "step_scale": 0.60,
    "step_scale_decay": 0.95,
    "n_sims_per_trial": 200,
    "rt_bin_width_ms": 20.0,
    "rt_max_ms": 5000.0,
    "eps": 1e-12,
    "score_seed_base": 0,
    "fit_objective": "choice_only",
    "fixed_model_params": {
        "dt_ms": 1.0,
        "max_duration_ms": 5000.0,
    },
}


def _validate_model_name(model_name: str) -> str:
    """Validate model name against supported fitting models."""
    model_name_str = str(model_name)
    if model_name_str not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{model_name_str}'. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    if model_name_str not in _PARAMETER_SPECS:
        raise ValueError(f"No parameter specification defined for '{model_name_str}'.")
    return model_name_str


def _as_1d_float_vector(values: np.ndarray | Iterable[float], *, name: str) -> np.ndarray:
    """Validate and return a finite one-dimensional float vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; found shape {vector.shape}.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _spec_bounds(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return vectorized lower/upper bounds for the model parameter spec."""
    spec = get_parameter_spec(model_name)
    lower = np.asarray([float(parameter["lower"]) for parameter in spec], dtype=float)
    upper = np.asarray([float(parameter["upper"]) for parameter in spec], dtype=float)
    return lower, upper


def _validate_block_ids(block_ids: Iterable[int], *, expected_count: int = 4) -> tuple[int, ...]:
    """Validate block-id sidecar metadata for block-wise model parameters."""
    normalized = tuple(int(block_id) for block_id in block_ids)
    if len(normalized) != expected_count:
        raise ValueError(
            f"block_ids must contain {expected_count} entries, got {len(normalized)}."
        )
    if len(set(normalized)) != expected_count:
        raise ValueError(f"block_ids must be unique, got {normalized}.")
    return normalized


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid that clips logits before exponentiation."""
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit_stable(p: np.ndarray, eps: float = _TRANSFORM_EPS) -> np.ndarray:
    """Stable logit with probability clipping away from 0 and 1."""
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def get_parameter_spec(model_name: str) -> list[dict[str, object]]:
    """Return deep-copied ordered parameter specification for one model."""
    model_name_str = _validate_model_name(model_name)
    return deepcopy(_PARAMETER_SPECS[model_name_str])


def eta_to_theta(model_name: str, eta: np.ndarray) -> np.ndarray:
    """Map unconstrained eta vector into bounded theta space via logistic transform."""
    model_name_str = _validate_model_name(model_name)
    eta_vector = _as_1d_float_vector(eta, name="eta")
    lower, upper = _spec_bounds(model_name_str)
    if eta_vector.size != lower.size:
        raise ValueError(
            f"eta length mismatch for model '{model_name_str}': "
            f"expected {lower.size}, got {eta_vector.size}."
        )

    probabilities = _sigmoid_stable(eta_vector)
    theta = lower + (upper - lower) * probabilities
    return np.clip(theta, lower, upper)


def theta_to_eta(model_name: str, theta: np.ndarray) -> np.ndarray:
    """Map bounded theta vector back to unconstrained eta via logit transform."""
    model_name_str = _validate_model_name(model_name)
    theta_vector = _as_1d_float_vector(theta, name="theta")
    lower, upper = _spec_bounds(model_name_str)
    if theta_vector.size != lower.size:
        raise ValueError(
            f"theta length mismatch for model '{model_name_str}': "
            f"expected {lower.size}, got {theta_vector.size}."
        )

    in_bounds = (theta_vector >= lower) & (theta_vector <= upper)
    if not bool(np.all(in_bounds)):
        invalid_indices = np.where(~in_bounds)[0].tolist()
        raise ValueError(
            f"theta has out-of-bounds values at indices {invalid_indices} "
            f"for model '{model_name_str}'."
        )

    probabilities = (theta_vector - lower) / (upper - lower)
    return _logit_stable(probabilities)


def theta_to_named_params(
    model_name: str,
    theta: np.ndarray,
    block_ids: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> dict[str, float]:
    """Convert theta vector to a dictionary keyed by parameter names."""
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
    """Build scorer-compatible `model_params` from theta values."""
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
            "use_block_sidecar_params": True,
            "block_params_used_in_scoring": True,
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
            "use_block_sidecar_params": True,
            "block_params_used_in_scoring": True,
        }

    return {
        "boundary_a": float(named["a"]),
        "non_decision_time_ms": float(named["t0"]),
        "llr_to_drift_scale": float(named["k_v"]),
        "start_k": float(named["k_z"]),
        # Keep diffusion fixed in this simplified workflow parameterization.
        "diffusion_sigma": 1.0,
    }


def _build_fit_config(fit_config: dict[str, object] | None) -> dict[str, object]:
    """Merge caller fit config with defaults and validate key constraints."""
    merged = deepcopy(_DEFAULT_FIT_CONFIG)
    if fit_config is not None:
        for key, value in fit_config.items():
            if key == "fixed_model_params" and isinstance(value, dict):
                merged_fixed = dict(merged.get("fixed_model_params", {}))
                merged_fixed.update(value)
                merged["fixed_model_params"] = merged_fixed
            else:
                merged[key] = value

    if int(merged["n_starts"]) <= 0:
        raise ValueError("n_starts must be > 0.")
    if int(merged["n_iterations"]) < 0:
        raise ValueError("n_iterations must be >= 0.")
    if float(merged["step_scale"]) <= 0.0:
        raise ValueError("step_scale must be > 0.")
    if float(merged["step_scale_decay"]) <= 0.0:
        raise ValueError("step_scale_decay must be > 0.")
    if int(merged["n_sims_per_trial"]) <= 0:
        raise ValueError("n_sims_per_trial must be > 0.")
    if float(merged["rt_bin_width_ms"]) <= 0.0:
        raise ValueError("rt_bin_width_ms must be > 0.")
    if float(merged["rt_max_ms"]) <= 0.0:
        raise ValueError("rt_max_ms must be > 0.")
    if float(merged["eps"]) <= 0.0:
        raise ValueError("eps must be > 0.")
    if str(merged["fit_objective"]) not in {"choice_only", "joint"}:
        raise ValueError("fit_objective must be one of {'choice_only', 'joint'}.")

    fixed_params = merged.get("fixed_model_params", {})
    if not isinstance(fixed_params, dict):
        raise ValueError("fixed_model_params must be a dictionary.")
    merged["fixed_model_params"] = dict(fixed_params)
    return merged


def _score_eta_candidate(
    *,
    df: pd.DataFrame,
    model_name: str,
    eta_vector: np.ndarray,
    fit_config: dict[str, object],
    score_seed: int,
) -> dict[str, Any]:
    """Evaluate one eta candidate by transforming and scoring its simulation NLL."""
    theta_vector = eta_to_theta(model_name, eta_vector)
    model_params = theta_to_scoring_model_params(model_name, theta_vector)

    fixed_params = dict(fit_config["fixed_model_params"])
    sidecar_override = fixed_params.get("use_block_sidecar_params", None)
    fixed_params.update(model_params)
    if sidecar_override is not None:
        # Allow caller overrides for sidecar usage after model-specific mapping.
        fixed_params["use_block_sidecar_params"] = bool(sidecar_override)

    score_output = score_model_simulation_likelihood(
        df,
        model_name=model_name,
        model_params=fixed_params,
        n_sims_per_trial=int(fit_config["n_sims_per_trial"]),
        rt_bin_width_ms=float(fit_config["rt_bin_width_ms"]),
        rt_max_ms=float(fit_config["rt_max_ms"]),
        eps=float(fit_config["eps"]),
        random_seed=int(score_seed),
    )
    aggregate_scores = dict(score_output["aggregate_scores"])

    fit_objective = str(fit_config.get("fit_objective", "choice_only"))
    if fit_objective == "choice_only":
        fit_objective_score = float(aggregate_scores["choice_only_score"])
    else:
        fit_objective_score = float(aggregate_scores["joint_score"])

    return {
        "eta_vector": np.asarray(eta_vector, dtype=float).copy(),
        "theta_vector": np.asarray(theta_vector, dtype=float).copy(),
        "aggregate_scores": aggregate_scores,
        "fit_objective": fit_objective,
        "fit_objective_score": float(fit_objective_score),
        "joint_score": float(aggregate_scores["joint_score"]),
        "choice_only_score": float(aggregate_scores["choice_only_score"]),
        "rt_only_cond_score": float(aggregate_scores["rt_only_cond_score"]),
        "model_params": fixed_params,
    }


def fit_model_parameters(
    df: pd.DataFrame,
    model_name: str,
    fit_config: dict[str, object] | None = None,
    random_seed: int = 0,
) -> dict[str, object]:
    """Fit one model using multi-start stochastic local search in eta space."""
    _validate_model_name(model_name)
    config = _build_fit_config(fit_config)
    parameter_spec = get_parameter_spec(model_name)
    n_parameters = len(parameter_spec)
    rng = np.random.default_rng(int(random_seed))

    trace_rows: list[dict[str, object]] = []
    n_evaluations = 0
    best_result: dict[str, Any] | None = None

    n_starts = int(config["n_starts"])
    n_iterations = int(config["n_iterations"])
    step_scale = float(config["step_scale"])
    step_decay = float(config["step_scale_decay"])
    base_seed = int(config["score_seed_base"])

    for start_index in range(n_starts):
        # Randomized start locations reduce sensitivity to local minima.
        current_eta = rng.normal(loc=0.0, scale=1.0, size=n_parameters)
        score_seed = base_seed + start_index * 100_003

        current_result = _score_eta_candidate(
            df=df,
            model_name=model_name,
            eta_vector=current_eta,
            fit_config=config,
            score_seed=score_seed,
        )
        n_evaluations += 1

        trace_rows.append(
            {
                "start_index": int(start_index),
                "iteration_index": -1,
                "fit_objective": str(current_result["fit_objective"]),
                "fit_objective_score": float(current_result["fit_objective_score"]),
                "joint_score": float(current_result["joint_score"]),
                "accepted": True,
            }
        )

        if best_result is None or current_result["fit_objective_score"] < best_result["fit_objective_score"]:
            best_result = current_result

        local_scale = step_scale
        for iteration_index in range(n_iterations):
            # Local random walk proposal with multiplicative step-size decay.
            proposed_eta = current_eta + rng.normal(
                loc=0.0,
                scale=local_scale,
                size=n_parameters,
            )
            score_seed += 1
            proposed_result = _score_eta_candidate(
                df=df,
                model_name=model_name,
                eta_vector=proposed_eta,
                fit_config=config,
                score_seed=score_seed,
            )
            n_evaluations += 1

            accepted = (
                proposed_result["fit_objective_score"]
                < current_result["fit_objective_score"]
            )
            if accepted:
                current_eta = proposed_eta
                current_result = proposed_result

            trace_rows.append(
                {
                    "start_index": int(start_index),
                    "iteration_index": int(iteration_index),
                    "fit_objective": str(proposed_result["fit_objective"]),
                    "fit_objective_score": float(proposed_result["fit_objective_score"]),
                    "joint_score": float(proposed_result["joint_score"]),
                    "accepted": bool(accepted),
                }
            )

            if current_result["fit_objective_score"] < best_result["fit_objective_score"]:
                best_result = current_result

            local_scale *= step_decay

    if best_result is None:
        raise RuntimeError("No optimization evaluations were performed.")

    best_theta = np.asarray(best_result["theta_vector"], dtype=float)
    best_eta = np.asarray(best_result["eta_vector"], dtype=float)
    best_named_params = theta_to_named_params(model_name, best_theta)

    return {
        "model_name": str(model_name),
        "fit_objective": str(best_result["fit_objective"]),
        "best_fit_objective_score": float(best_result["fit_objective_score"]),
        "best_joint_score": float(best_result["joint_score"]),
        "best_choice_only_score": float(best_result["choice_only_score"]),
        "best_rt_only_cond_score": float(best_result["rt_only_cond_score"]),
        "best_eta": best_eta,
        "best_theta": best_theta,
        "best_named_params": best_named_params,
        "best_model_params": dict(best_result["model_params"]),
        "n_parameters": int(n_parameters),
        "n_evaluations": int(n_evaluations),
        "n_starts": int(n_starts),
        "n_iterations": int(n_iterations),
        "random_seed": int(random_seed),
        "fit_config": config,
        "trace_table": pd.DataFrame(trace_rows),
    }
