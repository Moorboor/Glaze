from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .likelihood_scoring import score_model_simulation_likelihood
from .parameter_space import (
    eta_to_theta,
    get_parameter_spec,
    theta_to_named_params,
    theta_to_scoring_model_params,
)


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
    "fixed_model_params": {
        "dt_ms": 1.0,
        "max_duration_ms": 5000.0,
    },
}


def _build_fit_config(fit_config: dict[str, object] | None) -> dict[str, object]:
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
    theta_vector = eta_to_theta(model_name, eta_vector)
    model_params = theta_to_scoring_model_params(model_name, theta_vector)
    fixed_params = dict(fit_config["fixed_model_params"])
    fixed_params.update(model_params)

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
    aggregate = dict(score_output["aggregate_scores"])

    return {
        "eta_vector": np.asarray(eta_vector, dtype=float).copy(),
        "theta_vector": np.asarray(theta_vector, dtype=float).copy(),
        "aggregate_scores": aggregate,
        "joint_score": float(aggregate["joint_score"]),
        "choice_only_score": float(aggregate["choice_only_score"]),
        "rt_only_cond_score": float(aggregate["rt_only_cond_score"]),
        "model_params": fixed_params,
    }


def fit_model_parameters(
    df: pd.DataFrame,
    model_name: str,
    fit_config: dict[str, object] | None = None,
    random_seed: int = 0,
) -> dict[str, object]:
    """Fit one model via multi-start search in eta space against joint score."""
    config = _build_fit_config(fit_config)
    parameter_spec = get_parameter_spec(model_name)
    n_params = len(parameter_spec)
    rng = np.random.default_rng(int(random_seed))

    trace_rows: list[dict[str, object]] = []
    n_evaluations = 0
    best_result: dict[str, Any] | None = None

    n_starts = int(config["n_starts"])
    n_iterations = int(config["n_iterations"])
    step_scale = float(config["step_scale"])
    step_decay = float(config["step_scale_decay"])
    base_seed = int(config["score_seed_base"])

    for start_idx in range(n_starts):
        current_eta = rng.normal(loc=0.0, scale=1.0, size=n_params)
        score_seed = base_seed + start_idx * 100_003
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
                "start_index": int(start_idx),
                "iteration_index": -1,
                "joint_score": float(current_result["joint_score"]),
                "accepted": True,
            }
        )

        if best_result is None or current_result["joint_score"] < best_result["joint_score"]:
            best_result = current_result

        local_scale = step_scale
        for iteration_idx in range(n_iterations):
            proposed_eta = current_eta + rng.normal(
                loc=0.0,
                scale=local_scale,
                size=n_params,
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

            accepted = proposed_result["joint_score"] < current_result["joint_score"]
            if accepted:
                current_eta = proposed_eta
                current_result = proposed_result

            trace_rows.append(
                {
                    "start_index": int(start_idx),
                    "iteration_index": int(iteration_idx),
                    "joint_score": float(proposed_result["joint_score"]),
                    "accepted": bool(accepted),
                }
            )

            if current_result["joint_score"] < best_result["joint_score"]:
                best_result = current_result

            local_scale *= step_decay

    if best_result is None:
        raise RuntimeError("No optimization evaluations were performed.")

    best_theta = np.asarray(best_result["theta_vector"], dtype=float)
    best_eta = np.asarray(best_result["eta_vector"], dtype=float)
    best_named_params = theta_to_named_params(model_name, best_theta)

    return {
        "model_name": str(model_name),
        "best_joint_score": float(best_result["joint_score"]),
        "best_choice_only_score": float(best_result["choice_only_score"]),
        "best_rt_only_cond_score": float(best_result["rt_only_cond_score"]),
        "best_eta": best_eta,
        "best_theta": best_theta,
        "best_named_params": best_named_params,
        "best_model_params": dict(best_result["model_params"]),
        "n_parameters": int(n_params),
        "n_evaluations": int(n_evaluations),
        "n_starts": int(n_starts),
        "n_iterations": int(n_iterations),
        "random_seed": int(random_seed),
        "fit_config": config,
        "trace_table": pd.DataFrame(trace_rows),
    }
