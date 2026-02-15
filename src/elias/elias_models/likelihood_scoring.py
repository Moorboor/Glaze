from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EPSILON, SUPPORTED_MODEL_NAMES
from .continuous_models import (
    _attach_thresholds,
    _coerce_simulated_decision,
    _prepare_model_input,
    _temporary_numpy_seed,
)
from .data_validation import _normalize_choice_values_to_pm1
from .ddm_model import _sigmoid, _simulate_ddm_single_sample

try:
    from evan.glaze import psi_function, simulate_trial
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from evan.glaze import psi_function, simulate_trial


def _build_rt_bin_edges(rt_max_ms: float, rt_bin_width_ms: float) -> np.ndarray:
    """Build fixed-width RT histogram bin edges in milliseconds."""
    if rt_max_ms <= 0.0:
        raise ValueError("rt_max_ms must be > 0")
    if rt_bin_width_ms <= 0.0:
        raise ValueError("rt_bin_width_ms must be > 0")

    n_bins = int(np.ceil(float(rt_max_ms) / float(rt_bin_width_ms)))
    max_edge = float(n_bins) * float(rt_bin_width_ms)
    return np.linspace(0.0, max_edge, n_bins + 1, dtype=float)


def _estimate_rt_density_at_observed_value(
    *,
    observed_rt_ms: float,
    rt_samples_ms: np.ndarray,
    rt_bin_edges: np.ndarray,
    eps: float,
) -> float:
    """Estimate conditional RT density at the observed RT from histogram bins."""
    if not np.isfinite(observed_rt_ms):
        return float(eps)
    if observed_rt_ms < float(rt_bin_edges[0]) or observed_rt_ms > float(rt_bin_edges[-1]):
        return float(eps)

    samples = np.asarray(rt_samples_ms, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float(eps)

    counts, _ = np.histogram(samples, bins=rt_bin_edges)
    smoothed_counts = counts.astype(float) + float(eps)

    bin_width_ms = float(rt_bin_edges[1] - rt_bin_edges[0])
    densities = smoothed_counts / (np.sum(smoothed_counts) * bin_width_ms)

    bin_idx = int(np.searchsorted(rt_bin_edges, observed_rt_ms, side="right") - 1)
    bin_idx = int(np.clip(bin_idx, 0, len(densities) - 1))
    return float(max(densities[bin_idx], float(eps)))


def _simulate_continuous_trials_for_likelihood(
    model_df: pd.DataFrame,
    *,
    stop_on_sat: bool,
    n_sims_per_trial: int,
    max_duration_ms: float,
    dt_ms: float,
    noise_std: float,
    decision_time_ms: float,
    noise_gain: float,
    threshold_mode: str,
    random_seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate per-trial Monte Carlo decision/RT samples for continuous models."""
    if n_sims_per_trial <= 0:
        raise ValueError("n_sims_per_trial must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be > 0")

    thresholded_df = _attach_thresholds(model_df, threshold_mode=threshold_mode)
    dt_sec = float(dt_ms) / 1000.0

    decisions_by_trial: list[np.ndarray] = []
    rts_by_trial: list[np.ndarray] = []

    with _temporary_numpy_seed(random_seed):
        for row in thresholded_df.itertuples(index=False):
            decisions = np.zeros(n_sims_per_trial, dtype=int)
            rts_ms = np.zeros(n_sims_per_trial, dtype=float)
            for sample_idx in range(n_sims_per_trial):
                sim_result = simulate_trial(
                    prev_belief_L=float(row.prev_observed_belief_L),
                    current_LLR=float(row.LLR),
                    H=float(row.H),
                    belief_threshold=float(row.used_threshold),
                    max_duration_ms=float(max_duration_ms),
                    dt=float(dt_sec),
                    noise_std=float(noise_std),
                    decision_time_ms=float(decision_time_ms),
                    noise_gain=float(noise_gain),
                    stop_on_sat=bool(stop_on_sat),
                )
                decisions[sample_idx] = _coerce_simulated_decision(sim_result["decision"])
                rts_ms[sample_idx] = float(sim_result["reaction_time_ms"])

            decisions_by_trial.append(decisions)
            rts_by_trial.append(rts_ms)

    return decisions_by_trial, rts_by_trial


def _simulate_ddm_trials_for_likelihood(
    model_df: pd.DataFrame,
    *,
    n_sims_per_trial: int,
    dt_ms: float,
    max_duration_ms: float,
    boundary_a: float,
    non_decision_time_ms: float,
    llr_to_drift_scale: float,
    start_k: float,
    diffusion_sigma: float,
    random_seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate per-trial Monte Carlo decision/RT samples for the DDM model."""
    if n_sims_per_trial <= 0:
        raise ValueError("n_sims_per_trial must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be > 0")
    if boundary_a <= 0:
        raise ValueError("boundary_a must be > 0")
    if diffusion_sigma <= 0:
        raise ValueError("diffusion_sigma must be > 0")

    rng = np.random.default_rng(random_seed)

    decisions_by_trial: list[np.ndarray] = []
    rts_by_trial: list[np.ndarray] = []

    for row in model_df.itertuples(index=False):
        psi_t = float(psi_function(float(row.prev_observed_belief_L), float(row.H)))
        z_t = _sigmoid(float(start_k) * psi_t)
        z_t = float(np.clip(z_t, EPSILON, 1.0 - EPSILON))
        v_t = float(llr_to_drift_scale) * float(row.LLR)

        decisions = np.zeros(n_sims_per_trial, dtype=int)
        rts_ms = np.zeros(n_sims_per_trial, dtype=float)
        for sample_idx in range(n_sims_per_trial):
            raw_decision, decision_rt_ms, _ = _simulate_ddm_single_sample(
                v=v_t,
                a=float(boundary_a),
                z=z_t,
                dt_ms=float(dt_ms),
                max_duration_ms=float(max_duration_ms),
                diffusion_sigma=float(diffusion_sigma),
                rng=rng,
            )
            decisions[sample_idx] = _coerce_simulated_decision(raw_decision)
            rts_ms[sample_idx] = float(decision_rt_ms + float(non_decision_time_ms))

        decisions_by_trial.append(decisions)
        rts_by_trial.append(rts_ms)

    return decisions_by_trial, rts_by_trial


def score_model_simulation_likelihood(
    df: pd.DataFrame,
    *,
    model_name: str,
    model_params: dict[str, float | int | str] | None = None,
    n_sims_per_trial: int = 2000,
    rt_bin_width_ms: float = 20.0,
    rt_max_ms: float = 5000.0,
    eps: float = 1e-12,
    random_seed: int = 0,
) -> dict[str, object]:
    """Score trials with simulation-based choice and conditional RT likelihoods."""
    model_name_str = str(model_name)
    if model_name_str not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{model_name_str}'. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    if n_sims_per_trial <= 0:
        raise ValueError("n_sims_per_trial must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    params = {} if model_params is None else dict(model_params)
    model_df = _prepare_model_input(df)
    model_df["choice"] = _normalize_choice_values_to_pm1(model_df["choice"].to_numpy())

    if model_name_str in ("cont_threshold", "cont_asymptote"):
        simulated_decisions, simulated_rts_ms = _simulate_continuous_trials_for_likelihood(
            model_df=model_df,
            stop_on_sat=(model_name_str == "cont_asymptote"),
            n_sims_per_trial=int(n_sims_per_trial),
            max_duration_ms=float(params.get("max_duration_ms", 1500.0)),
            dt_ms=float(params.get("dt_ms", 10.0)),
            noise_std=float(params.get("noise_std", 0.7)),
            decision_time_ms=float(params.get("decision_time_ms", 50.0)),
            noise_gain=float(params.get("noise_gain", 3.5)),
            threshold_mode=str(
                params.get("threshold_mode", "participant_block_mean_abs_belief")
            ),
            random_seed=int(random_seed),
        )
    else:
        simulated_decisions, simulated_rts_ms = _simulate_ddm_trials_for_likelihood(
            model_df=model_df,
            n_sims_per_trial=int(n_sims_per_trial),
            dt_ms=float(params.get("dt_ms", 5.0)),
            max_duration_ms=float(params.get("max_duration_ms", 1500.0)),
            boundary_a=float(params.get("boundary_a", 1.0)),
            non_decision_time_ms=float(params.get("non_decision_time_ms", 200.0)),
            llr_to_drift_scale=float(params.get("llr_to_drift_scale", 1.0)),
            start_k=float(params.get("start_k", 0.1)),
            diffusion_sigma=float(params.get("diffusion_sigma", 1.0)),
            random_seed=int(random_seed),
        )

    rt_bin_edges = _build_rt_bin_edges(
        rt_max_ms=float(rt_max_ms),
        rt_bin_width_ms=float(rt_bin_width_ms),
    )

    trial_rows: list[dict[str, object]] = []
    for row, sampled_decisions, sampled_rts_ms in zip(
        model_df.itertuples(index=False),
        simulated_decisions,
        simulated_rts_ms,
    ):
        observed_choice = int(row.choice)
        observed_rt_ms = float(row.reaction_time_ms)

        p_choice = float(np.mean(sampled_decisions == observed_choice))
        p_choice = float(max(p_choice, float(eps)))

        matching_rts_ms = sampled_rts_ms[sampled_decisions == observed_choice]
        if matching_rts_ms.size == 0:
            p_rt_given_choice = float(eps)
        else:
            p_rt_given_choice = _estimate_rt_density_at_observed_value(
                observed_rt_ms=observed_rt_ms,
                rt_samples_ms=matching_rts_ms,
                rt_bin_edges=rt_bin_edges,
                eps=float(eps),
            )
        p_rt_given_choice = float(max(p_rt_given_choice, float(eps)))

        nll_choice = float(-np.log(p_choice))
        nll_rt_cond = float(-np.log(p_rt_given_choice))
        nll_joint = float(nll_choice + nll_rt_cond)

        trial_rows.append(
            {
                "row_id": int(row.row_id),
                "participant_id": str(row.participant_id),
                "block_id": int(row.block_id),
                "trial_index": int(row.trial_index),
                "model_name": model_name_str,
                "observed_choice": observed_choice,
                "observed_rt_ms": observed_rt_ms,
                "p_choice": p_choice,
                "p_rt_given_choice": p_rt_given_choice,
                "nll_choice": nll_choice,
                "nll_rt_cond": nll_rt_cond,
                "nll_joint": nll_joint,
                "n_sims_per_trial": int(n_sims_per_trial),
                "seed_used": int(random_seed),
            }
        )

    trial_scores = pd.DataFrame(trial_rows)
    aggregate_scores = {
        "model_name": model_name_str,
        "joint_score": float(trial_scores["nll_joint"].sum()),
        "choice_only_score": float(trial_scores["nll_choice"].sum()),
        "rt_only_cond_score": float(trial_scores["nll_rt_cond"].sum()),
        "n_trials": int(len(trial_scores)),
        "n_sims_per_trial": int(n_sims_per_trial),
        "random_seed": int(random_seed),
    }

    return {
        "trial_scores": trial_scores,
        "aggregate_scores": aggregate_scores,
    }
