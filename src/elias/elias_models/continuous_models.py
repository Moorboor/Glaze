from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EPSILON, MODEL_READY_COLUMNS
from .data_validation import (
    _coerce_numeric_columns,
    _drop_non_finite_rows,
    _validate_required_columns,
)

try:
    from evan.glaze import simulate_trial
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from evan.glaze import simulate_trial


def _prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize model input DataFrame."""
    _validate_required_columns(df, MODEL_READY_COLUMNS, context="model simulation")

    model_df = df.copy()
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df = _coerce_numeric_columns(
        model_df,
        [
            "row_id",
            "block_id",
            "trial_index",
            "LLR",
            "H",
            "choice",
            "reaction_time_ms",
            "belief_L",
            "prev_observed_belief_L",
        ],
    )
    model_df = _drop_non_finite_rows(
        model_df,
        [
            "row_id",
            "block_id",
            "trial_index",
            "LLR",
            "H",
            "choice",
            "reaction_time_ms",
            "belief_L",
            "prev_observed_belief_L",
        ],
        context="model simulation",
    )

    model_df = model_df.sort_values(["participant_id", "block_id", "trial_index"]).reset_index(
        drop=True
    )
    model_df["row_id"] = model_df["row_id"].astype(int)
    return model_df


def _attach_thresholds(
    df: pd.DataFrame,
    threshold_mode: str,
) -> pd.DataFrame:
    """Attach per participant-block thresholds."""
    if threshold_mode != "participant_block_mean_abs_belief":
        raise ValueError(
            f"Unsupported threshold_mode '{threshold_mode}'. "
            "Only 'participant_block_mean_abs_belief' is supported."
        )

    thresholds = (
        df.groupby(["participant_id", "block_id"], sort=False)["belief_L"]
        .agg(lambda x: max(float(np.mean(np.abs(x))), EPSILON))
        .rename("used_threshold")
        .reset_index()
    )

    out = df.merge(thresholds, on=["participant_id", "block_id"], how="left")
    out["used_threshold"] = out["used_threshold"].fillna(EPSILON).clip(lower=EPSILON)
    return out


@contextmanager
def _temporary_numpy_seed(seed: int):
    """Temporarily set and restore NumPy global RNG state."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _coerce_simulated_decision(raw_decision: float | int) -> int:
    """Coerce raw simulator decision to trinary coding (`-1`, `0`, `+1`)."""
    if float(raw_decision) > 0.0:
        return 1
    if float(raw_decision) < 0.0:
        return -1
    return 0


def _run_continuous_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    stop_on_sat: bool,
    max_duration_ms: float,
    dt_ms: float,
    noise_std: float,
    decision_time_ms: float,
    noise_gain: float,
    threshold_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    """Run one continuous-model variant using `simulate_trial`."""
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")

    model_df = _prepare_model_input(df)
    model_df = _attach_thresholds(model_df, threshold_mode=threshold_mode)

    results: list[dict[str, object]] = []

    with _temporary_numpy_seed(random_seed):
        for row in model_df.itertuples(index=False):
            sim_result = simulate_trial(
                prev_belief_L=float(row.prev_observed_belief_L),
                current_LLR=float(row.LLR),
                H=float(row.H),
                belief_threshold=float(row.used_threshold),
                max_duration_ms=float(max_duration_ms),
                dt=float(dt_ms) / 1000.0,
                noise_std=float(noise_std),
                decision_time_ms=float(decision_time_ms),
                noise_gain=float(noise_gain),
                stop_on_sat=bool(stop_on_sat),
            )

            results.append(
                {
                    "row_id": int(row.row_id),
                    "participant_id": str(row.participant_id),
                    "block_id": int(row.block_id),
                    "trial_index": int(row.trial_index),
                    "choice": int(row.choice),
                    "reaction_time_ms": float(row.reaction_time_ms),
                    "belief_L": float(row.belief_L),
                    "LLR": float(row.LLR),
                    "H": float(row.H),
                    "prev_observed_belief_L": float(row.prev_observed_belief_L),
                    "used_threshold": float(row.used_threshold),
                    "predicted_decision": int(sim_result["decision"]),
                    "predicted_rt_ms": float(sim_result["reaction_time_ms"]),
                    "predicted_belief": float(sim_result["final_belief"]),
                    "model_name": model_name,
                    "param_stop_on_sat": bool(stop_on_sat),
                    "param_max_duration_ms": float(max_duration_ms),
                    "param_dt_ms": float(dt_ms),
                    "param_noise_std": float(noise_std),
                    "param_decision_time_ms": float(decision_time_ms),
                    "param_noise_gain": float(noise_gain),
                    "param_threshold_mode": threshold_mode,
                    "param_random_seed": int(random_seed),
                }
            )

    return pd.DataFrame(results)


def run_model_a_threshold(
    df: pd.DataFrame,
    *,
    max_duration_ms: float = 1500.0,
    dt_ms: float = 10.0,
    noise_std: float = 0.7,
    decision_time_ms: float = 50.0,
    noise_gain: float = 3.5,
    threshold_mode: str = "participant_block_mean_abs_belief",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run Model A (continuous threshold variant)."""
    return _run_continuous_model(
        df,
        model_name="cont_threshold",
        stop_on_sat=False,
        max_duration_ms=max_duration_ms,
        dt_ms=dt_ms,
        noise_std=noise_std,
        decision_time_ms=decision_time_ms,
        noise_gain=noise_gain,
        threshold_mode=threshold_mode,
        random_seed=random_seed,
    )


def run_model_b_asymptote(
    df: pd.DataFrame,
    *,
    max_duration_ms: float = 1500.0,
    dt_ms: float = 10.0,
    noise_std: float = 0.7,
    decision_time_ms: float = 50.0,
    noise_gain: float = 3.5,
    threshold_mode: str = "participant_block_mean_abs_belief",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run Model B (continuous asymptote variant)."""
    return _run_continuous_model(
        df,
        model_name="cont_asymptote",
        stop_on_sat=True,
        max_duration_ms=max_duration_ms,
        dt_ms=dt_ms,
        noise_std=noise_std,
        decision_time_ms=decision_time_ms,
        noise_gain=noise_gain,
        threshold_mode=threshold_mode,
        random_seed=random_seed,
    )
