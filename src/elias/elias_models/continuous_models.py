# Continuous-model implementations for Model A (threshold) and Model B (asymptote).
# Main functions: run_model_a_threshold, run_model_b_asymptote.
# Internal helpers validate inputs, attach thresholds, and run `simulate_trial`.

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
    # Enforce a strict column contract so all model runners and scorers consume
    # the exact same schema regardless of upstream caller.
    _validate_required_columns(df, MODEL_READY_COLUMNS, context="model simulation")

    model_df = df.copy()
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    # Convert all model-relevant numeric fields up front to avoid mixed dtypes
    # and silent string concatenation/ordering bugs in simulation loops.
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

    # Sort once and keep deterministic row order to make stochastic runs
    # reproducible when the same random seed is used.
    model_df = model_df.sort_values(["participant_id", "block_id", "trial_index"]).reset_index(
        drop=True
    )
    model_df["row_id"] = model_df["row_id"].astype(int)
    return model_df


def _attach_thresholds(
    df: pd.DataFrame,
    threshold_mode: str,
    *,
    block_threshold_map: dict[int, float] | None = None,
    use_block_sidecar_params: bool = False,
) -> pd.DataFrame:
    """Attach per participant-block thresholds used by continuous simulators.

    Args:
        df: Model-ready trial table.
        threshold_mode: Runtime threshold policy label.
        block_threshold_map: Optional per-block threshold/asymptote values from
            parameter sidecars.
        use_block_sidecar_params: Whether sidecar block values should be consumed.

    Returns:
        DataFrame with one `used_threshold` value per row.
    """
    if threshold_mode != "participant_block_mean_abs_belief":
        raise ValueError(
            f"Unsupported threshold_mode '{threshold_mode}'. "
            "Only 'participant_block_mean_abs_belief' is supported."
        )

    if use_block_sidecar_params and block_threshold_map:
        # When sidecar parameters are enabled we attach explicit block values from
        # fitted parameter payloads. This enables true block-level parameter effects
        # in Step 3/4 scoring instead of only storing those values as metadata.
        sidecar_rows: list[dict[str, float]] = []
        for block_id_raw, threshold_value_raw in block_threshold_map.items():
            try:
                block_id = int(block_id_raw)
                threshold_value = float(threshold_value_raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(threshold_value):
                sidecar_rows.append(
                    {
                        "block_id": int(block_id),
                        "used_threshold": max(float(threshold_value), float(EPSILON)),
                    }
                )

        if sidecar_rows:
            sidecar_df = pd.DataFrame(sidecar_rows).drop_duplicates(subset=["block_id"], keep="last")
            out = df.merge(sidecar_df, on="block_id", how="left")
            if out["used_threshold"].notna().any():
                # Fill any missing blocks with default mode-derived thresholds to keep
                # behavior robust when sidecar maps are partial.
                default_thresholds = (
                    df.groupby(["participant_id", "block_id"], sort=False)["belief_L"]
                    .agg(lambda x: max(float(np.mean(np.abs(x))), EPSILON))
                    .rename("default_threshold")
                    .reset_index()
                )
                out = out.merge(default_thresholds, on=["participant_id", "block_id"], how="left")
                out["used_threshold"] = (
                    out["used_threshold"].fillna(out["default_threshold"]).fillna(EPSILON).clip(lower=EPSILON)
                )
                out = out.drop(columns=["default_threshold"])
                return out

    # For the currently supported mode, threshold is estimated from observed
    # belief magnitude per participant/block and reused for all trials in block.
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
    threshold_by_block_sidecar: dict[int, float] | None,
    asymptote_by_block_sidecar: dict[int, float] | None,
    use_block_sidecar_params: bool,
    random_seed: int,
) -> pd.DataFrame:
    """Run one continuous-model variant using `simulate_trial`."""
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")

    model_df = _prepare_model_input(df)
    # Choose the sidecar map corresponding to the specific continuous model.
    block_threshold_map = (
        threshold_by_block_sidecar if model_name == "cont_threshold" else asymptote_by_block_sidecar
    )
    # This is where Model A/B obtain their decision boundary input at runtime.
    model_df = _attach_thresholds(
        model_df,
        threshold_mode=threshold_mode,
        block_threshold_map=block_threshold_map,
        use_block_sidecar_params=bool(use_block_sidecar_params),
    )

    # Evan's `simulate_trial(..., stop_on_sat=True)` internally overwrites the
    # provided threshold with a dynamic asymptote. To make block sidecar values
    # effective for Model B as requested, we disable that overwrite when sidecar
    # parameters are explicitly enabled.
    effective_stop_on_sat = bool(stop_on_sat)
    if (
        model_name == "cont_asymptote"
        and bool(use_block_sidecar_params)
        and isinstance(block_threshold_map, dict)
        and len(block_threshold_map) > 0
    ):
        effective_stop_on_sat = False

    results: list[dict[str, object]] = []

    with _temporary_numpy_seed(random_seed):
        # Each observed trial gets one forward simulation under the chosen model
        # variant (`stop_on_sat` differs between A and B).
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
                stop_on_sat=bool(effective_stop_on_sat),
            )

            # Keep both observed and predicted values in one row so downstream
            # diagnostics can compute fit metrics without extra joins.
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
                    "param_effective_stop_on_sat": bool(effective_stop_on_sat),
                    "param_max_duration_ms": float(max_duration_ms),
                    "param_dt_ms": float(dt_ms),
                    "param_noise_std": float(noise_std),
                    "param_decision_time_ms": float(decision_time_ms),
                    "param_noise_gain": float(noise_gain),
                    "param_threshold_mode": threshold_mode,
                    "param_use_block_sidecar_params": bool(use_block_sidecar_params),
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
    threshold_by_block_sidecar: dict[int, float] | None = None,
    use_block_sidecar_params: bool = False,
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
        threshold_by_block_sidecar=threshold_by_block_sidecar,
        asymptote_by_block_sidecar=None,
        use_block_sidecar_params=bool(use_block_sidecar_params),
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
    asymptote_by_block_sidecar: dict[int, float] | None = None,
    use_block_sidecar_params: bool = False,
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
        threshold_by_block_sidecar=None,
        asymptote_by_block_sidecar=asymptote_by_block_sidecar,
        use_block_sidecar_params=bool(use_block_sidecar_params),
        random_seed=random_seed,
    )
