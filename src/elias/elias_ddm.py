from __future__ import annotations

import argparse
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Run this file directly via `python src/elias/elias_ddm.py`.
try:
    from evan_and_old_code_snapshots.evan.glaze import psi_function, simulate_trial
except ModuleNotFoundError:
    elias_src_root = Path(__file__).resolve().parent
    if str(elias_src_root) not in sys.path:
        sys.path.insert(0, str(elias_src_root))
    from evan_and_old_code_snapshots.evan.glaze import psi_function, simulate_trial


EPSILON = 1e-9

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "block_id",
    "trial_index",
    "hazard_rate",
    "noise_sigma",
    "LLR",
    "choice",
    "correct_side",
    "reaction_time_ms",
    "belief_L",
    "subjective_h_snapshot",
)

NUMERIC_INPUT_COLUMNS: tuple[str, ...] = (
    "block_id",
    "trial_index",
    "hazard_rate",
    "noise_sigma",
    "LLR",
    "choice",
    "correct_side",
    "reaction_time_ms",
    "belief_L",
    "subjective_h_snapshot",
)

MODEL_READY_COLUMNS: tuple[str, ...] = (
    "row_id",
    "participant_id",
    "block_id",
    "trial_index",
    "LLR",
    "H",
    "choice",
    "reaction_time_ms",
    "belief_L",
    "prev_observed_belief_L",
)


def _resolve_csv_path(csv_path: str | Path) -> Path:
    """Resolve a CSV path from cwd or repository root.

    Args:
        csv_path: CSV path to resolve. Can be absolute or relative.

    Returns:
        A resolved path that exists on disk.

    Raises:
        FileNotFoundError: If the path cannot be resolved to an existing file.
    """
    path = Path(csv_path)
    if path.exists():
        return path

    if not path.is_absolute():
        repo_root_candidate = Path(__file__).resolve().parents[2] / path
        if repo_root_candidate.exists():
            return repo_root_candidate

    raise FileNotFoundError(f"Could not find CSV file: {csv_path}")


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Validate that a DataFrame contains all required columns.

    Args:
        df: Input DataFrame to validate.
        required_columns: Required column names.
        context: Short context string used in error messages.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric dtype.

    Args:
        df: DataFrame containing columns to convert.
        columns: Column names to coerce with `pd.to_numeric`.

    Returns:
        The same DataFrame with requested columns converted. Invalid parses
        become `NaN`.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_non_finite_rows(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Drop rows containing non-finite numeric values.

    Args:
        df: Input DataFrame.
        numeric_columns: Columns that must be finite.
        context: Short context string used in warning messages.

    Returns:
        A filtered DataFrame. If non-finite rows are found, returns a copied
        DataFrame with those rows removed.
    """
    cols = list(numeric_columns)
    finite_mask = np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite_mask).sum())

    if dropped > 0:
        warnings.warn(
            f"Dropping {dropped} rows with non-finite numeric values during {context}.",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df.loc[finite_mask].copy()

    return df


def _validate_reset_on(reset_on: tuple[str, ...]) -> tuple[str, ...]:
    """Validate reset policy flags.

    Args:
        reset_on: Reset policy tuple. Allowed values are `"participant"` and
            `"block"`.

    Returns:
        The validated reset policy tuple.

    Raises:
        ValueError: If `reset_on` contains unsupported values.
    """
    allowed = {"participant", "block"}
    invalid = sorted(set(reset_on) - allowed)
    if invalid:
        raise ValueError(
            f"Invalid reset_on values: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return reset_on


def load_participant_data(
    csv_path: str | Path = "data/participants.csv",
    participant_ids: list[str] | None = None,
    hazard_col: str = "subjective_h_snapshot",
    reset_on: tuple[str, ...] = ("participant", "block"),
) -> pd.DataFrame:
    """Load participant data and derive model-ready state columns.

    Args:
        csv_path: Path to the participant CSV.
        participant_ids: Optional participant filter. If None, use all participants.
        hazard_col: Column to use as per-trial hazard input `H`.
        reset_on: Reset conditions for sequence state. Allowed values are
            "participant" and "block".

    Returns:
        DataFrame sorted by participant/block/trial with derived columns:
            - H
            - reset_state
            - prev_observed_belief_L
            - row_id

    Raises:
        FileNotFoundError: If CSV path cannot be resolved.
        ValueError: If required columns are missing, filters remove all rows, or
            reset/hazard inputs are invalid.
    """
    reset_on = _validate_reset_on(reset_on)

    resolved_csv_path = _resolve_csv_path(csv_path)
    df = pd.read_csv(resolved_csv_path)

    _validate_required_columns(df, REQUIRED_INPUT_COLUMNS, context="data loading")

    if hazard_col not in df.columns:
        raise ValueError(
            f"Invalid hazard_col '{hazard_col}'. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str)

    if participant_ids is not None:
        participant_ids_str = {str(pid) for pid in participant_ids}
        df = df[df["participant_id"].isin(participant_ids_str)].copy()
        if df.empty:
            raise ValueError(
                f"No rows remaining after participant filter {sorted(participant_ids_str)}"
            )

    df = _coerce_numeric_columns(df, NUMERIC_INPUT_COLUMNS)

    cols_to_check = list(NUMERIC_INPUT_COLUMNS)
    if hazard_col not in cols_to_check:
        cols_to_check.append(hazard_col)

    df = _drop_non_finite_rows(df, cols_to_check, context="data loading")
    if df.empty:
        raise ValueError("All rows were dropped due to non-finite values.")

    df = df.sort_values(["participant_id", "block_id", "trial_index"]).reset_index(
        drop=True
    )

    df["H"] = pd.to_numeric(df[hazard_col], errors="coerce")
    df = _drop_non_finite_rows(df, ["H"], context="hazard assignment")

    participant_changed = df["participant_id"] != df["participant_id"].shift(1)
    block_changed = (df["block_id"] != df["block_id"].shift(1)) | participant_changed

    reset_state = pd.Series(False, index=df.index)
    if "participant" in reset_on:
        reset_state |= participant_changed
    if "block" in reset_on:
        reset_state |= block_changed

    if not df.empty:
        reset_state.iloc[0] = True

    df["reset_state"] = reset_state.to_numpy(dtype=bool)

    df["prev_observed_belief_L"] = df["belief_L"].shift(1)
    df.loc[df["reset_state"], "prev_observed_belief_L"] = 0.0
    df["prev_observed_belief_L"] = (
        pd.to_numeric(df["prev_observed_belief_L"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    df["row_id"] = np.arange(len(df), dtype=int)

    return df.reset_index(drop=True)


def _prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize model input DataFrame.

    Args:
        df: Input DataFrame expected to contain `MODEL_READY_COLUMNS`.

    Returns:
        A normalized DataFrame sorted by participant/block/trial with numeric
        columns coerced and non-finite rows removed.

    Raises:
        ValueError: If required model columns are missing.
    """
    _validate_required_columns(df, MODEL_READY_COLUMNS, context="model simulation")

    model_df = df.copy()
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df = _coerce_numeric_columns(
        model_df,
        ["row_id", "block_id", "trial_index", "LLR", "H", "choice", "reaction_time_ms", "belief_L", "prev_observed_belief_L"],
    )
    model_df = _drop_non_finite_rows(
        model_df,
        ["row_id", "block_id", "trial_index", "LLR", "H", "choice", "reaction_time_ms", "belief_L", "prev_observed_belief_L"],
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
    """Attach per participant-block thresholds.

    Args:
        df: Model input DataFrame.
        threshold_mode: Threshold policy identifier.

    Returns:
        DataFrame with an added `used_threshold` column.

    Raises:
        ValueError: If `threshold_mode` is unsupported.
    """
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
    """Temporarily set and restore NumPy global RNG state.

    Args:
        seed: Temporary seed value applied inside the context.

    Yields:
        None.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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
    """Run one continuous-model variant using `simulate_trial`.

    Args:
        df: Model-ready input DataFrame.
        model_name: Output model label.
        stop_on_sat: Whether to use saturation-based stopping.
        max_duration_ms: Maximum trial duration in milliseconds.
        dt_ms: Integration step size in milliseconds.
        noise_std: Noise standard deviation for accumulation dynamics.
        decision_time_ms: Minimum decision time in milliseconds.
        noise_gain: Multiplicative gain applied to noise.
        threshold_mode: Threshold policy identifier.
        random_seed: Seed for deterministic simulation.

    Returns:
        Standardized per-trial predictions and metadata for the model.

    Raises:
        ValueError: If `max_duration_ms` or `dt_ms` are non-positive.
    """
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
                dt_ms=float(dt_ms),
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
    """Run Model A (continuous threshold variant).

    Args:
        df: Model-ready input DataFrame.
        max_duration_ms: Maximum trial duration in milliseconds.
        dt_ms: Integration step size in milliseconds.
        noise_std: Noise standard deviation for accumulation dynamics.
        decision_time_ms: Minimum decision time in milliseconds.
        noise_gain: Multiplicative gain applied to noise.
        threshold_mode: Threshold policy identifier.
        random_seed: Seed for deterministic simulation.

    Returns:
        Standardized per-trial predictions for Model A.
    """
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
    """Run Model B (continuous asymptote variant).

    Args:
        df: Model-ready input DataFrame.
        max_duration_ms: Maximum trial duration in milliseconds.
        dt_ms: Integration step size in milliseconds.
        noise_std: Noise standard deviation for accumulation dynamics.
        decision_time_ms: Minimum decision time in milliseconds.
        noise_gain: Multiplicative gain applied to noise.
        threshold_mode: Threshold policy identifier.
        random_seed: Seed for deterministic simulation.

    Returns:
        Standardized per-trial predictions for Model B.
    """
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


def _sigmoid(x: float) -> float:
    """Compute a numerically stable sigmoid.

    Args:
        x: Input scalar.

    Returns:
        Sigmoid-transformed scalar in `(0, 1)`.
    """
    clipped = np.clip(x, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _simulate_ddm_single_sample(
    *,
    v: float,
    a: float,
    z: float,
    dt_ms: float,
    max_duration_ms: float,
    diffusion_sigma: float,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    """Simulate one DDM trajectory until boundary crossing or timeout.

    Args:
        v: Drift rate.
        a: Symmetric boundary magnitude.
        z: Relative start point in `[0, 1]`.
        dt_ms: Integration step size in milliseconds.
        max_duration_ms: Timeout duration in milliseconds.
        diffusion_sigma: Diffusion noise scale.
        rng: NumPy random generator used for diffusion noise.

    Returns:
        Tuple of `(decision, rt_ms, terminal_x)` where:
            - `decision` is `1`, `-1`, or `0` (timeout),
            - `rt_ms` is decision time in milliseconds (without non-decision time),
            - `terminal_x` is the final decision variable value.
    """
    dt_sec = dt_ms / 1000.0
    sqrt_dt_sec = np.sqrt(dt_sec)

    # Map z from [0, 1] to [-a, a]
    x = (2.0 * z - 1.0) * a
    time_current_ms = 0.0

    while time_current_ms < max_duration_ms:
        x += v * dt_sec + diffusion_sigma * sqrt_dt_sec * float(rng.standard_normal())
        time_current_ms += dt_ms

        if x >= a:
            return 1, time_current_ms, x
        if x <= -a:
            return -1, time_current_ms, x

    return 0, float(max_duration_ms), x


def run_model_c_ddm(
    df: pd.DataFrame,
    *,
    n_samples_per_trial: int = 200,
    dt_ms: float = 5.0,
    max_duration_ms: float = 1500.0,
    boundary_a: float = 1.0,
    non_decision_time_ms: float = 200.0,
    llr_to_drift_scale: float = 1.0,
    start_k: float = 0.1,
    diffusion_sigma: float = 1.0,
    include_rt_samples: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run Model C (DDM driven by DNM trial signals).

    Mapping:
        Psi_t = psi_function(prev_observed_belief_L_t, H_t)
        z_t = sigmoid(start_k * Psi_t)
        v_t = llr_to_drift_scale * LLR_t

    Args:
        df: Model-ready input DataFrame.
        n_samples_per_trial: Monte Carlo samples per trial.
        dt_ms: Integration step size in milliseconds.
        max_duration_ms: Timeout duration in milliseconds.
        boundary_a: Symmetric boundary magnitude.
        non_decision_time_ms: Non-decision time added to sampled RT.
        llr_to_drift_scale: Scale from LLR to drift rate.
        start_k: Scale from Psi to start-point bias.
        diffusion_sigma: Diffusion noise scale.
        include_rt_samples: Whether to include raw RT samples in output.
        random_seed: Seed for deterministic simulation.

    Returns:
        Standardized per-trial predictions and RT summary statistics.

    Raises:
        ValueError: If sample count or numeric simulation parameters are invalid.
    """
    if n_samples_per_trial <= 0:
        raise ValueError("n_samples_per_trial must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be > 0")
    if boundary_a <= 0:
        raise ValueError("boundary_a must be > 0")
    if diffusion_sigma <= 0:
        raise ValueError("diffusion_sigma must be > 0")

    model_df = _prepare_model_input(df)
    rng = np.random.default_rng(random_seed)

    results: list[dict[str, object]] = []

    for row in model_df.itertuples(index=False):
        psi_t = float(psi_function(float(row.prev_observed_belief_L), float(row.H)))
        z_t = _sigmoid(float(start_k) * psi_t)
        z_t = float(np.clip(z_t, EPSILON, 1.0 - EPSILON))
        v_t = float(llr_to_drift_scale) * float(row.LLR)

        decisions = np.zeros(n_samples_per_trial, dtype=int)
        rts_ms = np.zeros(n_samples_per_trial, dtype=float)

        for i in range(n_samples_per_trial):
            decision, rt_ms, _ = _simulate_ddm_single_sample(
                v=v_t,
                a=float(boundary_a),
                z=z_t,
                dt_ms=float(dt_ms),
                max_duration_ms=float(max_duration_ms),
                diffusion_sigma=float(diffusion_sigma),
                rng=rng,
            )
            decisions[i] = int(decision)
            rts_ms[i] = float(rt_ms + non_decision_time_ms)

        prob_pos = float(np.mean(decisions == 1))
        prob_neg = float(np.mean(decisions == -1))
        timeout_rate = float(np.mean(decisions == 0))

        if prob_pos == 0.0 and prob_neg == 0.0:
            predicted_decision = 0
        elif prob_pos >= prob_neg:
            predicted_decision = 1
        else:
            predicted_decision = -1

        q10, q50, q90 = np.quantile(rts_ms, [0.10, 0.50, 0.90])

        row_dict: dict[str, object] = {
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
            "used_threshold": np.nan,
            "predicted_decision": int(predicted_decision),
            "predicted_rt_ms": float(np.median(rts_ms)),
            "predicted_belief": float(psi_t),
            "pred_decision_prob_pos": prob_pos,
            "pred_decision_prob_neg": prob_neg,
            "pred_timeout_rate": timeout_rate,
            "pred_rt_mean_ms": float(np.mean(rts_ms)),
            "pred_rt_std_ms": float(np.std(rts_ms)),
            "pred_rt_q10_ms": float(q10),
            "pred_rt_q50_ms": float(q50),
            "pred_rt_q90_ms": float(q90),
            "model_name": "ddm_dnm",
            "param_n_samples_per_trial": int(n_samples_per_trial),
            "param_dt_ms": float(dt_ms),
            "param_max_duration_ms": float(max_duration_ms),
            "param_boundary_a": float(boundary_a),
            "param_non_decision_time_ms": float(non_decision_time_ms),
            "param_llr_to_drift_scale": float(llr_to_drift_scale),
            "param_start_k": float(start_k),
            "param_diffusion_sigma": float(diffusion_sigma),
            "param_random_seed": int(random_seed),
        }

        if include_rt_samples:
            row_dict["pred_rt_samples_ms"] = rts_ms.copy()

        results.append(row_dict)

    return pd.DataFrame(results)


def run_all_models_for_participant(
    df: pd.DataFrame,
    participant_id: str,
    *,
    random_seed: int = 42,
    n_samples_per_trial: int = 200,
    include_rt_samples: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run models A, B, and C for one participant.

    Args:
        df: DataFrame containing one or more participants.
        participant_id: Participant identifier to simulate.
        random_seed: Seed for deterministic simulation.
        n_samples_per_trial: Monte Carlo samples per trial for Model C.
        include_rt_samples: Whether to include raw RT samples for Model C.

    Returns:
        Dictionary mapping model names to standardized output DataFrames.

    Raises:
        ValueError: If `participant_id` is absent from `df`.
    """
    _validate_required_columns(df, ["participant_id"], context="orchestration")

    participant_id_str = str(participant_id)
    subset = df[df["participant_id"].astype(str) == participant_id_str].copy()
    if subset.empty:
        available = sorted(df["participant_id"].astype(str).unique().tolist())
        raise ValueError(
            f"No rows for participant_id='{participant_id_str}'. Available: {available}"
        )

    out_a = run_model_a_threshold(subset, random_seed=random_seed)
    out_b = run_model_b_asymptote(subset, random_seed=random_seed)
    out_c = run_model_c_ddm(
        subset,
        n_samples_per_trial=n_samples_per_trial,
        include_rt_samples=include_rt_samples,
        random_seed=random_seed,
    )

    return {
        "cont_threshold": out_a,
        "cont_asymptote": out_b,
        "ddm_dnm": out_c,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the smoke-test demo.

    Returns:
        Configured argument parser for module CLI execution.
    """
    parser = argparse.ArgumentParser(description="Run participant-wise model simulations.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/participants.csv",
        help="Path to participant CSV file.",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default="P01",
        help="Participant id to simulate in demo mode.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="DDM Monte Carlo samples per trial.",
    )
    parser.add_argument(
        "--include-rt-samples",
        action="store_true",
        help="Include raw DDM RT sample arrays in output.",
    )
    parser.add_argument(
        "--hazard-col",
        type=str,
        default="subjective_h_snapshot",
        help="Column to use as hazard input H.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic simulations.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    loaded_df = load_participant_data(
        csv_path=args.csv_path,
        participant_ids=[args.participant_id],
        hazard_col=args.hazard_col,
        reset_on=("participant", "block"),
    )

    outputs = run_all_models_for_participant(
        loaded_df,
        participant_id=args.participant_id,
        random_seed=args.seed,
        n_samples_per_trial=args.samples,
        include_rt_samples=args.include_rt_samples,
    )

    print(
        f"Loaded {len(loaded_df)} rows for participant {args.participant_id}. "
        f"Running models: {', '.join(outputs.keys())}"
    )

    for model_name, model_df in outputs.items():
        mean_rt = float(model_df["predicted_rt_ms"].mean()) if not model_df.empty else float("nan")
        timeout_rate = (
            float((model_df["predicted_decision"] == 0).mean())
            if not model_df.empty
            else float("nan")
        )
        print(
            f"[{model_name}] rows={len(model_df)} "
            f"mean_predicted_rt_ms={mean_rt:.2f} "
            f"timeout_rate={timeout_rate:.3f}"
        )
