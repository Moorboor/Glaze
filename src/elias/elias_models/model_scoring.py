"""Simulation and likelihood scoring for Elias model comparison.

Main Entry Point:
    - score_model_simulation_likelihood: Pool trial scorer for fitting and evaluation.

Continuous Model Simulators (A/B):
    - _simulate_continuous_trials_for_likelihood: Monte Carlo sampler.
    - _attach_thresholds: Per-block threshold builder.

DDM Model Simulator (C):
    - _simulate_ddm_trials_for_likelihood: Monte Carlo sampler.
    - _simulate_ddm_single_sample: Single trajectory simulator.
    - _sigmoid: Stable logistic for start-point transform.

Input Preparation:
    - _prepare_model_input: Schema normalization for trial frames.

RT Density Estimation:
    - _estimate_rt_density_at_observed_value: Histogram-based conditional density.
    - _build_rt_bin_edges: Fixed-width bin constructor.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Scoring constants are local now that legacy constants.py is removed.
EPSILON = 1e-9
SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "cont_threshold",
    "cont_asymptote",
    "ddm_dnm",
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
    "prev_normative_belief_L",
)

try:
    # Keep Evan's backend untouched and consume it as-is.
    from evan.glaze import psi_function, simulate_trial
except ModuleNotFoundError:
    # Allow direct module runs where `src/` is not yet on PYTHONPATH.
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from evan.glaze import psi_function, simulate_trial


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Validate required columns for scoring internals."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric dtype with NaN for parse errors."""
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _drop_non_finite_rows(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Drop rows with non-finite values in required numeric columns."""
    columns = list(numeric_columns)
    finite_mask = np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped > 0:
        # Keep scoring robust by filtering bad rows while surfacing a warning.
        print(f"[model_scoring] dropped {dropped} non-finite rows during {context}.")
    return df.loc[finite_mask].copy()


def _normalize_choice_values_to_pm1(choice_values: np.ndarray) -> np.ndarray:
    """Normalize choices to signed coding `-1/+1` (mapping `0 -> -1`)."""
    raw = np.asarray(choice_values, dtype=float)
    valid_mask = np.isin(raw, (-1.0, 0.0, 1.0))
    if not bool(np.all(valid_mask)):
        invalid_values = np.unique(raw[~valid_mask]).tolist()
        raise ValueError(
            "Unsupported choice encoding for scoring. Expected {-1,0,1}; "
            f"found invalid values: {invalid_values}"
        )
    normalized = raw.copy()
    normalized[normalized == 0.0] = -1.0
    return normalized.astype(int)


def _prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize model input rows for scoring."""
    _validate_required_columns(df, MODEL_READY_COLUMNS, context="model scoring")

    model_df = _coerce_numeric_columns(
        df,
        (
            "row_id",
            "block_id",
            "trial_index",
            "LLR",
            "H",
            "choice",
            "reaction_time_ms",
            "belief_L",
            "prev_normative_belief_L",
        ),
    )
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df = _drop_non_finite_rows(
        model_df,
        (
            "row_id",
            "block_id",
            "trial_index",
            "LLR",
            "H",
            "choice",
            "reaction_time_ms",
            "belief_L",
            "prev_normative_belief_L",
        ),
        context="model scoring",
    )

    # Deterministic ordering makes stochastic scoring reproducible under a fixed seed.
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
    """Attach one threshold value per row for continuous model simulation."""
    if threshold_mode != "participant_block_mean_abs_belief":
        raise ValueError(
            f"Unsupported threshold_mode '{threshold_mode}'. "
            "Only 'participant_block_mean_abs_belief' is supported."
        )

    if use_block_sidecar_params and block_threshold_map:
        sidecar_rows: list[dict[str, float]] = []
        for block_id_raw, threshold_raw in block_threshold_map.items():
            try:
                block_id = int(block_id_raw)
                threshold_value = float(threshold_raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(threshold_value):
                sidecar_rows.append(
                    {
                        "block_id": block_id,
                        "used_threshold": max(float(threshold_value), EPSILON),
                    }
                )

        if sidecar_rows:
            sidecar_df = pd.DataFrame(sidecar_rows).drop_duplicates(
                subset=["block_id"],
                keep="last",
            )
            with_sidecar = df.merge(sidecar_df, on="block_id", how="left")
            if with_sidecar["used_threshold"].notna().any():
                # Fill missing sidecar blocks with participant-block defaults.
                source_col = "normative_belief_L" if "normative_belief_L" in df.columns else "belief_L"
                defaults = (
                    df.groupby(["participant_id", "block_id"], sort=False)[source_col]
                    .agg(lambda x: max(float(np.mean(np.abs(x))), EPSILON))
                    .rename("default_threshold")
                    .reset_index()
                )
                with_sidecar = with_sidecar.merge(
                    defaults,
                    on=["participant_id", "block_id"],
                    how="left",
                )
                with_sidecar["used_threshold"] = (
                    with_sidecar["used_threshold"]
                    .fillna(with_sidecar["default_threshold"])
                    .fillna(EPSILON)
                    .clip(lower=EPSILON)
                )
                return with_sidecar.drop(columns=["default_threshold"])

    source_col = "normative_belief_L" if "normative_belief_L" in df.columns else "belief_L"
    thresholds = (
        df.groupby(["participant_id", "block_id"], sort=False)[source_col]
        .agg(lambda x: max(float(np.mean(np.abs(x))), EPSILON))
        .rename("used_threshold")
        .reset_index()
    )
    out = df.merge(thresholds, on=["participant_id", "block_id"], how="left")
    out["used_threshold"] = out["used_threshold"].fillna(EPSILON).clip(lower=EPSILON)
    return out


@contextmanager
def _temporary_numpy_seed(seed: int):
    """Temporarily set NumPy global seed and restore prior state on exit."""
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _coerce_simulated_decision(raw_decision: float | int) -> int:
    """Coerce simulator decision output into signed discrete coding."""
    if float(raw_decision) > 0.0:
        return 1
    if float(raw_decision) < 0.0:
        return -1
    return 0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for DDM start-point transform."""
    clipped = np.clip(float(x), -60.0, 60.0)
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
    """Simulate one DDM trajectory until boundary hit or timeout."""
    dt_sec = float(dt_ms) / 1000.0
    sqrt_dt_sec = np.sqrt(dt_sec)

    # Map start fraction z in [0, 1] to evidence axis [-a, +a].
    evidence = (2.0 * float(z) - 1.0) * float(a)
    elapsed_ms = 0.0

    while elapsed_ms < float(max_duration_ms):
        # Euler-Maruyama update: drift term plus Gaussian diffusion noise.
        evidence += float(v) * dt_sec + float(diffusion_sigma) * sqrt_dt_sec * float(
            rng.standard_normal()
        )
        elapsed_ms += float(dt_ms)

        if evidence >= float(a):
            return 1, float(elapsed_ms), float(evidence)
        if evidence <= -float(a):
            return -1, float(elapsed_ms), float(evidence)

    return 0, float(max_duration_ms), float(evidence)


def _build_rt_bin_edges(rt_max_ms: float, rt_bin_width_ms: float) -> np.ndarray:
    """Construct fixed-width RT histogram bin edges in milliseconds."""
    if float(rt_max_ms) <= 0.0:
        raise ValueError("rt_max_ms must be > 0")
    if float(rt_bin_width_ms) <= 0.0:
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
    """Estimate conditional RT density at an observed RT via smoothed histogram."""
    if not np.isfinite(observed_rt_ms):
        return float(eps)
    if observed_rt_ms < float(rt_bin_edges[0]) or observed_rt_ms > float(rt_bin_edges[-1]):
        return float(eps)

    samples = np.asarray(rt_samples_ms, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float(eps)

    counts, _ = np.histogram(samples, bins=rt_bin_edges)
    # Add pseudo-counts so empty bins do not yield zero probability.
    smoothed_counts = counts.astype(float) + float(eps)
    bin_width_ms = float(rt_bin_edges[1] - rt_bin_edges[0])
    densities = smoothed_counts / (np.sum(smoothed_counts) * bin_width_ms)

    bin_index = int(np.searchsorted(rt_bin_edges, observed_rt_ms, side="right") - 1)
    bin_index = int(np.clip(bin_index, 0, len(densities) - 1))
    return float(max(densities[bin_index], float(eps)))


def _simulate_continuous_trials_for_likelihood(
    model_df: pd.DataFrame,
    *,
    stop_on_sat: bool,
    n_sims_per_trial: int,
    min_duration_ms: float,
    max_duration_ms: float,
    dt_ms: float,
    noise_std: float,
    decision_time_ms: float,
    noise_gain: float,
    threshold_mode: str,
    block_threshold_map: dict[int, float] | None,
    use_block_sidecar_params: bool,
    random_seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample trial-wise decision/RT distributions for continuous models A/B."""
    if int(n_sims_per_trial) <= 0:
        raise ValueError("n_sims_per_trial must be > 0")
    if float(dt_ms) <= 0.0:
        raise ValueError("dt_ms must be > 0")
    if float(min_duration_ms) < 0.0:
        raise ValueError("min_duration_ms must be >= 0")
    if float(max_duration_ms) <= 0.0:
        raise ValueError("max_duration_ms must be > 0")
    if float(min_duration_ms) > float(max_duration_ms):
        raise ValueError("min_duration_ms must be <= max_duration_ms")

    thresholded_df = _attach_thresholds(
        model_df,
        threshold_mode=threshold_mode,
        block_threshold_map=block_threshold_map,
        use_block_sidecar_params=bool(use_block_sidecar_params),
    )

    # When explicit block sidecars are used for Model B, avoid simulator overwrite.
    effective_stop_on_sat = bool(stop_on_sat)
    if bool(use_block_sidecar_params) and isinstance(block_threshold_map, dict) and block_threshold_map:
        effective_stop_on_sat = False

    dt_sec = float(dt_ms) / 1000.0
    decisions_by_trial: list[np.ndarray] = []
    rts_by_trial: list[np.ndarray] = []

    with _temporary_numpy_seed(int(random_seed)):
        for row in thresholded_df.itertuples(index=False):
            decisions = np.zeros(int(n_sims_per_trial), dtype=int)
            rts_ms = np.zeros(int(n_sims_per_trial), dtype=float)
            for sample_index in range(int(n_sims_per_trial)):
                sim_result = simulate_trial(
                    prev_belief_L=float(row.prev_normative_belief_L),
                    current_LLR=float(row.LLR),
                    H=float(row.H),
                    belief_threshold=float(row.used_threshold),
                    max_duration_ms=float(max_duration_ms),
                    dt=float(dt_sec),
                    noise_std=float(noise_std),
                    decision_time_ms=float(decision_time_ms),
                    noise_gain=float(noise_gain),
                    stop_on_sat=bool(effective_stop_on_sat),
                )
                decisions[sample_index] = _coerce_simulated_decision(sim_result["decision"])
                raw_rt_ms = float(sim_result["reaction_time_ms"])
                rts_ms[sample_index] = float(max(raw_rt_ms, float(min_duration_ms)))

            decisions_by_trial.append(decisions)
            rts_by_trial.append(rts_ms)

    return decisions_by_trial, rts_by_trial


def _simulate_ddm_trials_for_likelihood(
    model_df: pd.DataFrame,
    *,
    n_sims_per_trial: int,
    min_duration_ms: float,
    dt_ms: float,
    max_duration_ms: float,
    boundary_a: float,
    non_decision_time_ms: float,
    llr_to_drift_scale: float,
    start_k: float,
    diffusion_sigma: float,
    random_seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample trial-wise decision/RT distributions for DDM model C."""
    if int(n_sims_per_trial) <= 0:
        raise ValueError("n_sims_per_trial must be > 0")
    if float(dt_ms) <= 0.0:
        raise ValueError("dt_ms must be > 0")
    if float(min_duration_ms) < 0.0:
        raise ValueError("min_duration_ms must be >= 0")
    if float(max_duration_ms) <= 0.0:
        raise ValueError("max_duration_ms must be > 0")
    if float(boundary_a) <= 0.0:
        raise ValueError("boundary_a must be > 0")
    if float(diffusion_sigma) <= 0.0:
        raise ValueError("diffusion_sigma must be > 0")

    rng = np.random.default_rng(int(random_seed))
    decisions_by_trial: list[np.ndarray] = []
    rts_by_trial: list[np.ndarray] = []

    for row in model_df.itertuples(index=False):
        # Build trial-specific start bias and drift from normative prior + evidence.
        psi_t = float(psi_function(float(row.prev_normative_belief_L), float(row.H)))
        z_t = _sigmoid(float(start_k) * psi_t)
        z_t = float(np.clip(z_t, EPSILON, 1.0 - EPSILON))
        v_t = float(llr_to_drift_scale) * float(row.LLR)

        decisions = np.zeros(int(n_sims_per_trial), dtype=int)
        rts_ms = np.zeros(int(n_sims_per_trial), dtype=float)
        for sample_index in range(int(n_sims_per_trial)):
            raw_decision, decision_rt_ms, _ = _simulate_ddm_single_sample(
                v=v_t,
                a=float(boundary_a),
                z=z_t,
                dt_ms=float(dt_ms),
                max_duration_ms=float(max_duration_ms),
                diffusion_sigma=float(diffusion_sigma),
                rng=rng,
            )
            decisions[sample_index] = _coerce_simulated_decision(raw_decision)
            raw_rt_ms = float(decision_rt_ms + float(non_decision_time_ms))
            rts_ms[sample_index] = float(max(raw_rt_ms, float(min_duration_ms)))

        decisions_by_trial.append(decisions)
        rts_by_trial.append(rts_ms)

    return decisions_by_trial, rts_by_trial


def score_model_simulation_likelihood(
    df: pd.DataFrame,
    *,
    model_name: str,
    model_params: dict[str, float | int | str | dict[int, float]] | None = None,
    n_sims_per_trial: int = 2000,
    rt_bin_width_ms: float = 20.0,
    rt_max_ms: float = 5000.0,
    eps: float = 1e-12,
    random_seed: int = 0,
) -> dict[str, object]:
    """Score model likelihood by Monte Carlo choice and conditional RT terms."""
    model_name_str = str(model_name)
    if model_name_str not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{model_name_str}'. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    if int(n_sims_per_trial) <= 0:
        raise ValueError("n_sims_per_trial must be > 0")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0")

    params = {} if model_params is None else dict(model_params)
    model_df = _prepare_model_input(df)
    # Normalize once so all models are scored against consistent choice coding.
    model_df["choice"] = _normalize_choice_values_to_pm1(model_df["choice"].to_numpy(dtype=float))

    if model_name_str in ("cont_threshold", "cont_asymptote"):
        use_block_sidecar_params = bool(params.get("use_block_sidecar_params", False))
        block_threshold_map = (
            params.get("threshold_by_block_sidecar", {})
            if model_name_str == "cont_threshold"
            else params.get("asymptote_by_block_sidecar", {})
        )
        simulated_decisions, simulated_rts_ms = _simulate_continuous_trials_for_likelihood(
            model_df=model_df,
            stop_on_sat=(model_name_str == "cont_asymptote"),
            n_sims_per_trial=int(n_sims_per_trial),
            min_duration_ms=float(params.get("min_duration_ms", 0.0)),
            max_duration_ms=float(params.get("max_duration_ms", 1500.0)),
            dt_ms=float(params.get("dt_ms", 10.0)),
            noise_std=float(params.get("noise_std", 0.7)),
            decision_time_ms=float(params.get("decision_time_ms", 50.0)),
            noise_gain=float(params.get("noise_gain", 3.5)),
            threshold_mode=str(params.get("threshold_mode", "participant_block_mean_abs_belief")),
            block_threshold_map=(
                dict(block_threshold_map) if isinstance(block_threshold_map, dict) else None
            ),
            use_block_sidecar_params=bool(use_block_sidecar_params),
            random_seed=int(random_seed),
        )
    else:
        simulated_decisions, simulated_rts_ms = _simulate_ddm_trials_for_likelihood(
            model_df=model_df,
            n_sims_per_trial=int(n_sims_per_trial),
            min_duration_ms=float(params.get("min_duration_ms", 0.0)),
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

        # Choice likelihood is estimated from Monte Carlo frequency.
        p_choice = float(np.mean(sampled_decisions == observed_choice))
        p_choice = float(max(p_choice, float(eps)))

        # Conditional RT term uses only samples matching the observed choice.
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
