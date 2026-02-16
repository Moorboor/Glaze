# Objective environment generator for surrogate simulations.
# Main function: generate_environment_from_template.

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_validation import _validate_required_columns

# The old app uses centers 0.25 and 0.75 on a [0,1] axis.
_LEFT_MEAN = 0.25
_RIGHT_MEAN = 0.75
_CENTER = 0.5
_SCREEN_MIN = 0.05
_SCREEN_MAX = 0.95
_MIN_ABS_DISTANCE = 0.01


def _sigma_ratio_to_absolute_sigma(
    sigma_ratio: float | np.ndarray,
) -> float | np.ndarray:
    """Map app-level sigma ratio to absolute x-axis sigma."""
    return np.asarray(sigma_ratio, dtype=float) * float(_RIGHT_MEAN - _LEFT_MEAN)


def _llr_from_signed_distance(
    signed_distance: np.ndarray,
    *,
    sigma_ratio: np.ndarray,
) -> np.ndarray:
    """Compute LLR from signed center distance under symmetric Gaussians."""
    # For means at +/- m around center, LLR(d) = (2*m*d) / var.
    m = float(_RIGHT_MEAN - _CENTER)
    abs_sigma = _sigma_ratio_to_absolute_sigma(np.asarray(sigma_ratio, dtype=float))
    variance = np.maximum(abs_sigma**2, 1e-12)
    return (2.0 * m * np.asarray(signed_distance, dtype=float)) / variance


def objective_h_mean_from_template(
    df_template: pd.DataFrame,
    *,
    objective_h_col: str = "hazard_rate",
) -> float:
    """Return template mean objective hazard (fallback-safe)."""
    _validate_required_columns(df_template, (objective_h_col,), context="environment template")
    vals = pd.to_numeric(df_template[objective_h_col], errors="coerce")
    finite = vals[np.isfinite(vals.to_numpy(dtype=float))]
    if finite.empty:
        return 0.1
    return float(np.clip(float(finite.mean()), 1e-6, 1.0 - 1e-6))


def generate_environment_from_template(
    df_template: pd.DataFrame,
    *,
    random_seed: int,
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    trial_col: str = "trial_index",
    objective_h_col: str = "hazard_rate",
    noise_sigma_col: str = "noise_sigma",
) -> pd.DataFrame:
    """Generate hidden states and signed-distance evidence on existing row layout.

    Important guarantees:
    - Preserves the exact `(participant_id, block_id, trial_index)` structure.
    - Does not add or remove trials (Evan short block remains short).
    """
    _validate_required_columns(
        df_template,
        (participant_col, block_col, trial_col, objective_h_col, noise_sigma_col),
        context="environment generation",
    )

    out = df_template.copy()
    out[participant_col] = out[participant_col].astype(str)
    out[block_col] = pd.to_numeric(out[block_col], errors="coerce")
    out[trial_col] = pd.to_numeric(out[trial_col], errors="coerce")
    out[objective_h_col] = pd.to_numeric(out[objective_h_col], errors="coerce")
    out[noise_sigma_col] = pd.to_numeric(out[noise_sigma_col], errors="coerce")

    finite_mask = np.isfinite(
        out[[block_col, trial_col, objective_h_col, noise_sigma_col]].to_numpy(dtype=float)
    ).all(axis=1)
    if not bool(np.all(finite_mask)):
        n_bad = int((~finite_mask).sum())
        raise ValueError(f"Found {n_bad} non-finite rows in environment generation inputs.")

    # Preserve original row order and only overwrite environment-derived columns.
    out = out.reset_index(names="_orig_index")
    out = out.sort_values([participant_col, block_col, trial_col], kind="mergesort").reset_index(
        drop=True
    )

    rng = np.random.default_rng(int(random_seed))

    true_state = np.zeros(len(out), dtype=int)
    signed_distance = np.zeros(len(out), dtype=float)

    for (_, _), chunk in out.groupby([participant_col, block_col], sort=False):
        pos = chunk.index.to_numpy(dtype=int)
        h_vals = np.clip(
            chunk[objective_h_col].to_numpy(dtype=float),
            1e-6,
            1.0 - 1e-6,
        )
        sigma_ratio_vals = np.maximum(chunk[noise_sigma_col].to_numpy(dtype=float), 1e-6)

        # Bernoulli switching process with objective hazard.
        state = 1 if rng.random() < 0.5 else -1
        for local_i, (h_t, sigma_ratio_t) in enumerate(
            zip(h_vals, sigma_ratio_vals, strict=True)
        ):
            if rng.random() < float(h_t):
                state *= -1

            global_i = int(pos[local_i])
            true_state[global_i] = int(state)

            mean_x = _RIGHT_MEAN if state > 0 else _LEFT_MEAN
            sigma_abs = _sigma_ratio_to_absolute_sigma(float(sigma_ratio_t))

            # Keep sampling until the point is not too close to center.
            sampled_x = float(mean_x + rng.normal(loc=0.0, scale=float(sigma_abs)))
            sampled_x = float(np.clip(sampled_x, _SCREEN_MIN, _SCREEN_MAX))
            guard = 0
            while abs(sampled_x - _CENTER) < _MIN_ABS_DISTANCE and guard < 64:
                sampled_x = float(mean_x + rng.normal(loc=0.0, scale=float(sigma_abs)))
                sampled_x = float(np.clip(sampled_x, _SCREEN_MIN, _SCREEN_MAX))
                guard += 1

            signed_distance[global_i] = float(sampled_x - _CENTER)

    out["true_state"] = true_state.astype(int)
    out["signed_distance_from_center"] = signed_distance.astype(float)

    if "LLR" in out.columns:
        out["LLR_observed_legacy"] = pd.to_numeric(out["LLR"], errors="coerce")

    out["LLR"] = _llr_from_signed_distance(
        out["signed_distance_from_center"].to_numpy(dtype=float),
        sigma_ratio=out[noise_sigma_col].to_numpy(dtype=float),
    )

    out = out.sort_values("_orig_index", kind="mergesort").drop(columns=["_orig_index"])
    return out.reset_index(drop=True)
