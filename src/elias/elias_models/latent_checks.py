"""Step 5 latent-variable and hazard-signature diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .continuous_models import run_model_a_threshold, run_model_b_asymptote
from .data_validation import _validate_required_columns
from .ddm_model import run_model_c_ddm

try:
    from evan.glaze import psi_function
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from evan.glaze import psi_function


_HAZARD_TRIAL_COLUMNS: tuple[str, ...] = (
    "row_id",
    "participant_id",
    "block_id",
    "trial_index",
    "winner_model_name",
    "choice",
    "correct_side",
    "is_choice_correct",
    "reaction_time_ms",
    "H",
    "prev_observed_belief_L",
    "psi_t",
    "abs_prev_belief_L",
    "abs_psi_t",
    "psi_shrinkage_abs",
    "is_change_point",
)

_HAZARD_BLOCK_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "block_id",
    "winner_model_name",
    "n_trials",
    "n_change_points",
    "n_stable_trials",
    "change_point_accuracy",
    "stable_accuracy",
    "change_point_median_rt_ms",
    "stable_median_rt_ms",
    "h_vs_abs_psi_spearman",
    "h_vs_shrinkage_spearman",
    "mean_shrinkage_change",
    "mean_shrinkage_stable",
)

_LATENT_TRIAL_COLUMNS: tuple[str, ...] = (
    "row_id",
    "participant_id",
    "block_id",
    "trial_index",
    "split",
    "winner_model_name",
    "choice",
    "correct_side",
    "predicted_decision",
    "predicted_choice_binary",
    "predicted_timeout",
    "choice_match_excluding_timeout",
    "reaction_time_ms",
    "predicted_rt_ms",
    "belief_L",
    "predicted_belief",
    "H",
    "LLR",
)

_LATENT_BLOCK_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "block_id",
    "winner_model_name",
    "n_trials",
    "n_timeout_predictions",
    "timeout_rate",
    "observed_choice_rate",
    "predicted_choice_rate",
    "choice_accuracy_excluding_timeout",
    "mean_observed_rt_ms",
    "mean_predicted_rt_ms",
    "mae_rt_ms",
    "mean_observed_belief_L",
    "mean_predicted_belief",
    "mae_belief",
    "mean_H",
)

_CONTINUOUS_PARAM_KEYS: tuple[str, ...] = (
    "max_duration_ms",
    "dt_ms",
    "decision_time_ms",
    "noise_gain",
    "threshold_mode",
)

_DDM_PARAM_KEYS: tuple[str, ...] = (
    "dt_ms",
    "max_duration_ms",
    "boundary_a",
    "non_decision_time_ms",
    "llr_to_drift_scale",
    "start_k",
    "diffusion_sigma",
)


def _validate_winner_parameter_table(winner_parameter_table: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(
        winner_parameter_table,
        ("participant_id", "winner_model_name", "best_model_params"),
        context="Step 5 winner parameter table",
    )
    if winner_parameter_table.empty:
        raise ValueError("Step 5 winner parameter table is empty.")

    out = winner_parameter_table.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    out["winner_model_name"] = out["winner_model_name"].astype(str)
    if out["participant_id"].duplicated().any():
        duplicated = sorted(out.loc[out["participant_id"].duplicated(), "participant_id"].unique())
        raise ValueError(
            f"Step 5 winner parameter table contains duplicate participants: {duplicated}."
        )
    for row in out.itertuples(index=False):
        if not isinstance(row.best_model_params, dict):
            raise ValueError(
                "Step 5 winner parameter table requires `best_model_params` to be dict payloads."
            )
    return out


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(paired) < 2:
        return float(np.nan)
    if paired["x"].nunique(dropna=True) < 2 or paired["y"].nunique(dropna=True) < 2:
        return float(np.nan)
    return float(paired["x"].corr(paired["y"], method="spearman"))


def run_change_hazard_checks(
    df_all: pd.DataFrame,
    winner_parameter_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run change-point and hazard-signature checks on TEST rows."""
    _validate_required_columns(
        df_all,
        (
            "row_id",
            "participant_id",
            "block_id",
            "trial_index",
            "split",
            "choice",
            "correct_side",
            "reaction_time_ms",
            "H",
            "prev_observed_belief_L",
        ),
        context="Step 5 hazard-signature input",
    )
    winners = _validate_winner_parameter_table(winner_parameter_table)
    winner_map = winners.set_index("participant_id")["winner_model_name"].to_dict()

    df = df_all.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["split"] = df["split"].astype(str)
    test_df = df[df["split"] == "TEST"].copy()
    if test_df.empty:
        raise ValueError("Step 5 hazard-signature checks require TEST rows.")

    test_df["winner_model_name"] = test_df["participant_id"].map(winner_map)
    if test_df["winner_model_name"].isna().any():
        missing = sorted(
            test_df.loc[test_df["winner_model_name"].isna(), "participant_id"].unique().tolist()
        )
        raise ValueError(f"Missing winner model assignments for participants: {missing}")

    test_df = test_df.sort_values(["participant_id", "block_id", "trial_index"]).reset_index(drop=True)
    test_df["is_change_point"] = (
        test_df.groupby(["participant_id", "block_id"], sort=True)["correct_side"]
        .transform(lambda x: x.ne(x.shift(1)).fillna(False))
        .astype(int)
    )
    test_df["psi_t"] = [
        float(psi_function(float(prev_l), float(h_value)))
        for prev_l, h_value in zip(
            test_df["prev_observed_belief_L"].to_numpy(dtype=float),
            test_df["H"].to_numpy(dtype=float),
            strict=True,
        )
    ]
    test_df["abs_prev_belief_L"] = test_df["prev_observed_belief_L"].abs()
    test_df["abs_psi_t"] = test_df["psi_t"].abs()
    test_df["psi_shrinkage_abs"] = test_df["abs_prev_belief_L"] - test_df["abs_psi_t"]
    test_df["is_choice_correct"] = (
        pd.to_numeric(test_df["choice"], errors="coerce")
        == pd.to_numeric(test_df["correct_side"], errors="coerce")
    ).astype(int)

    trial_df = test_df[list(_HAZARD_TRIAL_COLUMNS)].copy()

    block_rows: list[dict[str, Any]] = []
    for (participant_id, block_id, winner_model_name), chunk in trial_df.groupby(
        ["participant_id", "block_id", "winner_model_name"],
        sort=True,
    ):
        change_chunk = chunk[chunk["is_change_point"] == 1]
        stable_chunk = chunk[chunk["is_change_point"] == 0]

        block_rows.append(
            {
                "participant_id": str(participant_id),
                "block_id": int(block_id),
                "winner_model_name": str(winner_model_name),
                "n_trials": int(len(chunk)),
                "n_change_points": int(len(change_chunk)),
                "n_stable_trials": int(len(stable_chunk)),
                "change_point_accuracy": (
                    float(change_chunk["is_choice_correct"].mean())
                    if not change_chunk.empty
                    else float(np.nan)
                ),
                "stable_accuracy": (
                    float(stable_chunk["is_choice_correct"].mean())
                    if not stable_chunk.empty
                    else float(np.nan)
                ),
                "change_point_median_rt_ms": (
                    float(change_chunk["reaction_time_ms"].median())
                    if not change_chunk.empty
                    else float(np.nan)
                ),
                "stable_median_rt_ms": (
                    float(stable_chunk["reaction_time_ms"].median())
                    if not stable_chunk.empty
                    else float(np.nan)
                ),
                "h_vs_abs_psi_spearman": _safe_spearman(chunk["H"], chunk["abs_psi_t"]),
                "h_vs_shrinkage_spearman": _safe_spearman(chunk["H"], chunk["psi_shrinkage_abs"]),
                "mean_shrinkage_change": (
                    float(change_chunk["psi_shrinkage_abs"].mean())
                    if not change_chunk.empty
                    else float(np.nan)
                ),
                "mean_shrinkage_stable": (
                    float(stable_chunk["psi_shrinkage_abs"].mean())
                    if not stable_chunk.empty
                    else float(np.nan)
                ),
            }
        )

    block_df = pd.DataFrame(block_rows, columns=_HAZARD_BLOCK_COLUMNS)
    return {
        "hazard_signature_trial": trial_df,
        "hazard_signature_block": block_df,
    }


def _run_winner_model_for_latents(
    participant_df: pd.DataFrame,
    *,
    winner_model_name: str,
    best_model_params: dict[str, Any],
    ddm_n_samples_per_trial: int,
    latent_cont_noise_std: float,
    random_seed: int,
) -> pd.DataFrame:
    if winner_model_name == "cont_threshold":
        kwargs = {
            key: best_model_params[key]
            for key in _CONTINUOUS_PARAM_KEYS
            if key in best_model_params
        }
        kwargs["noise_std"] = float(latent_cont_noise_std)
        kwargs["random_seed"] = int(random_seed)
        return run_model_a_threshold(participant_df, **kwargs)

    if winner_model_name == "cont_asymptote":
        kwargs = {
            key: best_model_params[key]
            for key in _CONTINUOUS_PARAM_KEYS
            if key in best_model_params
        }
        kwargs["noise_std"] = float(latent_cont_noise_std)
        kwargs["random_seed"] = int(random_seed)
        return run_model_b_asymptote(participant_df, **kwargs)

    if winner_model_name == "ddm_dnm":
        kwargs = {
            key: best_model_params[key]
            for key in _DDM_PARAM_KEYS
            if key in best_model_params
        }
        kwargs["n_samples_per_trial"] = int(ddm_n_samples_per_trial)
        kwargs["include_rt_samples"] = False
        kwargs["random_seed"] = int(random_seed)
        return run_model_c_ddm(participant_df, **kwargs)

    raise ValueError(f"Unsupported winner model for Step 5 latent reporting: {winner_model_name}")


def run_latent_reporting(
    df_all: pd.DataFrame,
    winner_parameter_table: pd.DataFrame,
    *,
    ddm_n_samples_per_trial: int,
    latent_cont_noise_std: float,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    """Re-simulate winner-model latents on ALL rows and summarize by block."""
    if int(ddm_n_samples_per_trial) <= 0:
        raise ValueError("ddm_n_samples_per_trial must be > 0.")
    if float(latent_cont_noise_std) < 0.0:
        raise ValueError("latent_cont_noise_std must be >= 0.")

    _validate_required_columns(
        df_all,
        (
            "row_id",
            "participant_id",
            "block_id",
            "trial_index",
            "split",
            "choice",
            "correct_side",
            "reaction_time_ms",
            "belief_L",
            "H",
            "LLR",
        ),
        context="Step 5 latent reporting input",
    )
    winners = _validate_winner_parameter_table(winner_parameter_table)
    df = df_all.copy()
    df["participant_id"] = df["participant_id"].astype(str)

    trial_tables: list[pd.DataFrame] = []
    sorted_winners = winners.sort_values("participant_id").reset_index(drop=True)
    for participant_idx, winner_row in enumerate(sorted_winners.itertuples(index=False)):
        participant_id = str(winner_row.participant_id)
        winner_model_name = str(winner_row.winner_model_name)
        best_model_params = dict(winner_row.best_model_params)

        participant_df = df[df["participant_id"] == participant_id].copy()
        if participant_df.empty:
            continue

        model_seed = int(random_seed) + participant_idx * 1_000_003
        sim_df = _run_winner_model_for_latents(
            participant_df,
            winner_model_name=winner_model_name,
            best_model_params=best_model_params,
            ddm_n_samples_per_trial=int(ddm_n_samples_per_trial),
            latent_cont_noise_std=float(latent_cont_noise_std),
            random_seed=model_seed,
        )

        sim_df["winner_model_name"] = winner_model_name
        enrichment = participant_df[
            ["row_id", "split", "correct_side"]
        ].copy()
        merged = sim_df.merge(enrichment, on="row_id", how="left", validate="one_to_one")
        merged["predicted_choice_binary"] = np.where(
            merged["predicted_decision"] == 1,
            1.0,
            np.where(merged["predicted_decision"] == -1, 0.0, np.nan),
        )
        merged["predicted_timeout"] = (merged["predicted_decision"] == 0).astype(int)
        merged["choice_match_excluding_timeout"] = np.where(
            np.isnan(merged["predicted_choice_binary"]),
            np.nan,
            (pd.to_numeric(merged["choice"], errors="coerce") == merged["predicted_choice_binary"]).astype(float),
        )
        trial_tables.append(merged[list(_LATENT_TRIAL_COLUMNS)].copy())

    trial_df = (
        pd.concat(trial_tables, ignore_index=True)
        if trial_tables
        else pd.DataFrame(columns=_LATENT_TRIAL_COLUMNS)
    )

    block_rows: list[dict[str, Any]] = []
    for (participant_id, block_id, winner_model_name), chunk in trial_df.groupby(
        ["participant_id", "block_id", "winner_model_name"],
        sort=True,
    ):
        block_rows.append(
            {
                "participant_id": str(participant_id),
                "block_id": int(block_id),
                "winner_model_name": str(winner_model_name),
                "n_trials": int(len(chunk)),
                "n_timeout_predictions": int(chunk["predicted_timeout"].sum()),
                "timeout_rate": float(chunk["predicted_timeout"].mean()),
                "observed_choice_rate": float(pd.to_numeric(chunk["choice"], errors="coerce").mean()),
                "predicted_choice_rate": float(chunk["predicted_choice_binary"].mean()),
                "choice_accuracy_excluding_timeout": float(chunk["choice_match_excluding_timeout"].mean()),
                "mean_observed_rt_ms": float(pd.to_numeric(chunk["reaction_time_ms"], errors="coerce").mean()),
                "mean_predicted_rt_ms": float(pd.to_numeric(chunk["predicted_rt_ms"], errors="coerce").mean()),
                "mae_rt_ms": float(
                    (
                        pd.to_numeric(chunk["predicted_rt_ms"], errors="coerce")
                        - pd.to_numeric(chunk["reaction_time_ms"], errors="coerce")
                    )
                    .abs()
                    .mean()
                ),
                "mean_observed_belief_L": float(pd.to_numeric(chunk["belief_L"], errors="coerce").mean()),
                "mean_predicted_belief": float(pd.to_numeric(chunk["predicted_belief"], errors="coerce").mean()),
                "mae_belief": float(
                    (
                        pd.to_numeric(chunk["predicted_belief"], errors="coerce")
                        - pd.to_numeric(chunk["belief_L"], errors="coerce")
                    )
                    .abs()
                    .mean()
                ),
                "mean_H": float(pd.to_numeric(chunk["H"], errors="coerce").mean()),
            }
        )

    block_df = pd.DataFrame(block_rows, columns=_LATENT_BLOCK_COLUMNS)
    return {
        "latent_trajectories_trial": trial_df,
        "latent_quantities_block": block_df,
    }
