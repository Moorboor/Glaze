"""Posterior predictive checks for Step 5 reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_validation import _validate_required_columns
from .likelihood_scoring import score_model_simulation_likelihood


_PPC_TRIAL_COLUMNS: tuple[str, ...] = (
    "row_id",
    "participant_id",
    "block_id",
    "trial_index",
    "winner_model_name",
    "observed_choice",
    "observed_rt_ms",
    "p_choice",
    "p_rt_given_choice",
    "nll_choice",
    "nll_rt_cond",
    "nll_joint",
    "n_sims_per_trial",
    "seed_used",
)

_PPC_BLOCK_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "block_id",
    "winner_model_name",
    "n_trials",
    "joint_score",
    "choice_only_score",
    "rt_only_cond_score",
    "joint_nll_per_trial",
    "choice_nll_per_trial",
    "rt_nll_per_trial",
    "n_sims_per_trial",
    "seed_used",
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


def run_posterior_predictive_checks(
    df_all: pd.DataFrame,
    winner_parameter_table: pd.DataFrame,
    *,
    n_sims_per_trial: int,
    rt_bin_width_ms: float,
    rt_max_ms: float,
    eps: float,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    """Run held-out posterior predictive checks for winner models.

    Step-5 policy:
    - Use TEST rows only.
    - Score each participant-block under that participant's Step-4 winner model.
    """
    if int(n_sims_per_trial) <= 0:
        raise ValueError("n_sims_per_trial must be > 0.")
    if float(rt_bin_width_ms) <= 0.0:
        raise ValueError("rt_bin_width_ms must be > 0.")
    if float(rt_max_ms) <= 0.0:
        raise ValueError("rt_max_ms must be > 0.")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0.")

    _validate_required_columns(
        df_all,
        ("participant_id", "block_id", "trial_index", "split"),
        context="Step 5 posterior predictive input",
    )

    winners = _validate_winner_parameter_table(winner_parameter_table)
    df = df_all.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["split"] = df["split"].astype(str)
    test_df = df[df["split"] == "TEST"].copy()
    if test_df.empty:
        raise ValueError("Step 5 posterior predictive checks require TEST rows.")

    trial_tables: list[pd.DataFrame] = []
    block_rows: list[dict[str, Any]] = []

    sorted_winners = winners.sort_values("participant_id").reset_index(drop=True)
    for participant_idx, winner_row in enumerate(sorted_winners.itertuples(index=False)):
        participant_id = str(winner_row.participant_id)
        winner_model_name = str(winner_row.winner_model_name)
        best_model_params = dict(winner_row.best_model_params)

        participant_test = test_df[test_df["participant_id"] == participant_id].copy()
        if participant_test.empty:
            continue

        for block_idx, (block_id, block_df) in enumerate(
            participant_test.groupby("block_id", sort=True)
        ):
            block_seed = int(random_seed) + participant_idx * 1_000_003 + block_idx * 1_009
            score_output = score_model_simulation_likelihood(
                block_df,
                model_name=winner_model_name,
                model_params=best_model_params,
                n_sims_per_trial=int(n_sims_per_trial),
                rt_bin_width_ms=float(rt_bin_width_ms),
                rt_max_ms=float(rt_max_ms),
                eps=float(eps),
                random_seed=block_seed,
            )

            trial_scores = score_output["trial_scores"].copy()
            trial_scores = trial_scores.rename(columns={"model_name": "winner_model_name"})
            trial_scores["winner_model_name"] = winner_model_name
            trial_scores = trial_scores[
                [
                    "row_id",
                    "participant_id",
                    "block_id",
                    "trial_index",
                    "winner_model_name",
                    "observed_choice",
                    "observed_rt_ms",
                    "p_choice",
                    "p_rt_given_choice",
                    "nll_choice",
                    "nll_rt_cond",
                    "nll_joint",
                    "n_sims_per_trial",
                    "seed_used",
                ]
            ]
            trial_tables.append(trial_scores)

            aggregate = dict(score_output["aggregate_scores"])
            n_trials = int(max(aggregate["n_trials"], 1))
            block_rows.append(
                {
                    "participant_id": participant_id,
                    "block_id": int(block_id),
                    "winner_model_name": winner_model_name,
                    "n_trials": int(aggregate["n_trials"]),
                    "joint_score": float(aggregate["joint_score"]),
                    "choice_only_score": float(aggregate["choice_only_score"]),
                    "rt_only_cond_score": float(aggregate["rt_only_cond_score"]),
                    "joint_nll_per_trial": float(aggregate["joint_score"]) / float(n_trials),
                    "choice_nll_per_trial": float(aggregate["choice_only_score"]) / float(n_trials),
                    "rt_nll_per_trial": float(aggregate["rt_only_cond_score"]) / float(n_trials),
                    "n_sims_per_trial": int(aggregate["n_sims_per_trial"]),
                    "seed_used": int(aggregate["random_seed"]),
                }
            )

    trial_df = (
        pd.concat(trial_tables, ignore_index=True)
        if trial_tables
        else pd.DataFrame(columns=_PPC_TRIAL_COLUMNS)
    )
    block_df = pd.DataFrame(block_rows, columns=_PPC_BLOCK_COLUMNS)

    return {
        "posterior_predictive_trial": trial_df,
        "posterior_predictive_block": block_df,
    }
