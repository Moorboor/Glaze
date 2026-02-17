"""Notebook-first orchestration for pooled Elias model comparison.

Function Inventory:
- `prepare_modeling_data`: Load, preprocess, fit subjective hazard, and build normative state; called by `run_model_comparison` and notebook Step 1/2.
- `fit_models_train_split`: Fit all candidate models on pooled TRAIN rows; called by `run_model_comparison` and notebook Step 3.
- `score_models_test_split`: Score fitted models on pooled TEST rows; called by `run_model_comparison` and notebook Step 4.
- `run_model_comparison`: One-call wrapper returning fit/score tables and winner; optional script entrypoint.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_pipeline import (
    attach_subjective_h_from_train,
    build_normative_belief_columns,
    fit_blockwise_subjective_h_choice_only,
    load_participant_data,
    preprocess_loaded_participant_data,
)
from .model_fitting import fit_model_parameters
from .model_scoring import score_model_simulation_likelihood


SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "cont_threshold",
    "cont_asymptote",
    "ddm_dnm",
)


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Validate required columns for workflow entry points."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _normalize_candidate_models(candidate_models: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Validate and normalize candidate model names for workflow calls."""
    if candidate_models is None:
        return tuple(SUPPORTED_MODEL_NAMES)

    normalized = tuple(str(name) for name in candidate_models)
    if len(normalized) == 0:
        raise ValueError("candidate_models must not be empty.")

    invalid = sorted(set(normalized) - set(SUPPORTED_MODEL_NAMES))
    if invalid:
        raise ValueError(
            f"Unsupported candidate_models: {invalid}. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    return normalized


def prepare_modeling_data(
    *,
    csv_path: str = "data/participants.csv",
    participant_ids: list[str] | None = None,
    min_rt_ms: float = 150.0,
    max_rt_ms: float = 5000.0,
    train_trial_max_index: int = 30,
    expected_blocks_per_participant: int = 4,
    nominal_trials_per_block_before: int = 40,
    subjective_h_beta: float = 1.0,
) -> dict[str, Any]:
    """Prepare modeling frame with TRAIN-inferred H and reconstructed normative state."""
    df_loaded = load_participant_data(
        csv_path=csv_path,
        participant_ids=participant_ids,
    )
    preprocessing_output = preprocess_loaded_participant_data(
        df_loaded,
        min_rt_ms=float(min_rt_ms),
        max_rt_ms=float(max_rt_ms),
        train_trial_max_index=int(train_trial_max_index),
        expected_blocks_per_participant=int(expected_blocks_per_participant),
        nominal_trials_per_block_before=int(nominal_trials_per_block_before),
    )
    df_preprocessed = preprocessing_output["df_all"].copy()

    subjective_h_table = fit_blockwise_subjective_h_choice_only(
        df_preprocessed,
        beta=float(subjective_h_beta),
    )
    df_with_h = attach_subjective_h_from_train(df_preprocessed, subjective_h_table)
    df_model = build_normative_belief_columns(df_with_h)

    _validate_required_columns(
        df_model,
        ("split", "H", "prev_normative_belief_L", "normative_belief_L"),
        context="prepare_modeling_data output",
    )

    return {
        "df_loaded": df_loaded,
        "preprocessing": preprocessing_output,
        "subjective_h_table": subjective_h_table,
        "df_model": df_model,
    }


def fit_models_train_split(
    df_model: pd.DataFrame,
    *,
    candidate_models: tuple[str, ...] | list[str] | None = None,
    fit_config: dict[str, object] | None = None,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Fit all candidate models on pooled TRAIN rows."""
    _validate_required_columns(df_model, ("split",), context="fit_models_train_split")
    model_names = _normalize_candidate_models(candidate_models)

    train_df = df_model[df_model["split"].astype(str) == "TRAIN"].copy()
    if train_df.empty:
        raise ValueError("No TRAIN rows available for pooled fitting.")

    fit_results: dict[str, dict[str, object]] = {}
    fit_rows: list[dict[str, object]] = []

    for index, model_name in enumerate(model_names):
        model_seed = int(random_seed) + (index * 1000)
        fit_result = fit_model_parameters(
            train_df,
            model_name=model_name,
            fit_config=fit_config,
            random_seed=model_seed,
        )
        fit_results[model_name] = fit_result
        fit_rows.append(
            {
                "model_name": str(model_name),
                "fit_objective": str(fit_result["fit_objective"]),
                "best_fit_objective_score": float(fit_result["best_fit_objective_score"]),
                "best_joint_score_train": float(fit_result["best_joint_score"]),
                "best_choice_only_score_train": float(fit_result["best_choice_only_score"]),
                "best_rt_only_cond_score_train": float(fit_result["best_rt_only_cond_score"]),
                "n_parameters": int(fit_result["n_parameters"]),
                "n_evaluations": int(fit_result["n_evaluations"]),
                "fit_seed": int(model_seed),
            }
        )

    fit_table = pd.DataFrame(fit_rows).sort_values(
        ["best_fit_objective_score", "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return {
        "train_df": train_df,
        "fit_results": fit_results,
        "fit_table": fit_table,
        "winner_model_name": str(fit_table.loc[0, "model_name"]),
    }


def score_models_test_split(
    df_model: pd.DataFrame,
    *,
    fitted_models: dict[str, dict[str, object]],
    n_sims_per_trial: int = 200,
    rt_bin_width_ms: float = 20.0,
    rt_max_ms: float = 5000.0,
    eps: float = 1e-12,
    random_seed: int = 10_000,
) -> dict[str, Any]:
    """Score fitted models on pooled TEST rows and rank by held-out joint NLL."""
    _validate_required_columns(df_model, ("split",), context="score_models_test_split")
    test_df = df_model[df_model["split"].astype(str) == "TEST"].copy()
    if test_df.empty:
        raise ValueError("No TEST rows available for pooled held-out scoring.")

    score_rows: list[dict[str, object]] = []
    trial_scores_by_model: dict[str, pd.DataFrame] = {}

    for index, model_name in enumerate(sorted(fitted_models.keys())):
        fit_result = fitted_models[model_name]
        model_params = dict(fit_result.get("best_model_params", {}))
        model_seed = int(random_seed) + (index * 1000)

        score_output = score_model_simulation_likelihood(
            test_df,
            model_name=str(model_name),
            model_params=model_params,
            n_sims_per_trial=int(n_sims_per_trial),
            rt_bin_width_ms=float(rt_bin_width_ms),
            rt_max_ms=float(rt_max_ms),
            eps=float(eps),
            random_seed=model_seed,
        )
        aggregate = dict(score_output["aggregate_scores"])
        trial_scores_by_model[str(model_name)] = score_output["trial_scores"].copy()

        score_rows.append(
            {
                "model_name": str(model_name),
                "joint_score_test": float(aggregate["joint_score"]),
                "choice_only_score_test": float(aggregate["choice_only_score"]),
                "rt_only_cond_score_test": float(aggregate["rt_only_cond_score"]),
                "n_trials_test": int(aggregate["n_trials"]),
                "score_seed": int(model_seed),
            }
        )

    score_table = pd.DataFrame(score_rows).sort_values(
        ["joint_score_test", "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return {
        "test_df": test_df,
        "score_table": score_table,
        "trial_scores_by_model": trial_scores_by_model,
        "winner_model_name": str(score_table.loc[0, "model_name"]),
    }


def run_model_comparison(
    *,
    csv_path: str = "data/participants.csv",
    participant_ids: list[str] | None = None,
    candidate_models: tuple[str, ...] | list[str] | None = None,
    fit_config: dict[str, object] | None = None,
    n_sims_per_trial: int = 200,
    rt_bin_width_ms: float = 20.0,
    rt_max_ms: float = 5000.0,
    eps: float = 1e-12,
    fit_seed: int = 0,
    score_seed: int = 10_000,
) -> dict[str, Any]:
    """Run full pooled workflow: prepare data, fit TRAIN, score TEST, rank models."""
    model_names = _normalize_candidate_models(candidate_models)

    prep_output = prepare_modeling_data(
        csv_path=csv_path,
        participant_ids=participant_ids,
    )
    fit_output = fit_models_train_split(
        prep_output["df_model"],
        candidate_models=model_names,
        fit_config=fit_config,
        random_seed=int(fit_seed),
    )
    score_output = score_models_test_split(
        prep_output["df_model"],
        fitted_models=fit_output["fit_results"],
        n_sims_per_trial=int(n_sims_per_trial),
        rt_bin_width_ms=float(rt_bin_width_ms),
        rt_max_ms=float(rt_max_ms),
        eps=float(eps),
        random_seed=int(score_seed),
    )

    overview_table = fit_output["fit_table"].merge(
        score_output["score_table"],
        on="model_name",
        how="left",
    ).sort_values(["joint_score_test", "model_name"], ascending=[True, True]).reset_index(drop=True)

    return {
        "data": prep_output,
        "fit": fit_output,
        "score": score_output,
        "overview_table": overview_table,
        "winner_model_name": str(score_output["winner_model_name"]),
    }
