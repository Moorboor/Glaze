"""Orchestration for pooled Elias model comparison workflow.

This module provides functions to prepare data, fit models, and score predictions
in a train/test split paradigm. Functions can be called individually or via
run_model_comparison() for end-to-end execution.

Functions:
    prepare_modeling_data: Load, preprocess, infer subjective hazard from TRAIN
        data, and build normative belief columns.
    fit_models_train_split: Fit candidate models on pooled TRAIN rows and rank
        by training objective.
    score_models_test_split: Evaluate fitted models on pooled TEST rows via
        Monte Carlo simulation and rank by held-out likelihood.
    run_model_comparison: Execute full pipeline: prepare → fit → score → rank.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import os
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


def _validate_n_jobs(n_jobs: int, *, context: str) -> int:
    """Validate and normalize process worker count for optional parallel execution."""
    n_jobs_int = int(n_jobs)
    if n_jobs_int <= 0:
        raise ValueError(f"{context}: n_jobs must be > 0, got {n_jobs_int}.")
    return n_jobs_int


def _default_max_workers() -> int:
    """Return a safe worker cap that leaves one CPU core free."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(int(cpu_count) - 1, 1)


def _requested_fit_starts(fit_config: dict[str, object] | None) -> int:
    """Return configured number of fit starts, defaulting to one when unspecified."""
    if fit_config is None:
        return 1
    starts = int(fit_config.get("n_starts", 1))
    return max(starts, 1)


def _single_start_fit_config(fit_config: dict[str, object] | None) -> dict[str, object] | None:
    """Return a deep-copied fit config constrained to a single optimization start."""
    if fit_config is None:
        return None
    cfg = deepcopy(fit_config)
    cfg["n_starts"] = 1
    return cfg


def _split_dataframe_evenly(df: pd.DataFrame, n_chunks: int) -> list[pd.DataFrame]:
    """Split dataframe into near-equal row chunks while preserving row order."""
    if int(n_chunks) <= 1 or len(df) <= 1:
        return [df.copy()]

    rows = int(len(df))
    chunk_count = min(int(n_chunks), rows)
    base_size, remainder = divmod(rows, chunk_count)
    chunks: list[pd.DataFrame] = []
    start = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < remainder else 0)
        stop = start + size
        if stop > start:
            chunks.append(df.iloc[start:stop].copy())
        start = stop
    return chunks


def _fit_model_job(
    *,
    job_index: int,
    train_df: pd.DataFrame,
    model_name: str,
    fit_config: dict[str, object] | None,
    model_seed: int,
) -> dict[str, object]:
    """Worker payload for fitting one model independently."""
    fit_result = fit_model_parameters(
        train_df,
        model_name=model_name,
        fit_config=fit_config,
        random_seed=int(model_seed),
    )
    return {
        "job_index": int(job_index),
        "model_name": str(model_name),
        "model_seed": int(model_seed),
        "fit_result": fit_result,
    }


def _score_model_job(
    *,
    job_index: int,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict[str, object],
    n_sims_per_trial: int,
    rt_bin_width_ms: float,
    rt_max_ms: float,
    eps: float,
    model_seed: int,
) -> dict[str, object]:
    """Worker payload for scoring one fitted model independently."""
    score_output = score_model_simulation_likelihood(
        test_df,
        model_name=str(model_name),
        model_params=dict(model_params),
        n_sims_per_trial=int(n_sims_per_trial),
        rt_bin_width_ms=float(rt_bin_width_ms),
        rt_max_ms=float(rt_max_ms),
        eps=float(eps),
        random_seed=int(model_seed),
    )
    return {
        "job_index": int(job_index),
        "model_name": str(model_name),
        "model_seed": int(model_seed),
        "aggregate_scores": dict(score_output["aggregate_scores"]),
        "trial_scores": score_output["trial_scores"].copy(),
    }


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
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Fit all candidate models on pooled TRAIN rows."""
    _validate_required_columns(df_model, ("split",), context="fit_models_train_split")
    model_names = _normalize_candidate_models(candidate_models)
    n_workers = _validate_n_jobs(n_jobs, context="fit_models_train_split")

    train_df = df_model[df_model["split"].astype(str) == "TRAIN"].copy()
    if train_df.empty:
        raise ValueError("No TRAIN rows available for pooled fitting.")

    fit_results: dict[str, dict[str, object]] = {}
    fit_rows: list[dict[str, object]] = []

    requested_n_starts = _requested_fit_starts(fit_config)
    use_start_parallelism = bool(requested_n_starts > 1 and n_workers > len(model_names))

    fit_jobs: list[dict[str, object]] = []
    job_index = 0
    for model_index, model_name in enumerate(model_names):
        model_seed_base = int(random_seed) + (model_index * 1000)
        if use_start_parallelism:
            single_start_config = _single_start_fit_config(fit_config)
            for start_index in range(requested_n_starts):
                fit_jobs.append(
                    {
                        "job_index": int(job_index),
                        "model_name": str(model_name),
                        "model_seed": int(model_seed_base + (start_index * 100_000)),
                        "model_seed_base": int(model_seed_base),
                        "start_index": int(start_index),
                        "fit_config": single_start_config,
                    }
                )
                job_index += 1
        else:
            fit_jobs.append(
                {
                    "job_index": int(job_index),
                    "model_name": str(model_name),
                    "model_seed": int(model_seed_base),
                    "model_seed_base": int(model_seed_base),
                    "start_index": 0,
                    "fit_config": fit_config,
                }
            )
            job_index += 1

    fit_payloads_by_index: dict[int, dict[str, object]] = {}
    if len(fit_jobs) == 1 or n_workers == 1:
        for fit_job in fit_jobs:
            payload = _fit_model_job(
                job_index=int(fit_job["job_index"]),
                train_df=train_df,
                model_name=str(fit_job["model_name"]),
                fit_config=fit_job.get("fit_config"),
                model_seed=int(fit_job["model_seed"]),
            )
            fit_payloads_by_index[int(payload["job_index"])] = payload
    else:
        max_workers = min(int(n_workers), len(fit_jobs), _default_max_workers())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _fit_model_job,
                    job_index=int(fit_job["job_index"]),
                    train_df=train_df,
                    model_name=str(fit_job["model_name"]),
                    fit_config=fit_job.get("fit_config"),
                    model_seed=int(fit_job["model_seed"]),
                ): int(fit_job["job_index"])
                for fit_job in fit_jobs
            }
            for future in as_completed(future_to_index):
                payload = future.result()
                fit_payloads_by_index[int(payload["job_index"])] = payload

    if use_start_parallelism:
        jobs_by_model: dict[str, list[dict[str, object]]] = {str(model_name): [] for model_name in model_names}
        for fit_job in fit_jobs:
            current_job_index = int(fit_job["job_index"])
            if current_job_index not in fit_payloads_by_index:
                raise RuntimeError(f"Missing fit payload for job index {current_job_index}.")
            payload = fit_payloads_by_index[current_job_index]
            model_name = str(fit_job["model_name"])
            jobs_by_model[model_name].append(
                {
                    "start_index": int(fit_job["start_index"]),
                    "model_seed_base": int(fit_job["model_seed_base"]),
                    "payload": payload,
                }
            )

        for model_name in model_names:
            grouped_jobs = sorted(
                jobs_by_model[str(model_name)],
                key=lambda entry: int(entry["start_index"]),
            )
            if len(grouped_jobs) != requested_n_starts:
                raise RuntimeError(
                    f"Expected {requested_n_starts} start payloads for model '{model_name}', "
                    f"found {len(grouped_jobs)}."
                )

            start_fit_results = [
                dict(entry["payload"]["fit_result"])
                for entry in grouped_jobs
            ]
            best_fit_result = min(
                start_fit_results,
                key=lambda result: float(result["best_fit_objective_score"]),
            )
            merged_fit_result = dict(best_fit_result)
            merged_fit_result["n_starts"] = int(requested_n_starts)
            merged_fit_result["n_evaluations"] = int(
                sum(int(result["n_evaluations"]) for result in start_fit_results)
            )
            merged_fit_result["random_seed"] = int(grouped_jobs[0]["model_seed_base"])

            if isinstance(merged_fit_result.get("fit_config"), dict):
                merged_fit_config = dict(merged_fit_result["fit_config"])
                merged_fit_config["n_starts"] = int(requested_n_starts)
                merged_fit_result["fit_config"] = merged_fit_config

            trace_frames: list[pd.DataFrame] = []
            for entry, start_result in zip(grouped_jobs, start_fit_results, strict=True):
                trace_table = start_result.get("trace_table")
                if isinstance(trace_table, pd.DataFrame):
                    trace_copy = trace_table.copy()
                    trace_copy["start_index"] = int(entry["start_index"])
                    trace_frames.append(trace_copy)
            merged_fit_result["trace_table"] = (
                pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
            )

            fit_results[str(model_name)] = merged_fit_result
            fit_rows.append(
                {
                    "model_name": str(model_name),
                    "fit_objective": str(merged_fit_result["fit_objective"]),
                    "best_fit_objective_score": float(merged_fit_result["best_fit_objective_score"]),
                    "best_joint_score_train": float(merged_fit_result["best_joint_score"]),
                    "best_choice_only_score_train": float(merged_fit_result["best_choice_only_score"]),
                    "best_rt_only_cond_score_train": float(merged_fit_result["best_rt_only_cond_score"]),
                    "n_parameters": int(merged_fit_result["n_parameters"]),
                    "n_evaluations": int(merged_fit_result["n_evaluations"]),
                    "fit_seed": int(grouped_jobs[0]["model_seed_base"]),
                }
            )
    else:
        for fit_job in fit_jobs:
            current_job_index = int(fit_job["job_index"])
            if current_job_index not in fit_payloads_by_index:
                raise RuntimeError(f"Missing fit payload for job index {current_job_index}.")
            payload = fit_payloads_by_index[current_job_index]
            model_name = str(payload["model_name"])
            model_seed = int(payload["model_seed"])
            fit_result = dict(payload["fit_result"])

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
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Score fitted models on pooled TEST rows and rank by held-out joint NLL."""
    _validate_required_columns(df_model, ("split",), context="score_models_test_split")
    n_workers = _validate_n_jobs(n_jobs, context="score_models_test_split")
    test_df = df_model[df_model["split"].astype(str) == "TEST"].copy()
    if test_df.empty:
        raise ValueError("No TEST rows available for pooled held-out scoring.")

    score_rows: list[dict[str, object]] = []
    trial_scores_by_model: dict[str, pd.DataFrame] = {}

    model_names = sorted(fitted_models.keys())
    chunk_jobs_per_model = 1
    if n_workers > len(model_names) and len(test_df) > len(model_names):
        # Increase parallel scoring safely by splitting each model over TEST chunks.
        chunk_jobs_per_model = max(1, min(4, int(n_workers) // len(model_names), len(test_df)))

    score_jobs: list[dict[str, object]] = []
    job_index = 0
    for model_index, model_name in enumerate(model_names):
        model_seed_base = int(random_seed) + (model_index * 1000)
        model_params = dict(fitted_models[model_name].get("best_model_params", {}))
        test_chunks = _split_dataframe_evenly(test_df, n_chunks=chunk_jobs_per_model)
        for chunk_index, test_chunk in enumerate(test_chunks):
            score_jobs.append(
                {
                    "job_index": int(job_index),
                    "model_name": str(model_name),
                    "model_seed": int(model_seed_base + (chunk_index * 100_000)),
                    "model_seed_base": int(model_seed_base),
                    "chunk_index": int(chunk_index),
                    "n_chunks": int(len(test_chunks)),
                    "test_df_chunk": test_chunk,
                    "model_params": model_params,
                }
            )
            job_index += 1

    score_payloads_by_index: dict[int, dict[str, object]] = {}
    if len(score_jobs) == 1 or n_workers == 1:
        for score_job in score_jobs:
            payload = _score_model_job(
                job_index=int(score_job["job_index"]),
                test_df=score_job["test_df_chunk"],
                model_name=str(score_job["model_name"]),
                model_params=dict(score_job["model_params"]),
                n_sims_per_trial=int(n_sims_per_trial),
                rt_bin_width_ms=float(rt_bin_width_ms),
                rt_max_ms=float(rt_max_ms),
                eps=float(eps),
                model_seed=int(score_job["model_seed"]),
            )
            score_payloads_by_index[int(payload["job_index"])] = payload
    else:
        max_workers = min(int(n_workers), len(score_jobs), _default_max_workers())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _score_model_job,
                    job_index=int(score_job["job_index"]),
                    test_df=score_job["test_df_chunk"],
                    model_name=str(score_job["model_name"]),
                    model_params=dict(score_job["model_params"]),
                    n_sims_per_trial=int(n_sims_per_trial),
                    rt_bin_width_ms=float(rt_bin_width_ms),
                    rt_max_ms=float(rt_max_ms),
                    eps=float(eps),
                    model_seed=int(score_job["model_seed"]),
                ): int(score_job["job_index"])
                for score_job in score_jobs
            }
            for future in as_completed(future_to_index):
                payload = future.result()
                score_payloads_by_index[int(payload["job_index"])] = payload

    jobs_by_model: dict[str, list[dict[str, object]]] = {str(model_name): [] for model_name in model_names}
    for score_job in score_jobs:
        current_job_index = int(score_job["job_index"])
        if current_job_index not in score_payloads_by_index:
            raise RuntimeError(f"Missing score payload for job index {current_job_index}.")
        payload = score_payloads_by_index[current_job_index]
        model_name = str(score_job["model_name"])
        jobs_by_model[model_name].append(
            {
                "chunk_index": int(score_job["chunk_index"]),
                "n_chunks": int(score_job["n_chunks"]),
                "model_seed_base": int(score_job["model_seed_base"]),
                "payload": payload,
            }
        )

    for model_name in model_names:
        grouped_jobs = sorted(
            jobs_by_model[str(model_name)],
            key=lambda entry: int(entry["chunk_index"]),
        )
        if not grouped_jobs:
            raise RuntimeError(f"No score payloads found for model '{model_name}'.")

        expected_chunks = int(grouped_jobs[0]["n_chunks"])
        if len(grouped_jobs) != expected_chunks:
            raise RuntimeError(
                f"Expected {expected_chunks} score chunks for model '{model_name}', "
                f"found {len(grouped_jobs)}."
            )

        aggregate_parts = [
            dict(entry["payload"]["aggregate_scores"])
            for entry in grouped_jobs
        ]
        trial_frames = [
            entry["payload"]["trial_scores"].copy()
            for entry in grouped_jobs
        ]
        merged_trial_scores = pd.concat(trial_frames, ignore_index=True)
        merged_aggregate = {
            "joint_score": float(sum(float(part["joint_score"]) for part in aggregate_parts)),
            "choice_only_score": float(sum(float(part["choice_only_score"]) for part in aggregate_parts)),
            "rt_only_cond_score": float(sum(float(part["rt_only_cond_score"]) for part in aggregate_parts)),
            "n_trials": int(sum(int(part["n_trials"]) for part in aggregate_parts)),
        }

        trial_scores_by_model[str(model_name)] = merged_trial_scores
        score_rows.append(
            {
                "model_name": str(model_name),
                "joint_score_test": float(merged_aggregate["joint_score"]),
                "choice_only_score_test": float(merged_aggregate["choice_only_score"]),
                "rt_only_cond_score_test": float(merged_aggregate["rt_only_cond_score"]),
                "n_trials_test": int(merged_aggregate["n_trials"]),
                "score_seed": int(grouped_jobs[0]["model_seed_base"]),
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
    fit_n_jobs: int = 1,
    score_n_jobs: int = 1,
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
        n_jobs=int(fit_n_jobs),
    )
    score_output = score_models_test_split(
        prep_output["df_model"],
        fitted_models=fit_output["fit_results"],
        n_sims_per_trial=int(n_sims_per_trial),
        rt_bin_width_ms=float(rt_bin_width_ms),
        rt_max_ms=float(rt_max_ms),
        eps=float(eps),
        random_seed=int(score_seed),
        n_jobs=int(score_n_jobs),
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
