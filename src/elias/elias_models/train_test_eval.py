"""Step 4 participant train/test fitting pipeline with persisted outputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import SUPPORTED_MODEL_NAMES
from .continuous_models import _prepare_model_input
from .data_validation import _validate_required_columns
from .likelihood_scoring import score_model_simulation_likelihood
from .optimizer_runner import fit_model_parameters
from .results_store import (
    build_basic_manifest,
    load_json,
    load_table_csv,
    prepare_run_dir,
    resolve_elias_data_root,
    save_json,
    save_table_csv,
)
from .winner_rules import apply_step4_winner_rules


STEP4_PIPELINE_NAME = "participant_fit"
_STEP4_TABLE_NAMES: tuple[str, ...] = (
    "participant_model_fits_train",
    "participant_model_scores_test",
    "participant_model_scores_test_blockwise",
    "participant_winner_table",
    "group_winner_counts",
    "group_winner_summary",
)

_DEFAULT_STEP4_PIPELINE_CONFIG: dict[str, object] = {
    "candidate_models": SUPPORTED_MODEL_NAMES,
    "fit_n_starts": 4,
    "fit_n_iterations": 8,
    "fit_n_sims_per_trial": 150,
    "eval_n_sims_per_trial": 150,
    "rt_bin_width_ms": 20.0,
    "rt_max_ms": 5000.0,
    "eps": 1e-12,
    "dt_ms": 1.0,
    "max_duration_ms": 5000.0,
    "random_seed": 0,
    "winner_primary_score_column": "joint_score",
    "winner_tie_tolerance": 1e-9,
}


def _validate_candidate_models(candidate_models: tuple[str, ...]) -> tuple[str, ...]:
    """Validate candidate model names against supported options."""
    normalized = tuple(str(model_name) for model_name in candidate_models)
    if len(normalized) == 0:
        raise ValueError("candidate_models must not be empty.")
    invalid = [model_name for model_name in normalized if model_name not in SUPPORTED_MODEL_NAMES]
    if invalid:
        raise ValueError(
            f"Unsupported candidate models: {invalid}. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    return normalized


def build_step4_pipeline_config(
    *,
    candidate_models: tuple[str, ...] = SUPPORTED_MODEL_NAMES,
    fit_n_starts: int = 4,
    fit_n_iterations: int = 8,
    fit_n_sims_per_trial: int = 150,
    eval_n_sims_per_trial: int = 150,
    rt_bin_width_ms: float = 20.0,
    rt_max_ms: float = 5000.0,
    eps: float = 1e-12,
    dt_ms: float = 1.0,
    max_duration_ms: float = 5000.0,
    random_seed: int = 0,
    winner_primary_score_column: str = "joint_score",
    winner_tie_tolerance: float = 1e-9,
) -> dict[str, object]:
    """Build canonical Step 4 pipeline configuration.

    Args:
        candidate_models: Candidate model names to fit and evaluate.
        fit_n_starts: Multi-start count per participant-model fit.
        fit_n_iterations: Local iterations per start.
        fit_n_sims_per_trial: Simulations per trial for fit objective calls.
        eval_n_sims_per_trial: Simulations per trial for TEST evaluation calls.
        rt_bin_width_ms: RT histogram bin width used in scoring.
        rt_max_ms: Maximum RT edge used in scoring histograms.
        eps: Smoothing floor for probability terms.
        dt_ms: Simulation integration step in milliseconds.
        max_duration_ms: Maximum simulation duration in milliseconds.
        random_seed: Base deterministic seed.
        winner_primary_score_column: Score column for winner selection.
        winner_tie_tolerance: Tie tolerance for winner selection.

    Returns:
        Normalized Step 4 configuration dictionary.
    """
    cfg = {
        "candidate_models": _validate_candidate_models(tuple(candidate_models)),
        "fit_n_starts": int(fit_n_starts),
        "fit_n_iterations": int(fit_n_iterations),
        "fit_n_sims_per_trial": int(fit_n_sims_per_trial),
        "eval_n_sims_per_trial": int(eval_n_sims_per_trial),
        "rt_bin_width_ms": float(rt_bin_width_ms),
        "rt_max_ms": float(rt_max_ms),
        "eps": float(eps),
        "dt_ms": float(dt_ms),
        "max_duration_ms": float(max_duration_ms),
        "random_seed": int(random_seed),
        "winner_primary_score_column": str(winner_primary_score_column),
        "winner_tie_tolerance": float(winner_tie_tolerance),
    }

    if cfg["fit_n_starts"] <= 0:
        raise ValueError("fit_n_starts must be > 0.")
    if cfg["fit_n_iterations"] < 0:
        raise ValueError("fit_n_iterations must be >= 0.")
    if cfg["fit_n_sims_per_trial"] <= 0:
        raise ValueError("fit_n_sims_per_trial must be > 0.")
    if cfg["eval_n_sims_per_trial"] <= 0:
        raise ValueError("eval_n_sims_per_trial must be > 0.")
    if cfg["rt_bin_width_ms"] <= 0.0:
        raise ValueError("rt_bin_width_ms must be > 0.")
    if cfg["rt_max_ms"] <= 0.0:
        raise ValueError("rt_max_ms must be > 0.")
    if cfg["eps"] <= 0.0:
        raise ValueError("eps must be > 0.")
    if cfg["dt_ms"] <= 0.0:
        raise ValueError("dt_ms must be > 0.")
    if cfg["max_duration_ms"] <= 0.0:
        raise ValueError("max_duration_ms must be > 0.")
    if cfg["winner_primary_score_column"] not in {"joint_score", "choice_only_score", "rt_only_cond_score", "bic_score"}:
        raise ValueError(
            "winner_primary_score_column must be one of "
            "{'joint_score', 'choice_only_score', 'rt_only_cond_score', 'bic_score'}."
        )
    if cfg["winner_tie_tolerance"] < 0.0:
        raise ValueError("winner_tie_tolerance must be >= 0.")
    return cfg


def _normalize_step4_pipeline_config(config: dict[str, object]) -> dict[str, object]:
    """Merge user config with defaults and validate resulting settings."""
    merged = deepcopy(_DEFAULT_STEP4_PIPELINE_CONFIG)
    merged.update(dict(config))
    return build_step4_pipeline_config(
        candidate_models=tuple(merged["candidate_models"]),
        fit_n_starts=int(merged["fit_n_starts"]),
        fit_n_iterations=int(merged["fit_n_iterations"]),
        fit_n_sims_per_trial=int(merged["fit_n_sims_per_trial"]),
        eval_n_sims_per_trial=int(merged["eval_n_sims_per_trial"]),
        rt_bin_width_ms=float(merged["rt_bin_width_ms"]),
        rt_max_ms=float(merged["rt_max_ms"]),
        eps=float(merged["eps"]),
        dt_ms=float(merged["dt_ms"]),
        max_duration_ms=float(merged["max_duration_ms"]),
        random_seed=int(merged["random_seed"]),
        winner_primary_score_column=str(merged["winner_primary_score_column"]),
        winner_tie_tolerance=float(merged["winner_tie_tolerance"]),
    )


def _fit_config_from_step4_config(config: dict[str, object]) -> dict[str, object]:
    """Build optimizer config from Step 4 configuration."""
    return {
        "n_starts": int(config["fit_n_starts"]),
        "n_iterations": int(config["fit_n_iterations"]),
        "n_sims_per_trial": int(config["fit_n_sims_per_trial"]),
        "rt_bin_width_ms": float(config["rt_bin_width_ms"]),
        "rt_max_ms": float(config["rt_max_ms"]),
        "eps": float(config["eps"]),
        "fixed_model_params": {
            "dt_ms": float(config["dt_ms"]),
            "max_duration_ms": float(config["max_duration_ms"]),
        },
    }


def _score_dataset_for_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    model_params: dict[str, object],
    n_sims_per_trial: int,
    rt_bin_width_ms: float,
    rt_max_ms: float,
    eps: float,
    random_seed: int,
) -> dict[str, object]:
    """Run simulation-based likelihood scoring for one dataset/model pair."""
    return score_model_simulation_likelihood(
        df,
        model_name=model_name,
        model_params=model_params,
        n_sims_per_trial=int(n_sims_per_trial),
        rt_bin_width_ms=float(rt_bin_width_ms),
        rt_max_ms=float(rt_max_ms),
        eps=float(eps),
        random_seed=int(random_seed),
    )


def run_step4_pipeline(
    df_all: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = "data/elias",
    config: dict[str, object],
    overwrite: bool = False,
) -> dict[str, object]:
    """Run Step 4 participant train/test pipeline and persist all artifacts.

    Args:
        df_all: Preprocessed participant DataFrame with `split` column.
        run_id: Stable run identifier.
        output_root: Root output folder, defaulting to `data/elias`.
        config: Step 4 pipeline configuration.
        overwrite: Whether to overwrite an existing run directory.

    Returns:
        Dictionary containing manifest/config metadata and in-memory tables.
    """
    _validate_required_columns(
        df_all,
        ("participant_id", "block_id", "trial_index", "split", "choice", "reaction_time_ms"),
        context="Step 4 pipeline input",
    )
    normalized_config = _normalize_step4_pipeline_config(config)

    paths = prepare_run_dir(
        output_root,
        pipeline_name=STEP4_PIPELINE_NAME,
        run_id=run_id,
        overwrite=overwrite,
    )
    config_path = save_json(normalized_config, paths["run_dir"] / "config.json")

    model_df = _prepare_model_input(df_all)
    model_df["participant_id"] = model_df["participant_id"].astype(str)
    model_df["split"] = model_df["split"].astype(str)

    valid_splits = {"TRAIN", "TEST"}
    present_splits = set(model_df["split"].unique().tolist())
    missing_splits = sorted(valid_splits - present_splits)
    if missing_splits:
        raise ValueError(f"Step 4 requires split labels TRAIN and TEST; missing: {missing_splits}")

    participant_ids = sorted(model_df["participant_id"].unique().tolist())
    if len(participant_ids) == 0:
        raise ValueError("No participants available for Step 4 pipeline.")
    candidate_models = tuple(normalized_config["candidate_models"])
    fit_config = _fit_config_from_step4_config(normalized_config)

    fit_rows: list[dict[str, Any]] = []
    test_score_rows: list[dict[str, Any]] = []
    test_block_score_rows: list[dict[str, Any]] = []

    base_seed = int(normalized_config["random_seed"])
    for participant_idx, participant_id in enumerate(participant_ids):
        participant_df = model_df[model_df["participant_id"] == str(participant_id)].copy()
        train_df = participant_df[participant_df["split"] == "TRAIN"].copy()
        test_df = participant_df[participant_df["split"] == "TEST"].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Participant '{participant_id}' requires both TRAIN and TEST rows. "
                f"Found TRAIN={len(train_df)}, TEST={len(test_df)}."
            )

        for model_idx, model_name in enumerate(candidate_models):
            # Seed schedule is deterministic across participants/models for reproducibility.
            fit_seed = base_seed + participant_idx * 1_000_003 + model_idx * 10_007
            test_seed = fit_seed + 1

            fit_output = fit_model_parameters(
                train_df,
                model_name=model_name,
                fit_config=fit_config,
                random_seed=int(fit_seed),
            )
            best_model_params = dict(fit_output["best_model_params"])
            best_model_params["dt_ms"] = float(normalized_config["dt_ms"])
            best_model_params["max_duration_ms"] = float(normalized_config["max_duration_ms"])

            fit_rows.append(
                {
                    "participant_id": str(participant_id),
                    "candidate_model_name": str(model_name),
                    "fit_seed": int(fit_seed),
                    "n_train_trials": int(len(train_df)),
                    "n_test_trials": int(len(test_df)),
                    "n_parameters": int(fit_output["n_parameters"]),
                    "n_evaluations": int(fit_output["n_evaluations"]),
                    "fit_n_starts": int(fit_output["n_starts"]),
                    "fit_n_iterations": int(fit_output["n_iterations"]),
                    "best_joint_score_train": float(fit_output["best_joint_score"]),
                    "best_choice_only_score_train": float(fit_output["best_choice_only_score"]),
                    "best_rt_only_cond_score_train": float(fit_output["best_rt_only_cond_score"]),
                    "best_theta": np.asarray(fit_output["best_theta"], dtype=float),
                    "best_eta": np.asarray(fit_output["best_eta"], dtype=float),
                    "best_named_params": dict(fit_output["best_named_params"]),
                    "best_model_params": best_model_params,
                }
            )

            test_score_output = _score_dataset_for_model(
                test_df,
                model_name=str(model_name),
                model_params=best_model_params,
                n_sims_per_trial=int(normalized_config["eval_n_sims_per_trial"]),
                rt_bin_width_ms=float(normalized_config["rt_bin_width_ms"]),
                rt_max_ms=float(normalized_config["rt_max_ms"]),
                eps=float(normalized_config["eps"]),
                random_seed=int(test_seed),
            )
            test_aggregate = dict(test_score_output["aggregate_scores"])
            n_params = int(fit_output["n_parameters"])
            n_trials = int(test_aggregate["n_trials"])
            bic_score = float(
                2.0 * float(test_aggregate["joint_score"])
                + float(n_params) * np.log(max(n_trials, 1))
            )

            test_score_rows.append(
                {
                    "participant_id": str(participant_id),
                    "candidate_model_name": str(model_name),
                    "fit_seed": int(fit_seed),
                    "test_seed": int(test_seed),
                    "n_train_trials": int(len(train_df)),
                    "n_test_trials": n_trials,
                    "n_parameters": n_params,
                    "joint_score": float(test_aggregate["joint_score"]),
                    "choice_only_score": float(test_aggregate["choice_only_score"]),
                    "rt_only_cond_score": float(test_aggregate["rt_only_cond_score"]),
                    "bic_score": bic_score,
                    "best_theta": np.asarray(fit_output["best_theta"], dtype=float),
                    "best_eta": np.asarray(fit_output["best_eta"], dtype=float),
                    "best_named_params": dict(fit_output["best_named_params"]),
                    "best_model_params": best_model_params,
                }
            )

            for block_idx, (block_id, block_test_df) in enumerate(
                test_df.groupby("block_id", sort=True)
            ):
                block_seed = test_seed + 1_000 + block_idx
                block_score_output = _score_dataset_for_model(
                    block_test_df,
                    model_name=str(model_name),
                    model_params=best_model_params,
                    n_sims_per_trial=int(normalized_config["eval_n_sims_per_trial"]),
                    rt_bin_width_ms=float(normalized_config["rt_bin_width_ms"]),
                    rt_max_ms=float(normalized_config["rt_max_ms"]),
                    eps=float(normalized_config["eps"]),
                    random_seed=int(block_seed),
                )
                block_aggregate = dict(block_score_output["aggregate_scores"])
                block_n_trials = int(block_aggregate["n_trials"])
                block_bic_score = float(
                    2.0 * float(block_aggregate["joint_score"])
                    + float(n_params) * np.log(max(block_n_trials, 1))
                )

                test_block_score_rows.append(
                    {
                        "participant_id": str(participant_id),
                        "block_id": int(block_id),
                        "candidate_model_name": str(model_name),
                        "fit_seed": int(fit_seed),
                        "test_seed": int(test_seed),
                        "block_test_seed": int(block_seed),
                        "n_block_test_trials": int(block_n_trials),
                        "n_parameters": n_params,
                        "joint_score": float(block_aggregate["joint_score"]),
                        "choice_only_score": float(block_aggregate["choice_only_score"]),
                        "rt_only_cond_score": float(block_aggregate["rt_only_cond_score"]),
                        "bic_score": block_bic_score,
                    }
                )

    participant_model_fits_train = pd.DataFrame(fit_rows)
    participant_model_scores_test = pd.DataFrame(test_score_rows)
    participant_model_scores_test_blockwise = pd.DataFrame(test_block_score_rows)

    winner_outputs = apply_step4_winner_rules(
        participant_model_scores_test,
        participant_model_scores_test_blockwise,
        primary_score_column=str(normalized_config["winner_primary_score_column"]),
        tie_tolerance=float(normalized_config["winner_tie_tolerance"]),
    )

    tables: dict[str, pd.DataFrame] = {
        "participant_model_fits_train": participant_model_fits_train,
        "participant_model_scores_test": participant_model_scores_test,
        "participant_model_scores_test_blockwise": participant_model_scores_test_blockwise,
        "participant_winner_table": winner_outputs["participant_winner_table"],
        "group_winner_counts": winner_outputs["group_winner_counts"],
        "group_winner_summary": winner_outputs["group_winner_summary"],
    }
    table_paths: dict[str, str] = {}
    for table_name in _STEP4_TABLE_NAMES:
        table_path = save_table_csv(tables[table_name], paths["tables_dir"] / f"{table_name}.csv")
        table_paths[table_name] = str(table_path)

    group_winner_model = (
        str(tables["group_winner_summary"]["group_winner_model_name"].iloc[0])
        if not tables["group_winner_summary"].empty
        else "unknown"
    )
    manifest_extra = {
        "n_participants": int(len(participant_ids)),
        "n_candidate_models": int(len(candidate_models)),
        "n_fit_rows": int(len(participant_model_fits_train)),
        "n_test_score_rows": int(len(participant_model_scores_test)),
        "n_block_score_rows": int(len(participant_model_scores_test_blockwise)),
        "group_winner_model_name": group_winner_model,
        "table_paths": table_paths,
    }
    manifest = build_basic_manifest(
        run_id=run_id,
        pipeline_name=STEP4_PIPELINE_NAME,
        output_root=output_root,
        run_dir=paths["run_dir"],
        config_path=config_path,
        status="completed",
        extra=manifest_extra,
    )
    manifest_path = save_json(manifest, paths["run_dir"] / "manifest.json")

    return {
        "run_id": str(run_id),
        "run_dir": paths["run_dir"],
        "tables_dir": paths["tables_dir"],
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "config": normalized_config,
        "tables": tables,
        "table_paths": table_paths,
    }


def load_step4_run(
    run_id: str,
    *,
    output_root: str | Path = "data/elias",
) -> dict[str, object]:
    """Load a persisted Step 4 run and decode all stored tables.

    Args:
        run_id: Step 4 run identifier.
        output_root: Root output folder, defaulting to `data/elias`.

    Returns:
        Dictionary with loaded manifest, config, and tables.
    """
    data_root = resolve_elias_data_root(output_root)
    run_dir = data_root / STEP4_PIPELINE_NAME / "runs" / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Step 4 run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    manifest = load_json(manifest_path)
    config = load_json(config_path)

    tables: dict[str, pd.DataFrame] = {}
    tables_dir = run_dir / "tables"
    for table_name in _STEP4_TABLE_NAMES:
        table_path = tables_dir / f"{table_name}.csv"
        if table_path.exists():
            tables[table_name] = load_table_csv(table_path)

    return {
        "run_id": str(run_id),
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "config": config,
        "tables": tables,
        "table_paths": {
            name: str((tables_dir / f"{name}.csv")) for name in tables.keys()
        },
    }


def list_step4_runs(
    *,
    output_root: str | Path = "data/elias",
) -> pd.DataFrame:
    """List persisted Step 4 runs with manifest metadata.

    Args:
        output_root: Root output folder, defaulting to `data/elias`.

    Returns:
        Run metadata table sorted by creation time descending.
    """
    data_root = resolve_elias_data_root(output_root)
    runs_root = data_root / STEP4_PIPELINE_NAME / "runs"
    if not runs_root.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "created_at_utc",
                "status",
                "n_participants",
                "n_fit_rows",
                "group_winner_model_name",
                "run_dir",
            ]
        )

    rows: list[dict[str, object]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        rows.append(
            {
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "created_at_utc": manifest.get("created_at_utc", ""),
                "status": manifest.get("status", "unknown"),
                "n_participants": manifest.get("n_participants", np.nan),
                "n_fit_rows": manifest.get("n_fit_rows", np.nan),
                "group_winner_model_name": manifest.get("group_winner_model_name", ""),
                "run_dir": str(run_dir),
            }
        )

    runs_df = pd.DataFrame(rows)
    if runs_df.empty:
        return runs_df
    return runs_df.sort_values("created_at_utc", ascending=False).reset_index(drop=True)
