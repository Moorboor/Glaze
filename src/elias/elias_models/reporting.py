#
# Cross-step Step 3/4/5 orchestration and Step 5 persistence helpers.
# Main functions: build_step5_pipeline_config, run_step345_pipeline,
# run_step34_pipeline, and Step5PipelineError.

"""Cross-step reporting orchestration helpers."""

from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .export_results import build_recovery_aware_conclusion, write_step5_report_markdown
from .latent_checks import run_change_hazard_checks, run_latent_reporting
from .posterior_predictive import run_posterior_predictive_checks
from .results_store import (
    build_basic_manifest,
    prepare_pipeline_run_root,
    prepare_run_dir,
    resolve_unified_run_root,
    save_json,
    save_table_csv,
)
from .surrogate_recovery import load_step3_run, run_step3_pipeline
from .subjective_h import attach_subjective_h_from_train, build_normative_belief_columns
from .train_test_eval import run_step4_pipeline


class Step5PipelineError(RuntimeError):
    """Raised when Step 5 fails after Step 3/4 results have been persisted."""

    def __init__(
        self,
        message: str,
        *,
        manifest_path: Path | None = None,
        error_log_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path
        self.error_log_path = error_log_path


_DEFAULT_STEP5_PIPELINE_CONFIG: dict[str, object] = {
    "ppc_n_sims_per_trial": 200,
    "ddm_n_samples_per_trial": 200,
    "rt_bin_width_ms": 20.0,
    "rt_max_ms": 5000.0,
    "eps": 1e-12,
    "random_seed": 0,
    "latent_cont_noise_std": 0.0,
    "workers": 1,
}

_STEP5_TABLE_NAMES: tuple[str, ...] = (
    "step5_posterior_predictive_trial",
    "step5_posterior_predictive_block",
    "step5_hazard_signature_trial",
    "step5_hazard_signature_block",
    "step5_latent_trajectories_trial",
    "step5_latent_quantities_block",
    "step5_final_conclusion",
)


def build_step5_pipeline_config(
    *,
    ppc_n_sims_per_trial: int = 200,
    ddm_n_samples_per_trial: int = 200,
    rt_bin_width_ms: float = 20.0,
    rt_max_ms: float = 5000.0,
    eps: float = 1e-12,
    random_seed: int = 0,
    latent_cont_noise_std: float = 0.0,
    workers: int = 1,
) -> dict[str, object]:
    """Build canonical Step 5 pipeline configuration.

    Args:
        ppc_n_sims_per_trial: Simulation count per trial for posterior predictive checks.
        ddm_n_samples_per_trial: DDM sample count per trial for latent reporting.
        rt_bin_width_ms: Reaction-time histogram bin width.
        rt_max_ms: Maximum reaction-time edge for scoring histograms.
        eps: Numerical floor for probability terms.
        random_seed: Base deterministic seed for Step 5 analyses.
        latent_cont_noise_std: Optional continuous-model noise for latent re-simulation.
        workers: Number of worker processes for parallel participant tasks.

    Returns:
        Normalized Step 5 configuration payload.
    """
    if int(ppc_n_sims_per_trial) <= 0:
        raise ValueError("ppc_n_sims_per_trial must be > 0.")
    if int(ddm_n_samples_per_trial) <= 0:
        raise ValueError("ddm_n_samples_per_trial must be > 0.")
    if float(rt_bin_width_ms) <= 0.0:
        raise ValueError("rt_bin_width_ms must be > 0.")
    if float(rt_max_ms) <= 0.0:
        raise ValueError("rt_max_ms must be > 0.")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0.")
    if float(latent_cont_noise_std) < 0.0:
        raise ValueError("latent_cont_noise_std must be >= 0.")
    if int(workers) < 1:
        raise ValueError("workers must be >= 1.")

    return {
        "ppc_n_sims_per_trial": int(ppc_n_sims_per_trial),
        "ddm_n_samples_per_trial": int(ddm_n_samples_per_trial),
        "rt_bin_width_ms": float(rt_bin_width_ms),
        "rt_max_ms": float(rt_max_ms),
        "eps": float(eps),
        "random_seed": int(random_seed),
        "latent_cont_noise_std": float(latent_cont_noise_std),
        "workers": int(workers),
    }


def _normalize_step5_pipeline_config(config: dict[str, object] | None) -> dict[str, object]:
    merged = deepcopy(_DEFAULT_STEP5_PIPELINE_CONFIG)
    if config is not None:
        merged.update(dict(config))
    return build_step5_pipeline_config(
        ppc_n_sims_per_trial=int(merged["ppc_n_sims_per_trial"]),
        ddm_n_samples_per_trial=int(merged["ddm_n_samples_per_trial"]),
        rt_bin_width_ms=float(merged["rt_bin_width_ms"]),
        rt_max_ms=float(merged["rt_max_ms"]),
        eps=float(merged["eps"]),
        random_seed=int(merged["random_seed"]),
        latent_cont_noise_std=float(merged["latent_cont_noise_std"]),
        workers=int(merged["workers"]),
    )


def _build_winner_parameter_table(step4_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    participant_winner_table = step4_tables.get("participant_winner_table", pd.DataFrame())
    participant_scores_test = step4_tables.get("participant_model_scores_test", pd.DataFrame())

    required_winner_columns = {"participant_id", "winner_model_name"}
    required_scores_columns = {"participant_id", "candidate_model_name", "best_model_params"}

    missing_winner = sorted(required_winner_columns - set(participant_winner_table.columns))
    if missing_winner:
        raise ValueError(
            f"Missing required Step 4 winner columns for Step 5 linkage: {missing_winner}"
        )
    missing_scores = sorted(required_scores_columns - set(participant_scores_test.columns))
    if missing_scores:
        raise ValueError(
            f"Missing required Step 4 score columns for Step 5 linkage: {missing_scores}"
        )

    winners = participant_winner_table[["participant_id", "winner_model_name"]].copy()
    winners["participant_id"] = winners["participant_id"].astype(str)
    winners["winner_model_name"] = winners["winner_model_name"].astype(str)

    scores = participant_scores_test[
        ["participant_id", "candidate_model_name", "best_model_params"]
    ].copy()
    scores["participant_id"] = scores["participant_id"].astype(str)
    scores["candidate_model_name"] = scores["candidate_model_name"].astype(str)

    merged = winners.merge(
        scores,
        how="left",
        left_on=["participant_id", "winner_model_name"],
        right_on=["participant_id", "candidate_model_name"],
        validate="one_to_one",
    )

    if merged["best_model_params"].isna().any():
        missing_participants = sorted(
            merged.loc[merged["best_model_params"].isna(), "participant_id"].unique().tolist()
        )
        raise ValueError(
            "Could not resolve Step 4 winner parameter payloads for participants: "
            f"{missing_participants}"
        )
    return merged[["participant_id", "winner_model_name", "best_model_params"]].copy()


def _save_step5_tables(
    *,
    step5_tables: dict[str, pd.DataFrame],
    tables_dir: Path,
) -> dict[str, str]:
    table_paths: dict[str, str] = {}
    for table_name in _STEP5_TABLE_NAMES:
        table_df = step5_tables.get(table_name, pd.DataFrame())
        table_path = save_table_csv(table_df, tables_dir / f"{table_name}.csv")
        table_paths[table_name] = str(table_path)
    return table_paths


def _run_step5_stage(
    *,
    df_all: pd.DataFrame,
    run_id: str,
    output_root: str | Path,
    step3_soft_gate: dict[str, Any],
    step4_tables: dict[str, pd.DataFrame],
    normalized_step5_config: dict[str, object],
    step5_overwrite: bool,
) -> dict[str, Any]:
    """Execute and persist Step 5 diagnostics from Step 4 winner outputs.

    Args:
        df_all: Preprocessed participant table.
        run_id: Shared run identifier.
        output_root: Elias output root path.
        step3_soft_gate: Step 3 soft-gate payload for recovery-aware conclusion.
        step4_tables: Step 4 persisted in-memory tables.
        normalized_step5_config: Validated Step 5 configuration.
        step5_overwrite: Whether existing `step5` folder should be replaced.

    Returns:
        Dictionary containing Step 5 paths, status, tables, and error metadata.
    """
    step5_paths = prepare_run_dir(
        output_root,
        pipeline_name="step5",
        run_id=str(run_id),
        overwrite=bool(step5_overwrite),
    )

    reports_dir = step5_paths["run_dir"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    step5_status = "completed"
    step5_error_message = ""
    step5_report_path = ""
    step5_error_log_path = ""
    step5_tables: dict[str, pd.DataFrame] = {
        table_name: pd.DataFrame() for table_name in _STEP5_TABLE_NAMES
    }
    step5_table_paths: dict[str, str] = {}

    try:
        # Step 5 diagnostics are computed from persisted Step 4 winners/params.
        winner_parameter_table = _build_winner_parameter_table(step4_tables)
        subjective_h_table = step4_tables.get("participant_subjective_h_train", pd.DataFrame())
        if not subjective_h_table.empty:
            df_step5_ready = attach_subjective_h_from_train(
                df_all,
                subjective_h_table,
                participant_col="participant_id",
                block_col="block_id",
                fitted_h_col="fitted_subjective_h",
                output_h_col="H",
            )
            df_step5_ready = build_normative_belief_columns(
                df_step5_ready,
                participant_col="participant_id",
                block_col="block_id",
                trial_col="trial_index",
                llr_col="LLR",
                hazard_col="H",
                output_prev_col="prev_normative_belief_L",
                output_curr_col="normative_belief_L",
                output_psi_col="psi_t",
            )
        else:
            # Keep legacy compatibility for old runs that predate subjective-H export.
            df_step5_ready = df_all.copy()
            if "H" not in df_step5_ready.columns:
                if "subjective_h_snapshot" in df_step5_ready.columns:
                    df_step5_ready["H"] = pd.to_numeric(
                        df_step5_ready["subjective_h_snapshot"],
                        errors="coerce",
                    )
                elif "hazard_rate" in df_step5_ready.columns:
                    df_step5_ready["H"] = pd.to_numeric(
                        df_step5_ready["hazard_rate"],
                        errors="coerce",
                    )
                else:
                    raise ValueError(
                        "Step 5 fallback path could not derive H: expected either "
                        "`subjective_h_snapshot` or `hazard_rate`."
                    )
            df_step5_ready = build_normative_belief_columns(
                df_step5_ready,
                participant_col="participant_id",
                block_col="block_id",
                trial_col="trial_index",
                llr_col="LLR",
                hazard_col="H",
                output_prev_col="prev_normative_belief_L",
                output_curr_col="normative_belief_L",
                output_psi_col="psi_t",
            )

        ppc_outputs = run_posterior_predictive_checks(
            df_step5_ready,
            winner_parameter_table,
            run_id=str(run_id),
            n_sims_per_trial=int(normalized_step5_config["ppc_n_sims_per_trial"]),
            rt_bin_width_ms=float(normalized_step5_config["rt_bin_width_ms"]),
            rt_max_ms=float(normalized_step5_config["rt_max_ms"]),
            eps=float(normalized_step5_config["eps"]),
            random_seed=int(normalized_step5_config["random_seed"]),
            workers=int(normalized_step5_config["workers"]),
        )
        hazard_outputs = run_change_hazard_checks(df_step5_ready, winner_parameter_table)
        latent_outputs = run_latent_reporting(
            df_step5_ready,
            winner_parameter_table,
            run_id=str(run_id),
            ddm_n_samples_per_trial=int(normalized_step5_config["ddm_n_samples_per_trial"]),
            latent_cont_noise_std=float(normalized_step5_config["latent_cont_noise_std"]),
            random_seed=int(normalized_step5_config["random_seed"]) + 37,
            workers=int(normalized_step5_config["workers"]),
        )

        conclusion_table = build_recovery_aware_conclusion(
            step3_soft_gate=dict(step3_soft_gate),
            step4_group_winner_summary=step4_tables.get("group_winner_summary", pd.DataFrame()),
            step5_posterior_predictive_block=ppc_outputs["posterior_predictive_block"],
            step5_hazard_signature_block=hazard_outputs["hazard_signature_block"],
            step5_latent_quantities_block=latent_outputs["latent_quantities_block"],
        )

        step5_tables = {
            "step5_posterior_predictive_trial": ppc_outputs["posterior_predictive_trial"],
            "step5_posterior_predictive_block": ppc_outputs["posterior_predictive_block"],
            "step5_hazard_signature_trial": hazard_outputs["hazard_signature_trial"],
            "step5_hazard_signature_block": hazard_outputs["hazard_signature_block"],
            "step5_latent_trajectories_trial": latent_outputs["latent_trajectories_trial"],
            "step5_latent_quantities_block": latent_outputs["latent_quantities_block"],
            "step5_final_conclusion": conclusion_table,
        }
        step5_table_paths = _save_step5_tables(
            step5_tables=step5_tables,
            tables_dir=step5_paths["tables_dir"],
        )
        report_path_obj = write_step5_report_markdown(
            reports_dir / "step5_report.md",
            run_id=str(run_id),
            step3_soft_gate=dict(step3_soft_gate),
            step4_group_winner_summary=step4_tables.get("group_winner_summary", pd.DataFrame()),
            conclusion_table=conclusion_table,
            step5_posterior_predictive_block=ppc_outputs["posterior_predictive_block"],
            step5_hazard_signature_block=hazard_outputs["hazard_signature_block"],
            step5_latent_quantities_block=latent_outputs["latent_quantities_block"],
        )
        step5_report_path = str(report_path_obj)
    except Exception as exc:
        step5_status = "failed"
        step5_error_message = str(exc)
        error_log_path = step5_paths["logs_dir"] / "step5_error.txt"
        error_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        step5_error_log_path = str(error_log_path)

    return {
        "step5_paths": step5_paths,
        "step5_status": step5_status,
        "step5_error_message": step5_error_message,
        "step5_report_path": step5_report_path,
        "step5_error_log_path": step5_error_log_path,
        "step5_tables": step5_tables,
        "step5_table_paths": step5_table_paths,
    }


def run_step345_pipeline(
    df_all: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = "data/elias",
    step3_config: dict[str, object],
    step4_config: dict[str, object],
    step5_config: dict[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run Step 3, Step 4, and Step 5 with one unified run folder.

    Args:
        df_all: Preprocessed participant table with TRAIN/TEST split labels.
        run_id: Stable run identifier shared across all three steps.
        output_root: Absolute or repository-relative Elias output root.
        step3_config: Step 3 configuration payload.
        step4_config: Step 4 configuration payload.
        step5_config: Optional Step 5 override payload.
        overwrite: Whether to replace an existing unified run folder.

    Returns:
        Pipeline metadata, step outputs, and persisted artifact paths.

    Raises:
        Step5PipelineError: If Step 5 fails after Step 3 and Step 4 were persisted.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")

    normalized_step5_config = _normalize_step5_pipeline_config(step5_config)
    # Enforce one clean master folder per run to avoid cross-step stale artifacts.
    run_root = prepare_pipeline_run_root(
        output_root,
        run_id=str(run_id),
        overwrite=overwrite,
    )

    step3_output = run_step3_pipeline(
        df_all,
        run_id=str(run_id),
        output_root=output_root,
        config=step3_config,
        overwrite=False,
    )
    step4_output = run_step4_pipeline(
        df_all,
        run_id=str(run_id),
        output_root=output_root,
        config=step4_config,
        overwrite=False,
    )

    step5_stage = _run_step5_stage(
        df_all=df_all,
        run_id=str(run_id),
        output_root=output_root,
        step3_soft_gate=dict(step3_output["manifest"].get("soft_gate", {})),
        step4_tables=step4_output["tables"],
        normalized_step5_config=normalized_step5_config,
        step5_overwrite=False,
    )
    step5_paths = step5_stage["step5_paths"]
    step5_status = str(step5_stage["step5_status"])
    step5_error_message = str(step5_stage["step5_error_message"])
    step5_report_path = str(step5_stage["step5_report_path"])
    step5_error_log_path = str(step5_stage["step5_error_log_path"])
    step5_tables = dict(step5_stage["step5_tables"])
    step5_table_paths = dict(step5_stage["step5_table_paths"])

    config_payload = {
        "run_id": str(run_id),
        "run_root": str(run_root),
        "step3_config": step3_output["config"],
        "step4_config": step4_output["config"],
        "step5_config": normalized_step5_config,
        "step5_status": step5_status,
    }
    config_path = save_json(config_payload, run_root / "config.json")

    manifest_extra = {
        "step3": {
            "run_id": str(run_id),
            "run_dir": str(step3_output["run_dir"]),
            "status": str(step3_output["manifest"].get("status", "unknown")),
            "soft_gate": step3_output["manifest"].get("soft_gate", {}),
        },
        "step4": {
            "run_id": str(run_id),
            "run_dir": str(step4_output["run_dir"]),
            "status": str(step4_output["manifest"].get("status", "unknown")),
            "group_winner_model_name": step4_output["manifest"].get(
                "group_winner_model_name", "unknown"
            ),
        },
        "step5": {
            "run_id": str(run_id),
            "run_dir": str(step5_paths["run_dir"]),
        },
        "step5_status": step5_status,
        "step5_table_paths": step5_table_paths,
        "step5_report_path": step5_report_path,
    }
    if step5_error_message:
        manifest_extra["step5_error_message"] = step5_error_message
    if step5_error_log_path:
        manifest_extra["step5_error_log_path"] = step5_error_log_path

    manifest = build_basic_manifest(
        run_id=str(run_id),
        pipeline_name="step345_pipeline",
        output_root=output_root,
        run_dir=run_root,
        config_path=config_path,
        status="completed" if step5_status == "completed" else "step5_failed",
        extra=manifest_extra,
    )
    manifest_path = save_json(manifest, run_root / "manifest.json")

    output = {
        "run_id": str(run_id),
        "run_dir": run_root,
        "run_root": run_root,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "step3_run_id": str(run_id),
        "step4_run_id": str(run_id),
        "step5_run_id": str(run_id),
        "step3_output": step3_output,
        "step4_output": step4_output,
        "step5_tables": step5_tables,
        "step5_table_paths": step5_table_paths,
        "step5_report_path": step5_report_path,
        "step5_run_dir": step5_paths["run_dir"],
    }

    if step5_status != "completed":
        raise Step5PipelineError(
            "Step 5 failed. Reporting manifest and error log were persisted.",
            manifest_path=manifest_path,
            error_log_path=Path(step5_error_log_path) if step5_error_log_path else None,
        )

    return output


def run_step45_pipeline(
    df_all: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = "data/elias",
    step4_config: dict[str, object],
    step5_config: dict[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run Step 4 and Step 5 using an existing Step 3 run in the same run root.

    Args:
        df_all: Preprocessed participant table with TRAIN/TEST split labels.
        run_id: Stable run identifier shared across all steps.
        output_root: Absolute or repository-relative Elias output root.
        step4_config: Step 4 configuration payload.
        step5_config: Optional Step 5 override payload.
        overwrite: Whether existing Step 4/5 folders should be replaced.

    Returns:
        Pipeline metadata, step outputs, and persisted artifact paths.

    Raises:
        FileNotFoundError: If required Step 3 artifacts are missing.
        Step5PipelineError: If Step 5 fails after Step 4 was persisted.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")

    run_root = resolve_unified_run_root(output_root, str(run_id))
    step3_dir = run_root / "step3"
    if not step3_dir.exists():
        raise FileNotFoundError(
            f"Step 3 directory is required for pipeline-run-45 but was not found: {step3_dir}"
        )
    step3_output = load_step3_run(run_id=str(run_id), output_root=output_root)
    step3_soft_gate = dict(step3_output["manifest"].get("soft_gate", {}))

    run_root.mkdir(parents=True, exist_ok=True)
    normalized_step5_config = _normalize_step5_pipeline_config(step5_config)
    step4_output = run_step4_pipeline(
        df_all,
        run_id=str(run_id),
        output_root=output_root,
        config=step4_config,
        overwrite=bool(overwrite),
    )

    step5_stage = _run_step5_stage(
        df_all=df_all,
        run_id=str(run_id),
        output_root=output_root,
        step3_soft_gate=step3_soft_gate,
        step4_tables=step4_output["tables"],
        normalized_step5_config=normalized_step5_config,
        step5_overwrite=bool(overwrite),
    )
    step5_paths = step5_stage["step5_paths"]
    step5_status = str(step5_stage["step5_status"])
    step5_error_message = str(step5_stage["step5_error_message"])
    step5_report_path = str(step5_stage["step5_report_path"])
    step5_error_log_path = str(step5_stage["step5_error_log_path"])
    step5_tables = dict(step5_stage["step5_tables"])
    step5_table_paths = dict(step5_stage["step5_table_paths"])

    config_payload = {
        "run_id": str(run_id),
        "run_root": str(run_root),
        "step3_config": step3_output["config"],
        "step4_config": step4_output["config"],
        "step5_config": normalized_step5_config,
        "step5_status": step5_status,
        "execution_mode": "step45_only",
    }
    config_path = save_json(config_payload, run_root / "config.json")

    manifest_extra = {
        "step3": {
            "run_id": str(run_id),
            "run_dir": str(step3_output["run_dir"]),
            "status": str(step3_output["manifest"].get("status", "unknown")),
            "soft_gate": step3_soft_gate,
        },
        "step4": {
            "run_id": str(run_id),
            "run_dir": str(step4_output["run_dir"]),
            "status": str(step4_output["manifest"].get("status", "unknown")),
            "group_winner_model_name": step4_output["manifest"].get(
                "group_winner_model_name", "unknown"
            ),
        },
        "step5": {
            "run_id": str(run_id),
            "run_dir": str(step5_paths["run_dir"]),
        },
        "step5_status": step5_status,
        "step5_table_paths": step5_table_paths,
        "step5_report_path": step5_report_path,
    }
    if step5_error_message:
        manifest_extra["step5_error_message"] = step5_error_message
    if step5_error_log_path:
        manifest_extra["step5_error_log_path"] = step5_error_log_path

    manifest = build_basic_manifest(
        run_id=str(run_id),
        pipeline_name="step45_pipeline",
        output_root=output_root,
        run_dir=run_root,
        config_path=config_path,
        status="completed" if step5_status == "completed" else "step5_failed",
        extra=manifest_extra,
    )
    manifest_path = save_json(manifest, run_root / "manifest.json")

    output = {
        "run_id": str(run_id),
        "run_dir": run_root,
        "run_root": run_root,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "step3_run_id": str(run_id),
        "step4_run_id": str(run_id),
        "step5_run_id": str(run_id),
        "step3_output": step3_output,
        "step4_output": step4_output,
        "step5_tables": step5_tables,
        "step5_table_paths": step5_table_paths,
        "step5_report_path": step5_report_path,
        "step5_run_dir": step5_paths["run_dir"],
    }

    if step5_status != "completed":
        raise Step5PipelineError(
            "Step 5 failed. Reporting manifest and error log were persisted.",
            manifest_path=manifest_path,
            error_log_path=Path(step5_error_log_path) if step5_error_log_path else None,
        )

    return output


def run_step34_pipeline(
    df_all: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = "data/elias",
    step3_config: dict[str, object],
    step4_config: dict[str, object],
    overwrite: bool = False,
) -> dict[str, object]:
    """Compatibility wrapper that now executes Step 3, Step 4, and Step 5.

    Args:
        df_all: Preprocessed participant table.
        run_id: Stable run identifier shared across all steps.
        output_root: Absolute or repository-relative Elias output root.
        step3_config: Step 3 configuration payload.
        step4_config: Step 4 configuration payload.
        overwrite: Whether to replace an existing unified run folder.

    Returns:
        Output dictionary from `run_step345_pipeline`.
    """
    default_step5_seed = int(step4_config.get("random_seed", 0))
    return run_step345_pipeline(
        df_all,
        run_id=run_id,
        output_root=output_root,
        step3_config=step3_config,
        step4_config=step4_config,
        step5_config=build_step5_pipeline_config(random_seed=default_step5_seed),
        overwrite=overwrite,
    )
