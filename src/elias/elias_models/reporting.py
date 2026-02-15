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
    prepare_run_dir,
    save_json,
    save_table_csv,
)
from .surrogate_recovery import run_step3_pipeline
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
) -> dict[str, object]:
    """Build canonical Step 5 pipeline configuration."""
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

    return {
        "ppc_n_sims_per_trial": int(ppc_n_sims_per_trial),
        "ddm_n_samples_per_trial": int(ddm_n_samples_per_trial),
        "rt_bin_width_ms": float(rt_bin_width_ms),
        "rt_max_ms": float(rt_max_ms),
        "eps": float(eps),
        "random_seed": int(random_seed),
        "latent_cont_noise_std": float(latent_cont_noise_std),
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
    """Run Step 3, Step 4, and Step 5 pipelines with one linked reporting manifest."""
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")

    normalized_step5_config = _normalize_step5_pipeline_config(step5_config)
    step3_run_id = f"{str(run_id)}__step3"
    step4_run_id = f"{str(run_id)}__step4"

    step3_output = run_step3_pipeline(
        df_all,
        run_id=step3_run_id,
        output_root=output_root,
        config=step3_config,
        overwrite=overwrite,
    )
    step4_output = run_step4_pipeline(
        df_all,
        run_id=step4_run_id,
        output_root=output_root,
        config=step4_config,
        overwrite=overwrite,
    )

    report_paths = prepare_run_dir(
        output_root,
        pipeline_name="reporting",
        run_id=str(run_id),
        overwrite=overwrite,
    )

    reports_dir = report_paths["run_dir"] / "reports"
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
        winner_parameter_table = _build_winner_parameter_table(step4_output["tables"])
        ppc_outputs = run_posterior_predictive_checks(
            df_all,
            winner_parameter_table,
            n_sims_per_trial=int(normalized_step5_config["ppc_n_sims_per_trial"]),
            rt_bin_width_ms=float(normalized_step5_config["rt_bin_width_ms"]),
            rt_max_ms=float(normalized_step5_config["rt_max_ms"]),
            eps=float(normalized_step5_config["eps"]),
            random_seed=int(normalized_step5_config["random_seed"]),
        )
        hazard_outputs = run_change_hazard_checks(df_all, winner_parameter_table)
        latent_outputs = run_latent_reporting(
            df_all,
            winner_parameter_table,
            ddm_n_samples_per_trial=int(normalized_step5_config["ddm_n_samples_per_trial"]),
            latent_cont_noise_std=float(normalized_step5_config["latent_cont_noise_std"]),
            random_seed=int(normalized_step5_config["random_seed"]) + 37,
        )

        conclusion_table = build_recovery_aware_conclusion(
            step3_soft_gate=dict(step3_output["manifest"].get("soft_gate", {})),
            step4_group_winner_summary=step4_output["tables"].get(
                "group_winner_summary", pd.DataFrame()
            ),
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
            tables_dir=report_paths["tables_dir"],
        )
        report_path_obj = write_step5_report_markdown(
            reports_dir / "step5_report.md",
            run_id=str(run_id),
            step3_soft_gate=dict(step3_output["manifest"].get("soft_gate", {})),
            step4_group_winner_summary=step4_output["tables"].get(
                "group_winner_summary", pd.DataFrame()
            ),
            conclusion_table=conclusion_table,
            step5_posterior_predictive_block=ppc_outputs["posterior_predictive_block"],
            step5_hazard_signature_block=hazard_outputs["hazard_signature_block"],
            step5_latent_quantities_block=latent_outputs["latent_quantities_block"],
        )
        step5_report_path = str(report_path_obj)
    except Exception as exc:
        step5_status = "failed"
        step5_error_message = str(exc)
        error_log_path = report_paths["logs_dir"] / "step5_error.txt"
        error_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        step5_error_log_path = str(error_log_path)

    config_payload = {
        "step3_run_id": step3_run_id,
        "step4_run_id": step4_run_id,
        "step3_config": step3_output["config"],
        "step4_config": step4_output["config"],
        "step5_config": normalized_step5_config,
        "step5_status": step5_status,
    }
    config_path = save_json(config_payload, report_paths["run_dir"] / "config.json")

    manifest_extra = {
        "step3": {
            "run_id": step3_run_id,
            "run_dir": str(step3_output["run_dir"]),
            "status": str(step3_output["manifest"].get("status", "unknown")),
            "soft_gate": step3_output["manifest"].get("soft_gate", {}),
        },
        "step4": {
            "run_id": step4_run_id,
            "run_dir": str(step4_output["run_dir"]),
            "status": str(step4_output["manifest"].get("status", "unknown")),
            "group_winner_model_name": step4_output["manifest"].get(
                "group_winner_model_name", "unknown"
            ),
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
        pipeline_name="reporting",
        output_root=output_root,
        run_dir=report_paths["run_dir"],
        config_path=config_path,
        status="completed" if step5_status == "completed" else "step5_failed",
        extra=manifest_extra,
    )
    manifest_path = save_json(manifest, report_paths["run_dir"] / "manifest.json")

    output = {
        "run_id": str(run_id),
        "run_dir": report_paths["run_dir"],
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "step3_run_id": step3_run_id,
        "step4_run_id": step4_run_id,
        "step3_output": step3_output,
        "step4_output": step4_output,
        "step5_tables": step5_tables,
        "step5_table_paths": step5_table_paths,
        "step5_report_path": step5_report_path,
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
    """Compatibility wrapper that now executes Step 3, 4, and 5."""
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
