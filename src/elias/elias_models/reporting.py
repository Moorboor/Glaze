"""Cross-step reporting orchestration helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .results_store import build_basic_manifest, prepare_run_dir, save_json
from .surrogate_recovery import run_step3_pipeline
from .train_test_eval import run_step4_pipeline


def run_step34_pipeline(
    df_all: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = "data/elias",
    step3_config: dict[str, object],
    step4_config: dict[str, object],
    overwrite: bool = False,
) -> dict[str, object]:
    """Run Step 3 and Step 4 pipelines and persist a linked master manifest.

    Args:
        df_all: Preprocessed participant dataset.
        run_id: Master run identifier.
        output_root: Root output folder, defaulting to `data/elias`.
        step3_config: Configuration dictionary for Step 3.
        step4_config: Configuration dictionary for Step 4.
        overwrite: Whether to overwrite existing run directories.

    Returns:
        Dictionary with linked Step 3/Step 4 outputs and master manifest paths.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")

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
    config_payload = {
        "step3_run_id": step3_run_id,
        "step4_run_id": step4_run_id,
        "step3_config": step3_output["config"],
        "step4_config": step4_output["config"],
        "step5_status": "not_implemented",
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
        "step5_status": "not_implemented",
    }
    manifest = build_basic_manifest(
        run_id=str(run_id),
        pipeline_name="reporting",
        output_root=output_root,
        run_dir=report_paths["run_dir"],
        config_path=config_path,
        status="step5_not_implemented",
        extra=manifest_extra,
    )
    manifest_path = save_json(manifest, report_paths["run_dir"] / "manifest.json")

    return {
        "run_id": str(run_id),
        "run_dir": report_paths["run_dir"],
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "step3_run_id": step3_run_id,
        "step4_run_id": step4_run_id,
        "step3_output": step3_output,
        "step4_output": step4_output,
    }
