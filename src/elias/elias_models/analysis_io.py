#
# Post-run analysis data loading and run-consistency checks for Step 3/4/5.
# Main functions: resolve_run_root, load_step3_artifacts, load_step4_artifacts,
# load_step5_artifacts, validate_run_manifest_consistency.

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from .results_store import (
    load_json,
    load_table_csv,
    resolve_elias_data_root,
)
from .surrogate_recovery import load_step3_run
from .train_test_eval import load_step4_run

_STEP5_TABLE_NAMES: tuple[str, ...] = (
    "step5_posterior_predictive_trial",
    "step5_posterior_predictive_block",
    "step5_hazard_signature_trial",
    "step5_hazard_signature_block",
    "step5_latent_trajectories_trial",
    "step5_latent_quantities_block",
    "step5_final_conclusion",
)


def resolve_run_root(run_id: str, output_root: str | Path = "data/elias") -> Path:
    """Resolve the unified run root for persisted Step 3/4/5 artifacts.

    Args:
        run_id: Stable run identifier.
        output_root: Repository-relative or absolute Elias output root.

    Returns:
        Absolute run root path, for example `.../data/elias/runs/<run_id>`.

    Raises:
        ValueError: If `run_id` is empty.
        FileNotFoundError: If the resolved run root does not exist.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")

    run_root = resolve_elias_data_root(output_root) / "runs" / str(run_id)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    return run_root


def _read_text_if_exists(path: Path) -> str:
    """Read UTF-8 text from disk if present.

    Args:
        path: Text file path.

    Returns:
        File text if present, otherwise an empty string.
    """
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _iter_nested_strings(value: Any) -> list[str]:
    """Collect all nested string values from dict/list payloads.

    Args:
        value: Arbitrary nested payload.

    Returns:
        Flat list of nested strings.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_iter_nested_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_iter_nested_strings(item))
        return out
    return []


def load_step3_artifacts(
    run_id: str,
    output_root: str | Path = "data/elias",
) -> dict[str, object]:
    """Load persisted Step 3 artifacts for post-run analysis.

    Args:
        run_id: Stable run identifier.
        output_root: Repository-relative or absolute Elias output root.

    Returns:
        Dictionary containing run metadata and Step 3 tables.
    """
    run_root = resolve_run_root(run_id, output_root=output_root)
    step3 = load_step3_run(run_id=run_id, output_root=output_root)
    return {
        "run_id": str(run_id),
        "run_root": run_root,
        "step_dir": run_root / "step3",
        "manifest": dict(step3["manifest"]),
        "config": dict(step3["config"]),
        "tables": dict(step3["tables"]),
        "raw": step3,
    }


def load_step4_artifacts(
    run_id: str,
    output_root: str | Path = "data/elias",
) -> dict[str, object]:
    """Load persisted Step 4 artifacts for post-run analysis.

    Args:
        run_id: Stable run identifier.
        output_root: Repository-relative or absolute Elias output root.

    Returns:
        Dictionary containing run metadata and Step 4 tables.
    """
    run_root = resolve_run_root(run_id, output_root=output_root)
    step4 = load_step4_run(run_id=run_id, output_root=output_root)
    return {
        "run_id": str(run_id),
        "run_root": run_root,
        "step_dir": run_root / "step4",
        "manifest": dict(step4["manifest"]),
        "config": dict(step4["config"]),
        "tables": dict(step4["tables"]),
        "raw": step4,
    }


def load_step5_artifacts(
    run_id: str,
    output_root: str | Path = "data/elias",
) -> dict[str, object]:
    """Load persisted Step 5 tables/report and optional failure log.

    Args:
        run_id: Stable run identifier.
        output_root: Repository-relative or absolute Elias output root.

    Returns:
        Dictionary with loaded Step 5 tables, report text, and optional error log text.
    """
    run_root = resolve_run_root(run_id, output_root=output_root)
    step5_dir = run_root / "step5"
    if not step5_dir.exists():
        raise FileNotFoundError(f"Step 5 directory not found: {step5_dir}")

    tables_dir = step5_dir / "tables"
    tables: dict[str, pd.DataFrame] = {}
    table_paths: dict[str, str] = {}
    missing_tables: list[str] = []

    for table_name in _STEP5_TABLE_NAMES:
        table_path = tables_dir / f"{table_name}.csv"
        if table_path.exists():
            tables[table_name] = load_table_csv(table_path)
            table_paths[table_name] = str(table_path)
        else:
            missing_tables.append(table_name)

    if missing_tables:
        warnings.warn(
            "Step 5 missing table files for run "
            f"'{run_id}': {missing_tables}",
            stacklevel=2,
        )

    report_path = step5_dir / "reports" / "step5_report.md"
    error_log_path = step5_dir / "logs" / "step5_error.txt"

    return {
        "run_id": str(run_id),
        "run_root": run_root,
        "step_dir": step5_dir,
        "tables": tables,
        "table_paths": table_paths,
        "missing_tables": missing_tables,
        "report_path": report_path,
        "report_text": _read_text_if_exists(report_path),
        "error_log_path": error_log_path,
        "error_log_text": _read_text_if_exists(error_log_path),
    }


def validate_run_manifest_consistency(
    run_id: str,
    output_root: str | Path = "data/elias",
) -> dict[str, object]:
    """Validate internal run-id/path consistency across persisted manifests.

    Args:
        run_id: Stable run identifier.
        output_root: Repository-relative or absolute Elias output root.

    Returns:
        Validation payload with status, checked files, and warning/error messages.
    """
    run_root = resolve_run_root(run_id, output_root=output_root)
    expected_fragment = f"/runs/{run_id}"

    files_to_check = {
        "pipeline_config": run_root / "config.json",
        "pipeline_manifest": run_root / "manifest.json",
        "step3_manifest": run_root / "step3" / "manifest.json",
        "step4_manifest": run_root / "step4" / "manifest.json",
    }

    warnings_list: list[str] = []
    errors_list: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}

    for label, path in files_to_check.items():
        if not path.exists():
            warnings_list.append(f"Missing file: {path}")
            continue
        try:
            payload = load_json(path)
        except Exception as exc:  # pragma: no cover - defensive parse path
            errors_list.append(f"Could not parse {path}: {exc}")
            continue
        payloads[label] = payload

        payload_run_id = payload.get("run_id")
        if payload_run_id is not None and str(payload_run_id) != str(run_id):
            errors_list.append(
                f"{label} has run_id={payload_run_id}, expected {run_id}."
            )

        for text_value in _iter_nested_strings(payload):
            if "/runs/" in text_value and expected_fragment not in text_value:
                warnings_list.append(
                    f"{label} contains path with mismatched run fragment: {text_value}"
                )

    is_consistent = not errors_list
    return {
        "run_id": str(run_id),
        "run_root": str(run_root),
        "is_consistent": bool(is_consistent),
        "warnings": warnings_list,
        "errors": errors_list,
        "files_checked": {k: str(v) for k, v in files_to_check.items()},
        "payloads_loaded": sorted(payloads.keys()),
    }
