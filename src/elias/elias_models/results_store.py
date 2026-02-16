#
# Storage helpers for all Elias pipelines.
# Main functions: resolve_elias_data_root, resolve_unified_run_root,
# prepare_pipeline_run_root, prepare_run_dir, save/load JSON and CSV tables.

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_JSON_PREFIX = "__JSON__::"


def resolve_elias_data_root(output_root: str | Path = "data/elias") -> Path:
    """Resolve the absolute Elias output root.

    Args:
        output_root: Absolute or repository-relative data root.

    Returns:
        Absolute path for the Elias output root.
    """
    output_path = Path(output_root)
    if output_path.is_absolute():
        return output_path

    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / output_path).resolve()


def resolve_unified_run_root(output_root: str | Path, run_id: str) -> Path:
    """Resolve the canonical per-run folder for unified storage.

    Args:
        output_root: Absolute or repository-relative Elias output root.
        run_id: Stable run identifier.

    Returns:
        Absolute run-root path `data_root/runs/<run_id>`.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")
    return resolve_elias_data_root(output_root) / "runs" / str(run_id)


def _timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp with second resolution.

    Returns:
        Timestamp string in UTC.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prepare_pipeline_run_root(
    output_root: str | Path,
    *,
    run_id: str,
    overwrite: bool = False,
) -> Path:
    """Create the unified root folder for a Step 3/4/5 pipeline run.

    Args:
        output_root: Absolute or repository-relative Elias output root.
        run_id: Stable run identifier.
        overwrite: Whether an existing run root should be removed first.

    Returns:
        Absolute unified run-root path.

    Raises:
        FileExistsError: If the run root exists and `overwrite` is False.
    """
    run_root = resolve_unified_run_root(output_root, run_id)

    if run_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_root}. "
                "Set overwrite=True to replace it."
            )
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def prepare_run_dir(
    output_root: str | Path,
    *,
    pipeline_name: str,
    run_id: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create canonical folders for one step inside a unified run folder.

    Args:
        output_root: Absolute or repository-relative Elias output root.
        pipeline_name: Step folder name, for example `step3`, `step4`, or `step5`.
        run_id: Stable run identifier shared across steps.
        overwrite: Whether an existing step folder should be removed first.

    Returns:
        Dictionary with canonical directories for the step.

    Raises:
        FileExistsError: If the step folder exists and `overwrite` is False.
    """
    if not str(pipeline_name).strip():
        raise ValueError("pipeline_name must not be empty.")

    data_root = resolve_elias_data_root(output_root)
    run_root = resolve_unified_run_root(output_root, run_id)
    run_dir = run_root / str(pipeline_name)
    tables_dir = run_dir / "tables"
    logs_dir = run_dir / "logs"

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Set overwrite=True to replace it."
            )
        shutil.rmtree(run_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "data_root": data_root,
        "run_root": run_root,
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "logs_dir": logs_dir,
    }


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON object to disk.

    Args:
        payload: JSON-serializable dictionary payload.
        path: Destination file path.

    Returns:
        Path to the written JSON file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON payload.
    """
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def _encode_object_cell(value: Any) -> Any:
    """Encode complex object values for CSV-safe storage.

    Args:
        value: Raw DataFrame cell value.

    Returns:
        Encoded cell value.
    """
    if isinstance(value, np.ndarray):
        return _JSON_PREFIX + json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, (dict, list, tuple)):
        return _JSON_PREFIX + json.dumps(value, separators=(",", ":"))
    return value


def _decode_object_cell(value: Any) -> Any:
    """Decode object values stored by `_encode_object_cell`.

    Args:
        value: Encoded DataFrame cell value.

    Returns:
        Decoded cell value.
    """
    if isinstance(value, str) and value.startswith(_JSON_PREFIX):
        return json.loads(value[len(_JSON_PREFIX) :])
    return value


def save_table_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame to CSV with safe object serialization.

    Args:
        df: Source DataFrame.
        path: Destination CSV path.

    Returns:
        Path to the written CSV file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    encoded = df.copy()
    object_columns = encoded.select_dtypes(include=["object"]).columns.tolist()
    for col in object_columns:
        encoded[col] = encoded[col].map(_encode_object_cell)

    encoded.to_csv(target, index=False)
    return target


def load_table_csv(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from CSV and decode object payloads.

    Args:
        path: Source CSV path.

    Returns:
        Decoded DataFrame.
    """
    target = Path(path)
    df = pd.read_csv(target)

    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_columns:
        df[col] = df[col].map(_decode_object_cell)

    return df


def build_basic_manifest(
    *,
    run_id: str,
    pipeline_name: str,
    output_root: str | Path,
    run_dir: Path,
    config_path: Path,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard manifest payload for persisted runs.

    Args:
        run_id: Stable run identifier.
        pipeline_name: Step or pipeline label.
        output_root: Absolute or repository-relative Elias output root.
        run_dir: Absolute run directory for the manifest owner.
        config_path: Absolute path to the persisted config file.
        status: Run status label.
        extra: Optional additional manifest fields.

    Returns:
        Manifest payload dictionary.
    """
    payload = {
        "run_id": str(run_id),
        "pipeline_name": str(pipeline_name),
        "created_at_utc": _timestamp_utc(),
        "output_root": str(resolve_elias_data_root(output_root)),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "status": str(status),
    }
    if extra:
        payload.update(extra)
    return payload
