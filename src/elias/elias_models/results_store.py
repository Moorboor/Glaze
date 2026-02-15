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
    """Resolve the Elias results root path relative to repository root."""
    output_path = Path(output_root)
    if output_path.is_absolute():
        return output_path

    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / output_path).resolve()


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prepare_run_dir(
    output_root: str | Path,
    *,
    pipeline_name: str,
    run_id: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create and return canonical directories for one pipeline run."""
    if not str(run_id).strip():
        raise ValueError("run_id must not be empty.")

    data_root = resolve_elias_data_root(output_root)

    # Create future-ready roots for cross-step persistence.
    (data_root / "participant_fit" / "runs").mkdir(parents=True, exist_ok=True)
    (data_root / "reporting" / "runs").mkdir(parents=True, exist_ok=True)

    runs_root = data_root / pipeline_name / "runs"
    run_dir = runs_root / str(run_id)
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
        "runs_root": runs_root,
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "logs_dir": logs_dir,
    }


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Save JSON payload to disk with UTF-8 encoding."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON payload from disk."""
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def _encode_object_cell(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _JSON_PREFIX + json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, (dict, list, tuple)):
        return _JSON_PREFIX + json.dumps(value, separators=(",", ":"))
    return value


def _decode_object_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_JSON_PREFIX):
        return json.loads(value[len(_JSON_PREFIX) :])
    return value


def save_table_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save DataFrame as CSV, serializing object payloads to JSON strings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    encoded = df.copy()
    object_columns = encoded.select_dtypes(include=["object"]).columns.tolist()
    for col in object_columns:
        encoded[col] = encoded[col].map(_encode_object_cell)

    encoded.to_csv(target, index=False)
    return target


def load_table_csv(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from CSV and decode serialized object payloads."""
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
    """Build a standard run manifest payload."""
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
