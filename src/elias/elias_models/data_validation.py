from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric dtype."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_non_finite_rows(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Drop rows containing non-finite numeric values."""
    cols = list(numeric_columns)
    finite_mask = np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite_mask).sum())

    if dropped > 0:
        warnings.warn(
            f"Dropping {dropped} rows with non-finite numeric values during {context}.",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df.loc[finite_mask].copy()

    return df


def _validate_reset_on(reset_on: tuple[str, ...]) -> tuple[str, ...]:
    """Validate reset policy flags."""
    allowed = {"participant", "block"}
    invalid = sorted(set(reset_on) - allowed)
    if invalid:
        raise ValueError(
            f"Invalid reset_on values: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return reset_on


def _normalize_choice_column_to_pm1(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize observed choices to signed coding (`-1`, `+1`)."""
    _validate_required_columns(df, ["choice"], context="choice normalization")

    out = df.copy()
    raw_choice = pd.to_numeric(out["choice"], errors="coerce")
    allowed_values = {-1.0, 0.0, 1.0}
    valid_mask = raw_choice.isin(allowed_values)
    if not bool(valid_mask.all()):
        invalid_values = (
            out.loc[~valid_mask, "choice"]
            .drop_duplicates()
            .astype(str)
            .sort_values()
            .tolist()
        )
        raise ValueError(
            "Unsupported choice encoding. Expected values in {-1, 0, 1} "
            f"before normalization; found invalid values: {invalid_values}"
        )

    choice_pm1 = raw_choice.replace({0.0: -1.0})
    out["choice"] = choice_pm1.astype(int)
    return out


def _normalize_choice_values_to_pm1(choice_values: np.ndarray) -> np.ndarray:
    """Normalize vectorized choice values to signed coding (`-1`, `+1`)."""
    raw = np.asarray(choice_values, dtype=float)
    valid_mask = np.isin(raw, (-1.0, 0.0, 1.0))
    if not bool(np.all(valid_mask)):
        invalid_values = np.unique(raw[~valid_mask]).tolist()
        raise ValueError(
            "Unsupported choice encoding for scoring. Expected values in {-1, 0, 1}; "
            f"found invalid values: {invalid_values}"
        )

    normalized = raw.copy()
    normalized[normalized == 0.0] = -1.0
    return normalized.astype(int)
