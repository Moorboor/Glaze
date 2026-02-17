"""Data preparation and subjective-state construction for the Elias workflow.

Function Inventory:
- `load_participant_data`: Load participant CSV rows and normalize choices to `-1/+1`; called by `core_workflow.prepare_modeling_data` and notebook Step 1.
- `preprocess_loaded_participant_data`: Apply exclusions and train/test split; called by `core_workflow.prepare_modeling_data` and notebook Step 1.
- `fit_blockwise_subjective_h_choice_only`: Fit one subjective hazard value per participant-block from TRAIN choices; called by `core_workflow.prepare_modeling_data` and notebook Step 2.
- `attach_subjective_h_from_train`: Merge fitted blockwise hazard back onto all rows; called by `core_workflow.prepare_modeling_data` and notebook Step 2.
- `build_normative_belief_columns`: Reconstruct prior and posterior normative beliefs from LLR and H; called by `core_workflow.prepare_modeling_data` and notebook Step 2.
- `SubjectiveHGrid`: Grid configuration container used by subjective-H search internals.
- `glaze_psi`: Numerically stable Glaze prior transform used in normative recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    # Reuse shared loading/preprocessing behavior already used in this repository.
    from common_helpers.preprocessing import (
        load_participant_data as _shared_load_participant_data,
        preprocess_loaded_participant_data as _shared_preprocess_loaded_participant_data,
    )
except ModuleNotFoundError:
    # Make direct module use robust when PYTHONPATH does not already include `src/`.
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from common_helpers.preprocessing import (
        load_participant_data as _shared_load_participant_data,
        preprocess_loaded_participant_data as _shared_preprocess_loaded_participant_data,
    )


# Required fields for preprocessing after loading and initial schema checks.
PREPROCESS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "choice",
    "reaction_time_ms",
    "LLR",
    "belief_L",
    "hazard_rate",
    "noise_sigma",
)


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Validate that all required columns are present in a DataFrame."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric dtype with NaN on parse failures."""
    coerced = df.copy()
    for column in columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def _drop_non_finite_rows(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Drop rows containing non-finite values in selected numeric columns."""
    numeric_cols = list(numeric_columns)
    finite_mask = np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all(axis=1)
    dropped_rows = int((~finite_mask).sum())
    if dropped_rows > 0:
        # Warn instead of hard-failing so callers can still proceed on valid rows.
        warnings.warn(
            f"Dropping {dropped_rows} rows with non-finite numeric values during {context}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return df.loc[finite_mask].copy()


def _validate_reset_on(reset_on: tuple[str, ...]) -> tuple[str, ...]:
    """Validate reset policy values used by the shared loader."""
    allowed = {"participant", "block"}
    invalid = sorted(set(reset_on) - allowed)
    if invalid:
        raise ValueError(
            f"Invalid reset_on values: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return reset_on


def _normalize_choice_values_to_pm1(choice_values: np.ndarray) -> np.ndarray:
    """Normalize vectorized choices to signed coding `-1/+1` (mapping `0 -> -1`)."""
    raw = np.asarray(choice_values, dtype=float)
    valid_mask = np.isin(raw, (-1.0, 0.0, 1.0))
    if not bool(np.all(valid_mask)):
        invalid_values = np.unique(raw[~valid_mask]).tolist()
        raise ValueError(
            "Unsupported choice encoding. Expected values in {-1, 0, 1}; "
            f"found invalid values: {invalid_values}"
        )
    normalized = raw.copy()
    normalized[normalized == 0.0] = -1.0
    return normalized.astype(int)


def _normalize_choice_column_to_pm1(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame `choice` column to signed coding `-1/+1`."""
    _validate_required_columns(df, ("choice",), context="choice normalization")
    normalized = df.copy()
    normalized["choice"] = _normalize_choice_values_to_pm1(
        pd.to_numeric(normalized["choice"], errors="coerce").to_numpy(dtype=float)
    )
    return normalized


@dataclass(frozen=True)
class SubjectiveHGrid:
    """Grid definition used by blockwise subjective-hazard fitting."""

    h_start: float = 0.001
    h_end: float = 0.999
    h_step: float = 0.01

    def values(self) -> np.ndarray:
        """Return validated candidate hazard values for grid search."""
        if self.h_step <= 0.0:
            raise ValueError("h_step must be > 0")
        if not (0.0 < self.h_start < 1.0 and 0.0 < self.h_end < 1.0):
            raise ValueError("h_start and h_end must be in (0, 1)")
        if self.h_start > self.h_end:
            raise ValueError("h_start must be <= h_end")
        # Half-step slack ensures the inclusive upper endpoint under float rounding.
        return np.round(
            np.arange(self.h_start, self.h_end + (self.h_step / 2.0), self.h_step),
            12,
        )


def glaze_psi(l_prev: float, h: float) -> float:
    """Compute numerically stable Glaze prior transform (Eq. 2)."""
    h_clamped = float(np.clip(h, 1e-9, 1.0 - 1e-9))
    if abs(float(l_prev)) > 100.0:
        # For extreme logits, use asymptotic form to avoid overflow in exp terms.
        return float(np.sign(float(l_prev)) * np.log((1.0 - h_clamped) / h_clamped))

    stability_ratio = (1.0 - h_clamped) / h_clamped
    term_pos = stability_ratio + np.exp(-float(l_prev))
    term_neg = stability_ratio + np.exp(float(l_prev))
    return float(float(l_prev) + np.log(term_pos) - np.log(term_neg))


def _belief_trajectory_for_h(llr_values: np.ndarray, *, h: float) -> np.ndarray:
    """Build posterior belief trajectory for one fixed hazard candidate."""
    beliefs = np.zeros(len(llr_values), dtype=float)
    l_prev = 0.0
    for index, llr in enumerate(llr_values):
        psi_t = glaze_psi(l_prev, h)
        l_curr = psi_t + float(llr)
        beliefs[index] = l_curr
        l_prev = l_curr
    return beliefs


def _choice_nll_from_beliefs(
    beliefs: np.ndarray,
    choices_pm1: np.ndarray,
    *,
    beta: float,
) -> float:
    """Compute choice negative log-likelihood under logistic readout from beliefs."""
    logits = np.clip(float(beta) * np.asarray(beliefs, dtype=float), -60.0, 60.0)
    prob_right = 1.0 / (1.0 + np.exp(-logits))
    # Convert signed choices to binary right-choice indicator.
    y_right = (np.asarray(choices_pm1, dtype=int) == 1).astype(float)
    likelihood = np.where(y_right > 0.5, prob_right, 1.0 - prob_right)
    return float(-np.sum(np.log(np.maximum(likelihood, 1e-12))))


def load_participant_data(
    csv_path: str | Path = "data/participants.csv",
    participant_ids: list[str] | None = None,
    hazard_col: str = "subjective_h_snapshot",
    reset_on: tuple[str, ...] = ("participant", "block"),
) -> pd.DataFrame:
    """Load participant data and normalize observed choices to `-1/+1`.

    Notes:
        `hazard_col` is accepted for compatibility but ignored in active modeling.
        Hazard is inferred from TRAIN behavior and then attached as `H` later.
    """
    _validate_reset_on(reset_on)
    if str(hazard_col) != "subjective_h_snapshot":
        # Keep caller compatibility while making current behavior explicit.
        warnings.warn(
            "hazard_col is deprecated in active workflow and is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    loaded_df = _shared_load_participant_data(
        csv_path=csv_path,
        participant_ids=participant_ids,
        hazard_col="hazard_rate",
        reset_on=reset_on,
    )
    loaded_df = _normalize_choice_column_to_pm1(loaded_df)
    # Remove legacy state columns so state reconstruction happens in one place.
    loaded_df = loaded_df.drop(columns=["H", "prev_observed_belief_L"], errors="ignore")
    return loaded_df


def preprocess_loaded_participant_data(
    df_loaded: pd.DataFrame,
    *,
    required_cols: tuple[str, ...] = PREPROCESS_REQUIRED_COLUMNS,
    min_rt_ms: float = 150.0,
    max_rt_ms: float = 5000.0,
    train_trial_max_index: int = 30,
    expected_blocks_per_participant: int = 4,
    nominal_trials_per_block_before: int = 40,
) -> dict[str, object]:
    """Apply filtering and split assignment to loaded participant data."""
    _validate_required_columns(
        df_loaded,
        ("participant_id", "block_id", "trial_index", *required_cols),
        context="preprocessing",
    )
    return _shared_preprocess_loaded_participant_data(
        df_loaded,
        required_cols=required_cols,
        min_rt_ms=float(min_rt_ms),
        max_rt_ms=float(max_rt_ms),
        train_trial_max_index=int(train_trial_max_index),
        expected_blocks_per_participant=int(expected_blocks_per_participant),
        nominal_trials_per_block_before=int(nominal_trials_per_block_before),
    )


def fit_blockwise_subjective_h_choice_only(
    df: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    trial_col: str = "trial_index",
    split_col: str = "split",
    train_label: str = "TRAIN",
    llr_col: str = "LLR",
    choice_col: str = "choice",
    beta: float = 1.0,
    h_grid: SubjectiveHGrid | None = None,
) -> pd.DataFrame:
    """Fit one subjective hazard value per participant-block from TRAIN choices."""
    _validate_required_columns(
        df,
        (participant_col, block_col, trial_col, split_col, llr_col, choice_col),
        context="subjective-h fitting",
    )
    if beta <= 0.0:
        raise ValueError("beta must be > 0")

    grid = SubjectiveHGrid() if h_grid is None else h_grid
    h_values = grid.values()

    work = _coerce_numeric_columns(df, (block_col, trial_col, llr_col, choice_col))
    work = work.copy()
    work[participant_col] = work[participant_col].astype(str)
    work[split_col] = work[split_col].astype(str)

    train_df = work[work[split_col] == str(train_label)].copy()
    if train_df.empty:
        raise ValueError("No TRAIN rows available for subjective-h fitting.")

    train_df = _drop_non_finite_rows(
        train_df,
        (block_col, trial_col, llr_col, choice_col),
        context="subjective-h fitting",
    )
    if train_df.empty:
        raise ValueError("No finite TRAIN rows remain for subjective-h fitting.")

    train_df = train_df.sort_values([participant_col, block_col, trial_col], kind="mergesort")
    train_df[choice_col] = _normalize_choice_values_to_pm1(
        train_df[choice_col].to_numpy(dtype=float)
    )

    rows: list[dict[str, object]] = []
    for (participant_id, block_id), chunk in train_df.groupby(
        [participant_col, block_col],
        sort=True,
    ):
        llr_values = chunk[llr_col].to_numpy(dtype=float)
        choice_values = chunk[choice_col].to_numpy(dtype=int)

        best_h: float | None = None
        best_nll: float | None = None
        for h_candidate in h_values:
            beliefs = _belief_trajectory_for_h(llr_values, h=float(h_candidate))
            nll = _choice_nll_from_beliefs(beliefs, choice_values, beta=float(beta))
            # Tie handling intentionally keeps first hit, which implies lower-H preference.
            if best_nll is None or nll < best_nll:
                best_nll = float(nll)
                best_h = float(h_candidate)

        if best_h is None or best_nll is None:
            raise RuntimeError(
                f"Failed subjective-h fit for participant={participant_id}, block={block_id}."
            )

        n_trials = int(len(chunk))
        rows.append(
            {
                "participant_id": str(participant_id),
                "block_id": int(block_id),
                "n_train_trials": n_trials,
                "fitted_subjective_h": float(best_h),
                "fit_choice_nll": float(best_nll),
                "fit_choice_nll_per_trial": float(best_nll / max(n_trials, 1)),
                "beta_fixed": float(beta),
                "h_grid_start": float(grid.h_start),
                "h_grid_end": float(grid.h_end),
                "h_grid_step": float(grid.h_step),
            }
        )

    return pd.DataFrame(rows)


def attach_subjective_h_from_train(
    df: pd.DataFrame,
    subjective_h_table: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    fitted_h_col: str = "fitted_subjective_h",
    output_h_col: str = "H",
) -> pd.DataFrame:
    """Attach fitted TRAIN blockwise subjective hazard to TRAIN and TEST rows."""
    _validate_required_columns(df, (participant_col, block_col), context="attach subjective h")
    _validate_required_columns(
        subjective_h_table,
        (participant_col, block_col, fitted_h_col),
        context="subjective-h table",
    )

    output = df.copy()
    output[participant_col] = output[participant_col].astype(str)
    output[block_col] = pd.to_numeric(output[block_col], errors="coerce")
    if output_h_col in output.columns:
        # Drop stale H columns to avoid merge suffix artifacts.
        output = output.drop(columns=[output_h_col])

    h_table = subjective_h_table[[participant_col, block_col, fitted_h_col]].copy()
    h_table[participant_col] = h_table[participant_col].astype(str)
    h_table[block_col] = pd.to_numeric(h_table[block_col], errors="coerce")

    output = output.merge(
        h_table.rename(columns={fitted_h_col: output_h_col}),
        on=[participant_col, block_col],
        how="left",
    )

    h_numeric = pd.to_numeric(output[output_h_col], errors="coerce")
    missing_mask = ~np.isfinite(h_numeric.to_numpy(dtype=float))
    if bool(np.any(missing_mask)):
        # Fallback keeps flow robust if a block lost all TRAIN rows due to filtering.
        fallback_h = float(pd.to_numeric(h_table[fitted_h_col], errors="coerce").dropna().mean())
        if not np.isfinite(fallback_h):
            fallback_h = 0.1
        output.loc[missing_mask, output_h_col] = float(fallback_h)

    output[output_h_col] = pd.to_numeric(output[output_h_col], errors="coerce").astype(float)
    output["subjective_h_fallback_used"] = missing_mask.astype(int)
    return output


def build_normative_belief_columns(
    df: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    trial_col: str = "trial_index",
    llr_col: str = "LLR",
    hazard_col: str = "H",
    output_prev_col: str = "prev_normative_belief_L",
    output_curr_col: str = "normative_belief_L",
    output_psi_col: str = "psi_t",
) -> pd.DataFrame:
    """Build recursive normative prior/posterior belief columns."""
    _validate_required_columns(
        df,
        (participant_col, block_col, trial_col, llr_col, hazard_col),
        context="normative-belief reconstruction",
    )

    output = _coerce_numeric_columns(df, (block_col, trial_col, llr_col, hazard_col)).copy()
    output[participant_col] = output[participant_col].astype(str)

    finite_mask = np.isfinite(
        output[[block_col, trial_col, llr_col, hazard_col]].to_numpy(dtype=float)
    ).all(axis=1)
    if not bool(np.all(finite_mask)):
        bad_rows = int((~finite_mask).sum())
        raise ValueError(
            f"Found {bad_rows} non-finite rows while building normative belief columns."
        )

    # Preserve original order while running recursion on a sorted view.
    output = output.reset_index(names="_orig_index")
    output = output.sort_values([participant_col, block_col, trial_col], kind="mergesort").reset_index(
        drop=True
    )

    prev_values = np.zeros(len(output), dtype=float)
    psi_values = np.zeros(len(output), dtype=float)
    curr_values = np.zeros(len(output), dtype=float)

    for _, chunk in output.groupby([participant_col, block_col], sort=False):
        row_positions = chunk.index.to_numpy(dtype=int)
        llr_vals = chunk[llr_col].to_numpy(dtype=float)
        h_vals = chunk[hazard_col].to_numpy(dtype=float)

        l_prev = 0.0
        for local_index, (llr_t, h_t) in enumerate(zip(llr_vals, h_vals, strict=True)):
            row_index = int(row_positions[local_index])
            prev_values[row_index] = float(l_prev)
            psi_t = glaze_psi(l_prev, float(h_t))
            l_curr = float(psi_t + float(llr_t))
            psi_values[row_index] = float(psi_t)
            curr_values[row_index] = float(l_curr)
            l_prev = l_curr

    output[output_prev_col] = prev_values
    output[output_psi_col] = psi_values
    output[output_curr_col] = curr_values

    output = output.sort_values("_orig_index", kind="mergesort").drop(columns=["_orig_index"])
    return output.reset_index(drop=True)
