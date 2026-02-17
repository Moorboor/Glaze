from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from common_helpers.combine_participant_data_csvs import (
    DEFAULT_PARTICIPANT_IDS,
    DEFAULT_SOURCE_CSV_PATHS,
    write_dataset_csv,
)


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "block_id",
    "trial_index",
    "hazard_rate",
    "noise_sigma",
    "LLR",
    "choice",
    "correct_side",
    "reaction_time_ms",
    "belief_L",
    "subjective_h_snapshot",
)

NUMERIC_INPUT_COLUMNS: tuple[str, ...] = (
    "block_id",
    "trial_index",
    "hazard_rate",
    "noise_sigma",
    "LLR",
    "choice",
    "correct_side",
    "reaction_time_ms",
    "belief_L",
    "subjective_h_snapshot",
)

PREPROCESS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "choice",
    "reaction_time_ms",
    "LLR",
    "belief_L",
    "subjective_h_snapshot",
    "H",
    "prev_observed_belief_L",
)


def _build_subjective_h_grid(
    *,
    h_start: float = 0.001,
    h_end: float = 0.999,
    h_step: float = 0.01,
) -> np.ndarray:
    """Build the hazard-rate grid used for subjective-H fitting.

    Args:
        h_start: Inclusive lower bound for grid search.
        h_end: Inclusive upper bound for grid search.
        h_step: Positive grid increment.

    Returns:
        1D numpy array of candidate hazard rates.

    Raises:
        ValueError: If bounds or step are invalid.
    """
    if h_step <= 0:
        raise ValueError(f"h_step must be > 0. Got {h_step}.")
    if not (0.0 < h_start < 1.0 and 0.0 < h_end < 1.0):
        raise ValueError(
            f"h_start and h_end must lie in (0, 1). Got {h_start} and {h_end}."
        )
    if h_start > h_end:
        raise ValueError(f"h_start must be <= h_end. Got {h_start} > {h_end}.")

    # Add half-step slack to include the end value despite floating-point jitter.
    h_grid = np.round(np.arange(h_start, h_end + (h_step / 2.0), h_step), 12)
    if h_grid.size == 0:
        raise ValueError("Subjective-H grid is empty. Check h_start/h_end/h_step.")
    return h_grid


def compute_subjective_h_snapshots(
    df: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    trial_col: str = "trial_index",
    llr_col: str = "LLR",
    choice_col: str = "choice",
    beta: float = 1.0,
    h_start: float = 0.001,
    h_end: float = 0.999,
    h_step: float = 0.01,
    round_decimals: int = 4,
) -> pd.Series:
    """Recompute per-trial subjective hazard snapshots via incremental grid search.

    This mirrors the app-side behavior used in the Triangles task export:
    each row is fit using all trials observed so far within its participant/block
    sequence, and ties are broken toward lower H values.

    Args:
        df: Input trial-level dataframe.
        participant_col: Participant column name.
        block_col: Block column name.
        trial_col: Trial index column name (within block).
        llr_col: Per-trial log-likelihood ratio column.
        choice_col: Per-trial choice column (must be 0 or 1).
        beta: Fixed inverse temperature used in the logistic choice rule.
        h_start: Inclusive lower grid bound for H.
        h_end: Inclusive upper grid bound for H.
        h_step: Grid step for H candidates.
        round_decimals: Number of decimals to round returned snapshots.

    Returns:
        Series aligned to `df.index` containing recomputed subjective H snapshots.

    Raises:
        ValueError: If required columns are missing or values are invalid.
    """
    _validate_required_columns(
        df,
        (participant_col, block_col, trial_col, llr_col, choice_col),
        context="subjective-h recomputation",
    )
    if beta <= 0:
        raise ValueError(f"beta must be > 0. Got {beta}.")

    h_grid = _build_subjective_h_grid(h_start=h_start, h_end=h_end, h_step=h_step)
    stability_ratio = (1.0 - h_grid) / h_grid

    df_work = df.copy()
    df_work[participant_col] = df_work[participant_col].astype(str)
    df_work[block_col] = pd.to_numeric(df_work[block_col], errors="coerce")
    df_work[trial_col] = pd.to_numeric(df_work[trial_col], errors="coerce")
    df_work[llr_col] = pd.to_numeric(df_work[llr_col], errors="coerce")
    df_work[choice_col] = pd.to_numeric(df_work[choice_col], errors="coerce")

    finite_required = np.isfinite(
        df_work[[block_col, trial_col, llr_col, choice_col]].to_numpy(dtype=float)
    ).all(axis=1)
    if not bool(np.all(finite_required)):
        bad_n = int((~finite_required).sum())
        raise ValueError(
            f"Found {bad_n} non-finite rows in [{block_col}, {trial_col}, {llr_col}, {choice_col}] "
            "during subjective-h recomputation."
        )

    choice_values = df_work[choice_col].to_numpy(dtype=float)
    invalid_choice_mask = ~np.isin(choice_values, [0.0, 1.0])
    if bool(np.any(invalid_choice_mask)):
        bad_values = np.unique(choice_values[invalid_choice_mask])
        raise ValueError(
            f"Choice column '{choice_col}' must only contain 0/1. Found: {bad_values.tolist()}"
        )

    df_sorted = df_work.sort_values(
        [participant_col, block_col, trial_col], kind="mergesort"
    )
    recomputed = pd.Series(index=df.index, dtype=float)

    for _, group_df in df_sorted.groupby([participant_col, block_col], sort=False):
        llr_values = group_df[llr_col].to_numpy(dtype=float)
        choice_binary = group_df[choice_col].to_numpy(dtype=int)

        # Per-candidate running state (belief + cumulative NLL) as trials accrue.
        beliefs = np.zeros_like(h_grid, dtype=float)
        cumulative_nll = np.zeros_like(h_grid, dtype=float)
        fitted_h = np.empty(len(group_df), dtype=float)

        for trial_i, (llr_value, choice_value) in enumerate(
            zip(llr_values, choice_binary, strict=True)
        ):
            prev_beliefs = beliefs
            psi_term = (
                prev_beliefs
                + np.log(stability_ratio + np.exp(-prev_beliefs))
                - np.log(stability_ratio + np.exp(prev_beliefs))
            )
            beliefs = psi_term + llr_value

            prob_right = 1.0 / (1.0 + np.exp(-(beta * beliefs)))
            likelihood = prob_right if choice_value == 1 else (1.0 - prob_right)
            cumulative_nll -= np.log(np.maximum(1e-10, likelihood))

            # np.argmin returns the first occurrence, matching "<" tie-breaking.
            best_idx = int(np.argmin(cumulative_nll))
            fitted_h[trial_i] = h_grid[best_idx]

        recomputed.loc[group_df.index] = np.round(fitted_h, int(round_decimals))

    return recomputed.astype(float)


def audit_subjective_h_snapshots(
    df: pd.DataFrame,
    *,
    subjective_h_col: str = "subjective_h_snapshot",
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    trial_col: str = "trial_index",
    llr_col: str = "LLR",
    choice_col: str = "choice",
    beta: float = 1.0,
    h_start: float = 0.001,
    h_end: float = 0.999,
    h_step: float = 0.01,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Audit stored subjective-H snapshots against recomputed values.

    Args:
        df: Input trial-level dataframe.
        subjective_h_col: Existing subjective-H column to audit.
        participant_col: Participant column name.
        block_col: Block column name.
        trial_col: Trial index column name.
        llr_col: LLR column name.
        choice_col: Choice column name.
        beta: Fixed inverse temperature in the logistic choice rule.
        h_start: Inclusive lower H grid bound.
        h_end: Inclusive upper H grid bound.
        h_step: H grid step.
        tolerance: Absolute mismatch threshold.

    Returns:
        DataFrame containing only mismatch rows, sorted by participant/block/trial.
        Columns include existing/recomputed H values and absolute error.
    """
    _validate_required_columns(
        df,
        (subjective_h_col, participant_col, block_col, trial_col, llr_col, choice_col),
        context="subjective-h audit",
    )
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0. Got {tolerance}.")

    existing_h = pd.to_numeric(df[subjective_h_col], errors="coerce")
    recomputed_h = compute_subjective_h_snapshots(
        df,
        participant_col=participant_col,
        block_col=block_col,
        trial_col=trial_col,
        llr_col=llr_col,
        choice_col=choice_col,
        beta=beta,
        h_start=h_start,
        h_end=h_end,
        h_step=h_step,
    )

    abs_error = np.abs(existing_h - recomputed_h)
    finite_mask = np.isfinite(existing_h.to_numpy(dtype=float)) & np.isfinite(
        recomputed_h.to_numpy(dtype=float)
    )
    mismatch_mask = finite_mask & (abs_error.to_numpy(dtype=float) > float(tolerance))

    mismatches = df.loc[mismatch_mask].copy()
    mismatches["existing_subjective_h_snapshot"] = existing_h.loc[mismatch_mask].to_numpy(
        dtype=float
    )
    mismatches["recomputed_subjective_h_snapshot"] = recomputed_h.loc[mismatch_mask].to_numpy(
        dtype=float
    )
    mismatches["abs_error"] = abs_error.loc[mismatch_mask].to_numpy(dtype=float)

    return mismatches.sort_values(
        [participant_col, block_col, trial_col], kind="mergesort"
    ).reset_index(drop=True)


def repair_subjective_h_snapshot_column(
    df: pd.DataFrame,
    *,
    output_col: str = "subjective_h_snapshot",
    participant_col: str = "participant_id",
    block_col: str = "block_id",
    trial_col: str = "trial_index",
    llr_col: str = "LLR",
    choice_col: str = "choice",
    beta: float = 1.0,
    h_start: float = 0.001,
    h_end: float = 0.999,
    h_step: float = 0.01,
    round_decimals: int = 4,
) -> pd.DataFrame:
    """Return a copy of `df` with a recomputed subjective-H column.

    Args:
        df: Input trial-level dataframe.
        output_col: Output column name for recomputed subjective-H snapshots.
        participant_col: Participant column name.
        block_col: Block column name.
        trial_col: Trial index column name.
        llr_col: LLR column name.
        choice_col: Choice column name.
        beta: Fixed inverse temperature in the logistic choice rule.
        h_start: Inclusive lower H grid bound.
        h_end: Inclusive upper H grid bound.
        h_step: H grid step.
        round_decimals: Decimal precision for the output snapshots.

    Returns:
        DataFrame copy with `output_col` replaced by recomputed values.
    """
    repaired_df = df.copy()
    repaired_df[output_col] = compute_subjective_h_snapshots(
        repaired_df,
        participant_col=participant_col,
        block_col=block_col,
        trial_col=trial_col,
        llr_col=llr_col,
        choice_col=choice_col,
        beta=beta,
        h_start=h_start,
        h_end=h_end,
        h_step=h_step,
        round_decimals=round_decimals,
    )
    return repaired_df


def _resolve_csv_path(csv_path: str | Path) -> Path:
    """Resolve a CSV path from cwd or repository root.

    Args:
        csv_path: CSV path to resolve.

    Returns:
        Resolved path that exists on disk.

    Raises:
        FileNotFoundError: If the path cannot be resolved.
    """
    path = Path(csv_path)
    if path.exists():
        return path.resolve()

    if not path.is_absolute():
        repo_root_candidate = (Path(__file__).resolve().parents[2] / path).resolve()
        if repo_root_candidate.exists():
            return repo_root_candidate

    raise FileNotFoundError(f"Could not find CSV file: {csv_path}")


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Required column names.
        context: Short context for error messages.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values.

    Args:
        df: Input DataFrame.
        columns: Columns to coerce.

    Returns:
        DataFrame with coerced numeric columns (`errors='coerce'`).
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_non_finite_rows(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Drop rows with non-finite values in selected numeric columns.

    Args:
        df: Input DataFrame.
        numeric_columns: Columns that must be finite.
        context: Short context for warning messages.

    Returns:
        Filtered DataFrame.
    """
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
    """Validate reset policy values.

    Args:
        reset_on: Reset policy values.

    Returns:
        Validated reset policy.

    Raises:
        ValueError: If unsupported reset keys are provided.
    """
    allowed = {"participant", "block"}
    invalid = sorted(set(reset_on) - allowed)
    if invalid:
        raise ValueError(
            f"Invalid reset_on values: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return reset_on


def build_default_participants_csv(
    output_filename: str | Path = "participants.csv",
    source_csv_paths: Sequence[str | Path] = DEFAULT_SOURCE_CSV_PATHS,
    participant_ids: Sequence[str] | None = DEFAULT_PARTICIPANT_IDS,
) -> Path:
    """Build merged participant CSV from the three default source CSVs.

    Args:
        output_filename: Output merged CSV name/path.
        source_csv_paths: Source participant CSV paths.
        participant_ids: Participant IDs aligned with source CSV order.

    Returns:
        Absolute path to the merged CSV.
    """
    return write_dataset_csv(
        output_filename=output_filename,
        source_csv_paths=source_csv_paths,
        participant_ids=participant_ids,
    )


def load_participant_data(
    csv_path: str | Path = "data/participants.csv",
    participant_ids: list[str] | None = None,
    hazard_col: str = "subjective_h_snapshot",
    reset_on: tuple[str, ...] = ("participant", "block"),
) -> pd.DataFrame:
    """Load participant data and derive model-ready state columns.

    Args:
        csv_path: Path to the merged participant CSV.
        participant_ids: Optional participant filter.
        hazard_col: Column to use as per-trial hazard input `H`.
        reset_on: Reset conditions for sequence state.

    Returns:
        DataFrame sorted by participant/block/trial with derived columns:
            - `H`
            - `reset_state`
            - `prev_observed_belief_L`
            - `row_id`

    Raises:
        FileNotFoundError: If `csv_path` is missing.
        ValueError: If columns are missing or filters remove all rows.
    """
    reset_on = _validate_reset_on(reset_on)

    try:
        resolved_csv_path = _resolve_csv_path(csv_path)
    except FileNotFoundError:
        csv_path_obj = Path(csv_path)
        # If the merged CSV is missing, I rebuild it from the default three source CSVs.
        if csv_path_obj.name == "participants.csv":
            resolved_csv_path = build_default_participants_csv(
                output_filename="participants.csv"
            )
        else:
            raise
    df = pd.read_csv(resolved_csv_path)

    _validate_required_columns(df, REQUIRED_INPUT_COLUMNS, context="data loading")

    if hazard_col not in df.columns:
        raise ValueError(
            f"Invalid hazard_col '{hazard_col}'. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str)

    if participant_ids is not None:
        participant_ids_str = {str(pid) for pid in participant_ids}
        df = df[df["participant_id"].isin(participant_ids_str)].copy()
        if df.empty:
            raise ValueError(
                f"No rows remaining after participant filter {sorted(participant_ids_str)}"
            )

    df = _coerce_numeric_columns(df, NUMERIC_INPUT_COLUMNS)
    cols_to_check = list(NUMERIC_INPUT_COLUMNS)
    if hazard_col not in cols_to_check:
        cols_to_check.append(hazard_col)

    df = _drop_non_finite_rows(df, cols_to_check, context="data loading")
    if df.empty:
        raise ValueError("All rows were dropped due to non-finite values.")

    df = df.sort_values(["participant_id", "block_id", "trial_index"]).reset_index(
        drop=True
    )

    df["H"] = pd.to_numeric(df[hazard_col], errors="coerce")
    df = _drop_non_finite_rows(df, ["H"], context="hazard assignment")

    participant_changed = df["participant_id"] != df["participant_id"].shift(1)
    block_changed = (df["block_id"] != df["block_id"].shift(1)) | participant_changed

    reset_state = pd.Series(False, index=df.index)
    if "participant" in reset_on:
        reset_state |= participant_changed
    if "block" in reset_on:
        reset_state |= block_changed
    if not df.empty:
        reset_state.iloc[0] = True

    df["reset_state"] = reset_state.to_numpy(dtype=bool)

    df["prev_observed_belief_L"] = df["belief_L"].shift(1)
    df.loc[df["reset_state"], "prev_observed_belief_L"] = 0.0
    df["prev_observed_belief_L"] = (
        pd.to_numeric(df["prev_observed_belief_L"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    df["row_id"] = np.arange(len(df), dtype=int)
    return df.reset_index(drop=True)


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
    """Apply exclusions, split labels, and preprocessing summary tables.

    Args:
        df_loaded: DataFrame returned by `load_participant_data`.
        required_cols: Columns that must parse to finite numeric values.
        min_rt_ms: Inclusive minimum reaction-time cutoff in milliseconds.
        max_rt_ms: Inclusive maximum reaction-time cutoff in milliseconds.
        train_trial_max_index: Max trial index assigned to `TRAIN`.
        expected_blocks_per_participant: Expected number of blocks per participant.
        nominal_trials_per_block_before: Nominal pre-exclusion trials per block.

    Returns:
        Dictionary with:
            - `df_all`
            - `removed_rows_df`
            - `preprocessing_overview_table`
            - `participant_structure_table`
            - `blocks_per_participant`
            - `before_n`
            - `after_n`
            - `removed_n`
            - `safety_check_changed_data`

    Raises:
        ValueError: If required columns are missing or RT bounds are invalid.
    """
    if min_rt_ms > max_rt_ms:
        raise ValueError("min_rt_ms must be <= max_rt_ms")

    required_input_columns = ["participant_id", "block_id", "trial_index", *required_cols]
    _validate_required_columns(
        df_loaded,
        required_input_columns,
        context="preprocessing",
    )

    df_before_exclusions = df_loaded.copy()
    df_work = df_loaded.copy()
    df_work = _coerce_numeric_columns(df_work, required_cols)

    valid_mask = pd.Series(
        np.isfinite(df_work[list(required_cols)].to_numpy(dtype=float)).all(axis=1),
        index=df_work.index,
    )
    rt_mask = (
        (df_work["reaction_time_ms"] >= float(min_rt_ms))
        & (df_work["reaction_time_ms"] <= float(max_rt_ms))
    )
    keep_mask = valid_mask & rt_mask

    removed_rows_df = df_work.loc[~keep_mask].copy()
    removed_rows_df["removed_invalid_required"] = (
        ~valid_mask.loc[removed_rows_df.index]
    ).astype(int)
    removed_rows_df["removed_rt_out_of_range"] = (
        ~rt_mask.loc[removed_rows_df.index]
    ).astype(int)

    df_all = df_work.loc[keep_mask].copy()
    df_all = df_all.sort_values(["participant_id", "block_id", "trial_index"]).copy()
    df_all["split"] = np.where(
        df_all["trial_index"] <= int(train_trial_max_index), "TRAIN", "TEST"
    )

    blocks_per_participant = (
        df_all.groupby("participant_id")["block_id"].nunique().rename("n_blocks_after")
    )

    before_counts = (
        df_before_exclusions.groupby(["participant_id", "block_id"])
        .size()
        .rename("n_before")
    )
    after_counts = df_all.groupby(["participant_id", "block_id"]).size().rename("n_after")
    split_counts = (
        df_all.groupby(["participant_id", "block_id", "split"])
        .size()
        .unstack(fill_value=0)
    )

    preprocessing_overview_table = (
        pd.concat([before_counts, after_counts, split_counts], axis=1)
        .fillna(0)
        .reset_index()
    )
    preprocessing_overview_table["n_before"] = preprocessing_overview_table["n_before"].astype(
        int
    )
    preprocessing_overview_table["n_after"] = preprocessing_overview_table["n_after"].astype(
        int
    )
    preprocessing_overview_table["TRAIN"] = preprocessing_overview_table.get(
        "TRAIN", 0
    ).astype(int)
    preprocessing_overview_table["TEST"] = preprocessing_overview_table.get(
        "TEST", 0
    ).astype(int)
    preprocessing_overview_table["n_dropped"] = (
        preprocessing_overview_table["n_before"] - preprocessing_overview_table["n_after"]
    )
    preprocessing_overview_table = preprocessing_overview_table.rename(
        columns={"TRAIN": "n_train", "TEST": "n_test"}
    )
    preprocessing_overview_table = preprocessing_overview_table[
        [
            "participant_id",
            "block_id",
            "n_train",
            "n_test",
            "n_dropped",
            "n_before",
            "n_after",
        ]
    ]

    nominal_before_mask = (
        preprocessing_overview_table["n_before"] == int(nominal_trials_per_block_before)
    )
    # Create a new dataframe by adding a column that flags whether each block has the nominal number of trials before exclusions
    participant_structure_table = (
        preprocessing_overview_table.assign(
            block_is_nominal_before=nominal_before_mask.astype(int)
        )
        # Group the data by participant_id, keeping participant_id as a regular column (not index)
        .groupby("participant_id", as_index=False)
        # Aggregate the grouped data by computing:
        .agg(
            # Count the number of unique block_ids per participant (blocks before exclusions)
            n_blocks_before=("block_id", "nunique"),
            # Count the number of unique block_ids per participant (blocks after exclusions) - same operation
            n_blocks_after=("block_id", "nunique"),
            # Check if ALL blocks for this participant have nominal trial counts before exclusions (returns True/False)
            all_blocks_nominal_before=("block_is_nominal_before", lambda x: bool(np.all(x == 1))),
        )
    )
    participant_structure_table["n_blocks_after"] = participant_structure_table[
        "participant_id"
    ].map(blocks_per_participant).fillna(0).astype(int)
    participant_structure_table["has_expected_blocks_before"] = (
        participant_structure_table["n_blocks_before"] == int(expected_blocks_per_participant)
    )
    participant_structure_table["has_expected_blocks_after"] = (
        participant_structure_table["n_blocks_after"] == int(expected_blocks_per_participant)
    )

    before_n = int(len(df_work))
    after_n = int(len(df_all))
    removed_n = int(before_n - after_n)

    return {
        "df_all": df_all,
        "removed_rows_df": removed_rows_df,
        "preprocessing_overview_table": preprocessing_overview_table,
        "participant_structure_table": participant_structure_table,
        "blocks_per_participant": blocks_per_participant,
        "before_n": before_n,
        "after_n": after_n,
        "removed_n": removed_n,
        "safety_check_changed_data": bool(removed_n > 0),
    }


def combine_and_load_participant_data(
    *,
    output_filename: str | Path = "participants.csv",
    source_csv_paths: Sequence[str | Path] = DEFAULT_SOURCE_CSV_PATHS,
    source_participant_ids: Sequence[str] | None = DEFAULT_PARTICIPANT_IDS,
    participant_ids: list[str] | None = None,
    hazard_col: str = "subjective_h_snapshot",
    reset_on: tuple[str, ...] = ("participant", "block"),
) -> tuple[Path, pd.DataFrame]:
    """Combine source CSVs and return loaded participant data in one call.

    Args:
        output_filename: Merged CSV output filename/path.
        source_csv_paths: Source participant CSV paths.
        source_participant_ids: Participant IDs aligned with source paths.
        participant_ids: Optional participant subset after loading merged CSV.
        hazard_col: Hazard column name used for `H`.
        reset_on: Reset policy used by loader.

    Returns:
        Tuple of merged CSV path and loaded participant DataFrame.
    """
    merged_csv_path = build_default_participants_csv(
        output_filename=output_filename,
        source_csv_paths=source_csv_paths,
        participant_ids=source_participant_ids,
    )
    loaded_df = load_participant_data(
        csv_path=merged_csv_path,
        participant_ids=participant_ids,
        hazard_col=hazard_col,
        reset_on=reset_on,
    )
    return merged_csv_path, loaded_df
