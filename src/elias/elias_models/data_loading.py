from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .constants import PREPROCESS_REQUIRED_COLUMNS
from .data_validation import _normalize_choice_column_to_pm1

try:
    from common_helpers.preprocessing import (
        load_participant_data as shared_load_participant_data,
        preprocess_loaded_participant_data as shared_preprocess_loaded_participant_data,
    )
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from common_helpers.preprocessing import (
        load_participant_data as shared_load_participant_data,
        preprocess_loaded_participant_data as shared_preprocess_loaded_participant_data,
    )


def load_participant_data(
    csv_path: str | Path = "data/participants.csv",
    participant_ids: list[str] | None = None,
    hazard_col: str = "subjective_h_snapshot",
    reset_on: tuple[str, ...] = ("participant", "block"),
) -> pd.DataFrame:
    """Load participant data and normalize observed choices to `-1/+1`."""
    loaded_df = shared_load_participant_data(
        csv_path=csv_path,
        participant_ids=participant_ids,
        hazard_col=hazard_col,
        reset_on=reset_on,
    )
    return _normalize_choice_column_to_pm1(loaded_df)


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
    """Apply shared preprocessing for exclusions, splits, and overview tables."""
    return shared_preprocess_loaded_participant_data(
        df_loaded,
        required_cols=required_cols,
        min_rt_ms=min_rt_ms,
        max_rt_ms=max_rt_ms,
        train_trial_max_index=train_trial_max_index,
        expected_blocks_per_participant=expected_blocks_per_participant,
        nominal_trials_per_block_before=nominal_trials_per_block_before,
    )
