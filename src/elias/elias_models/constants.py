from __future__ import annotations

import sys
from pathlib import Path

try:
    from common_helpers.preprocessing import (
        NUMERIC_INPUT_COLUMNS as SHARED_NUMERIC_INPUT_COLUMNS,
        PREPROCESS_REQUIRED_COLUMNS as SHARED_PREPROCESS_REQUIRED_COLUMNS,
        REQUIRED_INPUT_COLUMNS as SHARED_REQUIRED_INPUT_COLUMNS,
    )
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from common_helpers.preprocessing import (
        NUMERIC_INPUT_COLUMNS as SHARED_NUMERIC_INPUT_COLUMNS,
        PREPROCESS_REQUIRED_COLUMNS as SHARED_PREPROCESS_REQUIRED_COLUMNS,
        REQUIRED_INPUT_COLUMNS as SHARED_REQUIRED_INPUT_COLUMNS,
    )


EPSILON = 1e-9

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = SHARED_REQUIRED_INPUT_COLUMNS
NUMERIC_INPUT_COLUMNS: tuple[str, ...] = SHARED_NUMERIC_INPUT_COLUMNS

MODEL_READY_COLUMNS: tuple[str, ...] = (
    "row_id",
    "participant_id",
    "block_id",
    "trial_index",
    "LLR",
    "H",
    "choice",
    "reaction_time_ms",
    "belief_L",
    "prev_observed_belief_L",
)

PREPROCESS_REQUIRED_COLUMNS: tuple[str, ...] = SHARED_PREPROCESS_REQUIRED_COLUMNS

SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "cont_threshold",
    "cont_asymptote",
    "ddm_dnm",
)
