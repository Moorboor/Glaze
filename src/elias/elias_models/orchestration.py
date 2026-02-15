from __future__ import annotations

import pandas as pd

from .continuous_models import run_model_a_threshold, run_model_b_asymptote
from .data_validation import _validate_required_columns
from .ddm_model import run_model_c_ddm


def run_all_models_for_participant(
    df: pd.DataFrame,
    participant_id: str,
    *,
    random_seed: int = 42,
    n_samples_per_trial: int = 200,
    include_rt_samples: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run models A, B, and C for one participant."""
    _validate_required_columns(df, ["participant_id"], context="orchestration")

    participant_id_str = str(participant_id)
    subset = df[df["participant_id"].astype(str) == participant_id_str].copy()
    if subset.empty:
        available = sorted(df["participant_id"].astype(str).unique().tolist())
        raise ValueError(
            f"No rows for participant_id='{participant_id_str}'. Available: {available}"
        )

    out_a = run_model_a_threshold(subset, random_seed=random_seed)
    out_b = run_model_b_asymptote(subset, random_seed=random_seed)
    out_c = run_model_c_ddm(
        subset,
        n_samples_per_trial=n_samples_per_trial,
        include_rt_samples=include_rt_samples,
        random_seed=random_seed,
    )

    return {
        "cont_threshold": out_a,
        "cont_asymptote": out_b,
        "ddm_dnm": out_c,
    }
