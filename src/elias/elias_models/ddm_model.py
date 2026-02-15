from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EPSILON
from .continuous_models import _prepare_model_input

try:
    from evan.glaze import psi_function
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from evan.glaze import psi_function


def _sigmoid(x: float) -> float:
    """Compute a numerically stable sigmoid."""
    clipped = np.clip(x, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _simulate_ddm_single_sample(
    *,
    v: float,
    a: float,
    z: float,
    dt_ms: float,
    max_duration_ms: float,
    diffusion_sigma: float,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    """Simulate one DDM trajectory until boundary crossing or timeout."""
    dt_sec = dt_ms / 1000.0
    sqrt_dt_sec = np.sqrt(dt_sec)

    x = (2.0 * z - 1.0) * a
    time_current_ms = 0.0

    while time_current_ms < max_duration_ms:
        x += v * dt_sec + diffusion_sigma * sqrt_dt_sec * float(rng.standard_normal())
        time_current_ms += dt_ms

        if x >= a:
            return 1, time_current_ms, x
        if x <= -a:
            return -1, time_current_ms, x

    return 0, float(max_duration_ms), x


def run_model_c_ddm(
    df: pd.DataFrame,
    *,
    n_samples_per_trial: int = 200,
    dt_ms: float = 5.0,
    max_duration_ms: float = 1500.0,
    boundary_a: float = 1.0,
    non_decision_time_ms: float = 200.0,
    llr_to_drift_scale: float = 1.0,
    start_k: float = 0.1,
    diffusion_sigma: float = 1.0,
    include_rt_samples: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run Model C (DDM driven by DNM trial signals)."""
    if n_samples_per_trial <= 0:
        raise ValueError("n_samples_per_trial must be > 0")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be > 0")
    if boundary_a <= 0:
        raise ValueError("boundary_a must be > 0")
    if diffusion_sigma <= 0:
        raise ValueError("diffusion_sigma must be > 0")

    model_df = _prepare_model_input(df)
    rng = np.random.default_rng(random_seed)

    results: list[dict[str, object]] = []

    for row in model_df.itertuples(index=False):
        psi_t = float(psi_function(float(row.prev_observed_belief_L), float(row.H)))
        z_t = _sigmoid(float(start_k) * psi_t)
        z_t = float(np.clip(z_t, EPSILON, 1.0 - EPSILON))
        v_t = float(llr_to_drift_scale) * float(row.LLR)

        decisions = np.zeros(n_samples_per_trial, dtype=int)
        rts_ms = np.zeros(n_samples_per_trial, dtype=float)

        for i in range(n_samples_per_trial):
            decision, rt_ms, _ = _simulate_ddm_single_sample(
                v=v_t,
                a=float(boundary_a),
                z=z_t,
                dt_ms=float(dt_ms),
                max_duration_ms=float(max_duration_ms),
                diffusion_sigma=float(diffusion_sigma),
                rng=rng,
            )
            decisions[i] = int(decision)
            rts_ms[i] = float(rt_ms + non_decision_time_ms)

        prob_pos = float(np.mean(decisions == 1))
        prob_neg = float(np.mean(decisions == -1))
        timeout_rate = float(np.mean(decisions == 0))

        if prob_pos == 0.0 and prob_neg == 0.0:
            predicted_decision = 0
        elif prob_pos >= prob_neg:
            predicted_decision = 1
        else:
            predicted_decision = -1

        q10, q50, q90 = np.quantile(rts_ms, [0.10, 0.50, 0.90])

        row_dict: dict[str, object] = {
            "row_id": int(row.row_id),
            "participant_id": str(row.participant_id),
            "block_id": int(row.block_id),
            "trial_index": int(row.trial_index),
            "choice": int(row.choice),
            "reaction_time_ms": float(row.reaction_time_ms),
            "belief_L": float(row.belief_L),
            "LLR": float(row.LLR),
            "H": float(row.H),
            "prev_observed_belief_L": float(row.prev_observed_belief_L),
            "used_threshold": np.nan,
            "predicted_decision": int(predicted_decision),
            "predicted_rt_ms": float(np.median(rts_ms)),
            "predicted_belief": float(psi_t),
            "pred_decision_prob_pos": prob_pos,
            "pred_decision_prob_neg": prob_neg,
            "pred_timeout_rate": timeout_rate,
            "pred_rt_mean_ms": float(np.mean(rts_ms)),
            "pred_rt_std_ms": float(np.std(rts_ms)),
            "pred_rt_q10_ms": float(q10),
            "pred_rt_q50_ms": float(q50),
            "pred_rt_q90_ms": float(q90),
            "model_name": "ddm_dnm",
            "param_n_samples_per_trial": int(n_samples_per_trial),
            "param_dt_ms": float(dt_ms),
            "param_max_duration_ms": float(max_duration_ms),
            "param_boundary_a": float(boundary_a),
            "param_non_decision_time_ms": float(non_decision_time_ms),
            "param_llr_to_drift_scale": float(llr_to_drift_scale),
            "param_start_k": float(start_k),
            "param_diffusion_sigma": float(diffusion_sigma),
            "param_random_seed": int(random_seed),
        }

        if include_rt_samples:
            row_dict["pred_rt_samples_ms"] = rts_ms.copy()

        results.append(row_dict)

    return pd.DataFrame(results)
