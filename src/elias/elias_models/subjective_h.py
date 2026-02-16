# Glaze-style subjective hazard inference and normative belief reconstruction.
# Main functions: fit_blockwise_subjective_h_choice_only,
# attach_subjective_h_from_train, build_normative_belief_columns.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_validation import _normalize_choice_values_to_pm1, _validate_required_columns


@dataclass(frozen=True)
class SubjectiveHGrid:
    """Grid definition used for blockwise subjective-H search."""

    h_start: float = 0.001
    h_end: float = 0.999
    h_step: float = 0.01

    def values(self) -> np.ndarray:
        if self.h_step <= 0:
            raise ValueError("h_step must be > 0")
        if not (0.0 < self.h_start < 1.0 and 0.0 < self.h_end < 1.0):
            raise ValueError("h_start and h_end must be in (0, 1)")
        if self.h_start > self.h_end:
            raise ValueError("h_start must be <= h_end")
        return np.round(
            np.arange(self.h_start, self.h_end + (self.h_step / 2.0), self.h_step),
            12,
        )


def glaze_psi(l_prev: float, h: float) -> float:
    """Glaze Eq. 2 prior transform (numerically stable)."""
    h_clamped = float(np.clip(h, 1e-9, 1.0 - 1e-9))
    if abs(float(l_prev)) > 100.0:
        return float(np.sign(float(l_prev)) * np.log((1.0 - h_clamped) / h_clamped))

    stability_ratio = (1.0 - h_clamped) / h_clamped
    term_pos = stability_ratio + np.exp(-float(l_prev))
    term_neg = stability_ratio + np.exp(float(l_prev))
    return float(float(l_prev) + np.log(term_pos) - np.log(term_neg))


def _belief_trajectory_for_h(
    llr_values: np.ndarray,
    *,
    h: float,
) -> np.ndarray:
    """Compute normative beliefs from LLR sequence under one H candidate."""
    beliefs = np.zeros(len(llr_values), dtype=float)
    l_prev = 0.0
    for idx, llr in enumerate(llr_values):
        psi_t = glaze_psi(l_prev, h)
        l_curr = psi_t + float(llr)  # Glaze Eq. 1
        beliefs[idx] = l_curr
        l_prev = l_curr
    return beliefs


def _choice_nll_from_beliefs(
    beliefs: np.ndarray,
    choices_pm1: np.ndarray,
    *,
    beta: float,
) -> float:
    """Negative log-likelihood of choices given belief trajectory."""
    logits = np.clip(beta * np.asarray(beliefs, dtype=float), -60.0, 60.0)
    prob_right = 1.0 / (1.0 + np.exp(-logits))
    # choices_pm1 uses {-1,+1}; convert to binary right-choice indicator.
    y_right = (np.asarray(choices_pm1, dtype=int) == 1).astype(float)
    likelihood = np.where(y_right > 0.5, prob_right, 1.0 - prob_right)
    return float(-np.sum(np.log(np.maximum(likelihood, 1e-12))))


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
    """Fit one subjective hazard value per participant-block from TRAIN choices.

    The objective is Glaze-consistent choice-only NLL with fixed beta.
    """
    _validate_required_columns(
        df,
        (participant_col, block_col, trial_col, split_col, llr_col, choice_col),
        context="subjective-H fitting",
    )
    if beta <= 0.0:
        raise ValueError("beta must be > 0")

    grid = SubjectiveHGrid() if h_grid is None else h_grid
    h_values = grid.values()

    work = df.copy()
    work[participant_col] = work[participant_col].astype(str)
    work[block_col] = pd.to_numeric(work[block_col], errors="coerce")
    work[trial_col] = pd.to_numeric(work[trial_col], errors="coerce")
    work[llr_col] = pd.to_numeric(work[llr_col], errors="coerce")
    work[choice_col] = pd.to_numeric(work[choice_col], errors="coerce")
    work[split_col] = work[split_col].astype(str)

    train_df = work[work[split_col] == str(train_label)].copy()
    if train_df.empty:
        raise ValueError("No TRAIN rows available for subjective-H fitting.")

    finite_mask = np.isfinite(
        train_df[[block_col, trial_col, llr_col, choice_col]].to_numpy(dtype=float)
    ).all(axis=1)
    if not bool(np.all(finite_mask)):
        n_bad = int((~finite_mask).sum())
        raise ValueError(
            f"Found {n_bad} non-finite rows while fitting subjective H on TRAIN data."
        )

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

        best_h = None
        best_nll = None
        for h_candidate in h_values:
            beliefs = _belief_trajectory_for_h(llr_values, h=float(h_candidate))
            nll = _choice_nll_from_beliefs(beliefs, choice_values, beta=float(beta))
            # np.argmin-style tie handling: first value wins, so lower H wins by construction.
            if best_nll is None or nll < best_nll:
                best_nll = float(nll)
                best_h = float(h_candidate)

        if best_h is None or best_nll is None:
            raise RuntimeError(
                f"Failed to fit subjective H for participant={participant_id}, block={block_id}."
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
    """Attach fitted TRAIN blockwise subjective H to TRAIN+TEST rows."""
    _validate_required_columns(
        df,
        (participant_col, block_col),
        context="attach subjective H",
    )
    _validate_required_columns(
        subjective_h_table,
        (participant_col, block_col, fitted_h_col),
        context="subjective-H table",
    )

    out = df.copy()
    out[participant_col] = out[participant_col].astype(str)
    out[block_col] = pd.to_numeric(out[block_col], errors="coerce")
    # Prevent merge suffix collisions when callers pass frames that already
    # carry a legacy/placeholder `H` column.
    if output_h_col in out.columns:
        out = out.drop(columns=[output_h_col])

    h_table = subjective_h_table[[participant_col, block_col, fitted_h_col]].copy()
    h_table[participant_col] = h_table[participant_col].astype(str)
    h_table[block_col] = pd.to_numeric(h_table[block_col], errors="coerce")

    out = out.merge(
        h_table.rename(columns={fitted_h_col: output_h_col}),
        on=[participant_col, block_col],
        how="left",
    )

    missing_mask = ~np.isfinite(pd.to_numeric(out[output_h_col], errors="coerce"))
    if bool(np.any(missing_mask)):
        # Keep pipeline robust if one block has no TRAIN rows after exclusions.
        fallback_h = float(
            pd.to_numeric(h_table[fitted_h_col], errors="coerce").dropna().mean()
        )
        if not np.isfinite(fallback_h):
            fallback_h = 0.1
        out.loc[missing_mask, output_h_col] = float(fallback_h)
    out[output_h_col] = pd.to_numeric(out[output_h_col], errors="coerce").astype(float)
    out["subjective_h_fallback_used"] = missing_mask.astype(int)
    return out


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
    """Build recursive normative prior/posterior belief columns.

    This is the active state input used by all candidate models.
    """
    _validate_required_columns(
        df,
        (participant_col, block_col, trial_col, llr_col, hazard_col),
        context="normative-belief reconstruction",
    )

    out = df.copy()
    out[participant_col] = out[participant_col].astype(str)
    out[block_col] = pd.to_numeric(out[block_col], errors="coerce")
    out[trial_col] = pd.to_numeric(out[trial_col], errors="coerce")
    out[llr_col] = pd.to_numeric(out[llr_col], errors="coerce")
    out[hazard_col] = pd.to_numeric(out[hazard_col], errors="coerce")

    finite_mask = np.isfinite(
        out[[block_col, trial_col, llr_col, hazard_col]].to_numpy(dtype=float)
    ).all(axis=1)
    if not bool(np.all(finite_mask)):
        n_bad = int((~finite_mask).sum())
        raise ValueError(
            f"Found {n_bad} non-finite rows while building normative belief columns."
        )

    # Keep original row order while running recursion on a sorted view.
    out = out.reset_index(names="_orig_index")
    out = out.sort_values([participant_col, block_col, trial_col], kind="mergesort").reset_index(
        drop=True
    )
    prev_values = np.zeros(len(out), dtype=float)
    psi_values = np.zeros(len(out), dtype=float)
    curr_values = np.zeros(len(out), dtype=float)

    for _, chunk in out.groupby([participant_col, block_col], sort=False):
        pos = chunk.index.to_numpy(dtype=int)
        llr_vals = chunk[llr_col].to_numpy(dtype=float)
        h_vals = chunk[hazard_col].to_numpy(dtype=float)

        l_prev = 0.0
        for local_i, (llr_t, h_t) in enumerate(zip(llr_vals, h_vals, strict=True)):
            global_i = int(pos[local_i])
            prev_values[global_i] = float(l_prev)
            psi_t = glaze_psi(l_prev, float(h_t))
            l_curr = float(psi_t + float(llr_t))
            psi_values[global_i] = float(psi_t)
            curr_values[global_i] = float(l_curr)
            l_prev = l_curr

    out[output_prev_col] = prev_values
    out[output_psi_col] = psi_values
    out[output_curr_col] = curr_values

    out = out.sort_values("_orig_index", kind="mergesort").drop(columns=["_orig_index"])
    return out.reset_index(drop=True)
