"""Participant- and group-level winner rules for Step 4 evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Validate that a table contains the expected columns.

    Args:
        df: DataFrame to validate.
        required_columns: Required column names.
        context: Context label for error messages.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric_scores(
    df: pd.DataFrame,
    *,
    score_columns: tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    """Coerce selected score columns to finite numeric values.

    Args:
        df: Input score table.
        score_columns: Score columns to coerce.
        context: Context label for error messages.

    Returns:
        DataFrame with numeric score columns.

    Raises:
        ValueError: If non-finite values are found after coercion.
    """
    out = df.copy()
    for col in score_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    finite_mask = np.isfinite(out[list(score_columns)].to_numpy(dtype=float)).all(axis=1)
    if not bool(np.all(finite_mask)):
        bad_rows = int((~finite_mask).sum())
        raise ValueError(
            f"{context} contains {bad_rows} rows with non-finite score values."
        )
    return out


def _find_tied_min_models(
    score_chunk: pd.DataFrame,
    *,
    model_column: str,
    score_column: str,
    tie_tolerance: float,
) -> tuple[list[str], float]:
    """Find models tied at the minimum score within tolerance.

    Args:
        score_chunk: Participant- or block-level score rows.
        model_column: Column holding model names.
        score_column: Primary score column.
        tie_tolerance: Inclusive tolerance around the minimum score.

    Returns:
        Tuple of tied model names and minimum score.
    """
    min_score = float(score_chunk[score_column].min())
    tied = score_chunk[
        score_chunk[score_column] <= (min_score + float(tie_tolerance))
    ][model_column].astype(str)
    tied_models = sorted(set(tied.tolist()))
    return tied_models, min_score


def _build_empty_outputs() -> dict[str, pd.DataFrame]:
    """Build empty output tables with stable schemas."""
    participant_winner_table = pd.DataFrame(
        columns=[
            "participant_id",
            "winner_model_name",
            "winner_joint_score",
            "winner_choice_only_score",
            "winner_rt_only_cond_score",
            "winner_bic_score",
            "primary_score_column",
            "tie_on_primary",
            "tied_models_primary",
            "participant_tie_break_applied",
            "participant_tie_break_rule",
            "n_blocks_test",
            "n_blocks_matching_winner",
            "blockwise_consistency",
        ]
    )
    group_winner_counts = pd.DataFrame(
        columns=[
            "model_name",
            "participant_win_count",
            "test_joint_score_sum",
        ]
    )
    group_winner_summary = pd.DataFrame(
        columns=[
            "group_winner_model_name",
            "n_participants",
            "n_candidate_models",
            "max_vote_count",
            "vote_tie",
            "vote_tied_models",
            "tie_break_applied",
            "tie_break_rule",
            "tie_break_joint_sums_for_vote_ties",
            "primary_score_column",
            "tie_tolerance",
        ]
    )
    return {
        "participant_winner_table": participant_winner_table,
        "group_winner_counts": group_winner_counts,
        "group_winner_summary": group_winner_summary,
    }


def apply_step4_winner_rules(
    test_scores: pd.DataFrame,
    block_test_scores: pd.DataFrame,
    *,
    primary_score_column: str = "joint_score",
    tie_tolerance: float = 1e-9,
) -> dict[str, pd.DataFrame]:
    """Apply Step 4 participant and group winner rules.

    Args:
        test_scores: Participant-level TEST score rows with one row per
            participant-model pair.
        block_test_scores: Block-level TEST score rows with one row per
            participant-block-model tuple.
        primary_score_column: Primary score column used for winner selection.
        tie_tolerance: Inclusive tolerance for tie detection around minimum score.

    Returns:
        Dictionary with:
            - ``participant_winner_table``
            - ``group_winner_counts``
            - ``group_winner_summary``
    """
    if float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be >= 0.")
    if test_scores.empty:
        return _build_empty_outputs()

    _validate_required_columns(
        test_scores,
        (
            "participant_id",
            "candidate_model_name",
            primary_score_column,
            "joint_score",
            "choice_only_score",
            "rt_only_cond_score",
            "bic_score",
        ),
        context="Step 4 participant TEST scores",
    )
    _validate_required_columns(
        block_test_scores,
        (
            "participant_id",
            "block_id",
            "candidate_model_name",
            primary_score_column,
        ),
        context="Step 4 blockwise TEST scores",
    )

    score_df = _coerce_numeric_scores(
        test_scores,
        score_columns=(primary_score_column, "joint_score", "choice_only_score", "rt_only_cond_score", "bic_score"),
        context="Step 4 participant TEST scores",
    )
    block_df = _coerce_numeric_scores(
        block_test_scores,
        score_columns=(primary_score_column,),
        context="Step 4 blockwise TEST scores",
    )
    score_df = score_df.copy()
    block_df = block_df.copy()
    score_df["participant_id"] = score_df["participant_id"].astype(str)
    block_df["participant_id"] = block_df["participant_id"].astype(str)
    score_df["candidate_model_name"] = score_df["candidate_model_name"].astype(str)
    block_df["candidate_model_name"] = block_df["candidate_model_name"].astype(str)

    participant_rows: list[dict[str, object]] = []
    for participant_id, participant_chunk in score_df.groupby("participant_id", sort=True):
        tied_models, min_primary = _find_tied_min_models(
            participant_chunk,
            model_column="candidate_model_name",
            score_column=primary_score_column,
            tie_tolerance=float(tie_tolerance),
        )
        tie_on_primary = len(tied_models) > 1

        # Deterministic participant-level tie handling is lexical only.
        winner_model = tied_models[0]
        participant_tie_break_applied = bool(tie_on_primary)
        participant_tie_break_rule = "lexical_model_name" if tie_on_primary else "none"

        winner_row = participant_chunk[
            participant_chunk["candidate_model_name"] == winner_model
        ].sort_values(
            by=[primary_score_column, "candidate_model_name"],
            ascending=[True, True],
        ).iloc[0]

        participant_block_chunk = block_df[
            block_df["participant_id"] == str(participant_id)
        ].copy()
        n_blocks_test = int(participant_block_chunk["block_id"].nunique())
        n_blocks_matching = 0
        if n_blocks_test > 0:
            for _, block_chunk in participant_block_chunk.groupby("block_id", sort=True):
                block_tied_models, _ = _find_tied_min_models(
                    block_chunk,
                    model_column="candidate_model_name",
                    score_column=primary_score_column,
                    tie_tolerance=float(tie_tolerance),
                )
                if winner_model in set(block_tied_models):
                    n_blocks_matching += 1

        blockwise_consistency = (
            float(n_blocks_matching) / float(n_blocks_test)
            if n_blocks_test > 0
            else np.nan
        )

        participant_rows.append(
            {
                "participant_id": str(participant_id),
                "winner_model_name": str(winner_model),
                "winner_joint_score": float(winner_row["joint_score"]),
                "winner_choice_only_score": float(winner_row["choice_only_score"]),
                "winner_rt_only_cond_score": float(winner_row["rt_only_cond_score"]),
                "winner_bic_score": float(winner_row["bic_score"]),
                "primary_score_column": str(primary_score_column),
                "tie_on_primary": bool(tie_on_primary),
                "tied_models_primary": tied_models,
                "participant_tie_break_applied": participant_tie_break_applied,
                "participant_tie_break_rule": participant_tie_break_rule,
                "n_blocks_test": n_blocks_test,
                "n_blocks_matching_winner": int(n_blocks_matching),
                "blockwise_consistency": blockwise_consistency,
                "winner_primary_score_value": float(min_primary),
            }
        )

    participant_winner_table = pd.DataFrame(participant_rows).sort_values(
        ["participant_id"],
        ascending=[True],
    ).reset_index(drop=True)

    model_names = sorted(score_df["candidate_model_name"].astype(str).unique().tolist())
    vote_counts = (
        participant_winner_table["winner_model_name"]
        .value_counts(dropna=False)
        .reindex(model_names, fill_value=0)
    )
    joint_sums = (
        score_df.groupby("candidate_model_name", as_index=True)["joint_score"]
        .sum()
        .reindex(model_names, fill_value=np.nan)
    )

    group_winner_counts = pd.DataFrame(
        {
            "model_name": model_names,
            "participant_win_count": vote_counts.to_numpy(dtype=int),
            "test_joint_score_sum": joint_sums.to_numpy(dtype=float),
        }
    ).sort_values(
        ["participant_win_count", "test_joint_score_sum", "model_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    max_vote = int(group_winner_counts["participant_win_count"].max())
    vote_tied_rows = group_winner_counts[
        group_winner_counts["participant_win_count"] == max_vote
    ].copy()
    vote_tied_models = sorted(vote_tied_rows["model_name"].astype(str).tolist())
    vote_tie = len(vote_tied_models) > 1

    if vote_tie:
        tie_break_joint_sums = {
            model_name: float(
                vote_tied_rows.loc[vote_tied_rows["model_name"] == model_name, "test_joint_score_sum"].iloc[0]
            )
            for model_name in vote_tied_models
        }
        min_joint_sum = min(tie_break_joint_sums.values())
        sum_tied_models = sorted(
            [
                model_name
                for model_name, total_joint in tie_break_joint_sums.items()
                if float(total_joint) <= (float(min_joint_sum) + float(tie_tolerance))
            ]
        )
        group_winner_model = sum_tied_models[0]
        tie_break_applied = True
        tie_break_rule = "sum_test_joint_score_then_lexical"
    else:
        tie_break_joint_sums = {
            vote_tied_models[0]: float(
                vote_tied_rows.loc[
                    vote_tied_rows["model_name"] == vote_tied_models[0], "test_joint_score_sum"
                ].iloc[0]
            )
        }
        group_winner_model = vote_tied_models[0]
        tie_break_applied = False
        tie_break_rule = "none"

    group_winner_summary = pd.DataFrame(
        [
            {
                "group_winner_model_name": str(group_winner_model),
                "n_participants": int(participant_winner_table["participant_id"].nunique()),
                "n_candidate_models": int(len(model_names)),
                "max_vote_count": int(max_vote),
                "vote_tie": bool(vote_tie),
                "vote_tied_models": vote_tied_models,
                "tie_break_applied": bool(tie_break_applied),
                "tie_break_rule": str(tie_break_rule),
                "tie_break_joint_sums_for_vote_ties": tie_break_joint_sums,
                "primary_score_column": str(primary_score_column),
                "tie_tolerance": float(tie_tolerance),
            }
        ]
    )

    return {
        "participant_winner_table": participant_winner_table,
        "group_winner_counts": group_winner_counts,
        "group_winner_summary": group_winner_summary,
    }
