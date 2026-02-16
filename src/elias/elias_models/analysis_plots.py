#
# Post-run plotting utilities for Step 3/4/5 analysis in notebooks.
# Main functions: plot_step3_recovery_diagnostics, plot_step4_testscore_diagnostics,
# plot_step4_winner_diagnostics, plot_step5_ppc_hazard_diagnostics,
# plot_step5_latent_block_diagnostics, plot_step5_latent_model_comparison.

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_or_show_figure(
    fig: plt.Figure,
    path: Path | None,
    show: bool,
    dpi: int = 150,
) -> None:
    """Save and/or display a matplotlib figure.

    Args:
        fig: Figure to save or display.
        path: Optional output path for PNG export.
        show: Whether to display the figure inline/interactively.
        dpi: Figure save DPI when `path` is provided.
    """
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")

    if bool(show):
        plt.show()
    else:
        plt.close(fig)


def _resolve_plot_path(
    output_dir: str | Path | None,
    filename: str,
    save: bool,
) -> Path | None:
    """Resolve an output plot path only when saving is enabled.

    Args:
        output_dir: Output directory or None.
        filename: Plot filename.
        save: Whether saving is enabled.

    Returns:
        Resolved output path or None.
    """
    if not bool(save) or output_dir is None:
        return None
    return Path(output_dir) / str(filename)


def _is_non_empty(df: pd.DataFrame | None) -> bool:
    """Check whether a DataFrame is present and non-empty.

    Args:
        df: Candidate DataFrame.

    Returns:
        True when DataFrame is not None and has at least one row.
    """
    return df is not None and not df.empty


def _plot_matrix_heatmap(
    df: pd.DataFrame,
    *,
    title: str,
    output_path: Path | None,
    show: bool,
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render a numeric matrix DataFrame as an annotated heatmap.

    Args:
        df: Numeric matrix table.
        title: Plot title.
        output_path: Optional PNG output path.
        show: Whether to display the figure.
        dpi: Save DPI.
        vmin: Optional lower color bound.
        vmax: Optional upper color bound.
    """
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        warnings.warn(f"Skipping empty heatmap: {title}", stacklevel=2)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    x_labels = [str(col) for col in df.columns.tolist()]
    if values.shape[0] == len(x_labels):
        y_labels = x_labels
    else:
        y_labels = [f"row_{idx + 1}" for idx in range(values.shape[0])]

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            cell = values[i, j]
            text = "nan" if not np.isfinite(cell) else f"{cell:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=9)

    save_or_show_figure(fig, output_path, show=show, dpi=dpi)


def plot_step3_recovery_diagnostics(
    step3_tables: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path | None = None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> dict[str, str]:
    """Plot Step 3 recovery diagnostics from persisted tables.

    Args:
        step3_tables: Mapping of Step 3 table names to DataFrames.
        output_dir: Directory where PNG files should be written.
        save: Whether to save plot files.
        show: Whether to display plots.
        dpi: Save DPI.

    Returns:
        Mapping of plot labels to written PNG paths.
    """
    saved_paths: dict[str, str] = {}

    joint_rates = step3_tables.get("model_recovery_joint_rates", pd.DataFrame())
    if _is_non_empty(joint_rates):
        path = _resolve_plot_path(output_dir, "step3_joint_recovery_rates.png", save=save)
        _plot_matrix_heatmap(
            joint_rates,
            title="Step 3 Joint Recovery Rates",
            output_path=path,
            show=show,
            dpi=dpi,
            vmin=0.0,
            vmax=1.0,
        )
        if path is not None:
            saved_paths["step3_joint_recovery_rates"] = str(path)
    else:
        warnings.warn("Step 3 joint recovery rates table is empty.", stacklevel=2)

    bic_rates = step3_tables.get("model_recovery_bic_rates", pd.DataFrame())
    if _is_non_empty(bic_rates):
        path = _resolve_plot_path(output_dir, "step3_bic_recovery_rates.png", save=save)
        _plot_matrix_heatmap(
            bic_rates,
            title="Step 3 BIC Recovery Rates",
            output_path=path,
            show=show,
            dpi=dpi,
            vmin=0.0,
            vmax=1.0,
        )
        if path is not None:
            saved_paths["step3_bic_recovery_rates"] = str(path)
    else:
        warnings.warn("Step 3 BIC recovery rates table is empty.", stacklevel=2)

    param_summary = step3_tables.get("parameter_recovery_summary", pd.DataFrame())
    if _is_non_empty(param_summary) and {"model_name", "param_name", "mae"}.issubset(
        param_summary.columns
    ):
        mae_df = param_summary.copy()
        mae_df["mae"] = pd.to_numeric(mae_df["mae"], errors="coerce")
        mae_df["label"] = (
            mae_df["model_name"].astype(str) + ":" + mae_df["param_name"].astype(str)
        )
        mae_df = mae_df.sort_values("mae", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(mae_df))))
        ax.barh(mae_df["label"], mae_df["mae"], color="#33658A")
        ax.set_xlabel("MAE")
        ax.set_ylabel("Model:Parameter")
        ax.set_title("Step 3 Parameter Recovery MAE")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        path = _resolve_plot_path(output_dir, "step3_parameter_recovery_mae.png", save=save)
        save_or_show_figure(fig, path, show=show, dpi=dpi)
        if path is not None:
            saved_paths["step3_parameter_recovery_mae"] = str(path)
    else:
        warnings.warn(
            "Step 3 parameter recovery summary is empty or missing required columns.",
            stacklevel=2,
        )

    fit_results = step3_tables.get("fit_results", pd.DataFrame())
    if _is_non_empty(fit_results) and {"candidate_model_name", "joint_score"}.issubset(
        fit_results.columns
    ):
        plot_df = fit_results.copy()
        plot_df["joint_score"] = pd.to_numeric(plot_df["joint_score"], errors="coerce")
        groups = sorted(plot_df["candidate_model_name"].astype(str).unique().tolist())
        data = [
            plot_df.loc[
                plot_df["candidate_model_name"].astype(str) == model_name,
                "joint_score",
            ]
            .dropna()
            .to_numpy(dtype=float)
            for model_name in groups
        ]

        if any(len(arr) > 0 for arr in data):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(data, labels=groups, showfliers=True)
            ax.set_ylabel("Joint Score (lower is better)")
            ax.set_title("Step 3 Fit Joint-Score Distribution by Candidate Model")
            ax.grid(axis="y", alpha=0.3)

            path = _resolve_plot_path(output_dir, "step3_joint_score_distribution.png", save=save)
            save_or_show_figure(fig, path, show=show, dpi=dpi)
            if path is not None:
                saved_paths["step3_joint_score_distribution"] = str(path)
    else:
        warnings.warn(
            "Step 3 fit results are empty or missing required columns for score distribution.",
            stacklevel=2,
        )

    return saved_paths


def plot_step4_testscore_diagnostics(
    step4_tables: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path | None = None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> dict[str, str]:
    """Plot Step 4 TEST-score diagnostics from persisted tables.

    Args:
        step4_tables: Mapping of Step 4 table names to DataFrames.
        output_dir: Directory where PNG files should be written.
        save: Whether to save plot files.
        show: Whether to display plots.
        dpi: Save DPI.

    Returns:
        Mapping of plot labels to written PNG paths.
    """
    saved_paths: dict[str, str] = {}

    test_scores = step4_tables.get("participant_model_scores_test", pd.DataFrame())
    needed_cols = {"participant_id", "candidate_model_name", "joint_score"}
    if not (_is_non_empty(test_scores) and needed_cols.issubset(test_scores.columns)):
        warnings.warn(
            "Step 4 TEST score table is empty or missing required columns.",
            stacklevel=2,
        )
        return saved_paths

    scores_df = test_scores.copy()
    scores_df["joint_score"] = pd.to_numeric(scores_df["joint_score"], errors="coerce")
    participants = sorted(scores_df["participant_id"].astype(str).unique().tolist())

    fig, ax = plt.subplots(figsize=(8, 4))
    for participant_id in participants:
        chunk = scores_df[scores_df["participant_id"].astype(str) == participant_id].copy()
        chunk = chunk.sort_values("candidate_model_name")
        ax.plot(
            chunk["candidate_model_name"].astype(str),
            chunk["joint_score"].to_numpy(dtype=float),
            marker="o",
            label=str(participant_id),
        )
    ax.set_xlabel("Candidate Model")
    ax.set_ylabel("TEST Joint Score (lower is better)")
    ax.set_title("Step 4 TEST Joint Scores by Participant")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Participant", fontsize=8)
    path = _resolve_plot_path(output_dir, "step4_test_joint_scores_by_participant.png", save=save)
    save_or_show_figure(fig, path, show=show, dpi=dpi)
    if path is not None:
        saved_paths["step4_test_joint_scores_by_participant"] = str(path)

    aggregate = (
        scores_df.groupby("candidate_model_name", as_index=False)["joint_score"]
        .median()
        .sort_values("joint_score", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(aggregate["candidate_model_name"], aggregate["joint_score"], color="#2A9D8F")
    ax.set_xlabel("Candidate Model")
    ax.set_ylabel("Median TEST Joint Score")
    ax.set_title("Step 4 Median TEST Joint Score by Model")
    ax.grid(axis="y", alpha=0.3)
    path = _resolve_plot_path(output_dir, "step4_test_joint_score_medians.png", save=save)
    save_or_show_figure(fig, path, show=show, dpi=dpi)
    if path is not None:
        saved_paths["step4_test_joint_score_medians"] = str(path)

    return saved_paths


def plot_step4_winner_diagnostics(
    step4_tables: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path | None = None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> dict[str, str]:
    """Plot Step 4 winner and consistency diagnostics.

    Args:
        step4_tables: Mapping of Step 4 table names to DataFrames.
        output_dir: Directory where PNG files should be written.
        save: Whether to save plot files.
        show: Whether to display plots.
        dpi: Save DPI.

    Returns:
        Mapping of plot labels to written PNG paths.
    """
    saved_paths: dict[str, str] = {}

    winner_table = step4_tables.get("participant_winner_table", pd.DataFrame())
    if _is_non_empty(winner_table) and {"participant_id", "blockwise_consistency"}.issubset(
        winner_table.columns
    ):
        plot_df = winner_table.copy()
        plot_df["blockwise_consistency"] = pd.to_numeric(
            plot_df["blockwise_consistency"],
            errors="coerce",
        )
        plot_df = plot_df.sort_values("participant_id").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(plot_df["participant_id"].astype(str), plot_df["blockwise_consistency"], color="#8E6C8A")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Participant")
        ax.set_ylabel("Blockwise Consistency")
        ax.set_title("Step 4 Winner Blockwise Consistency")
        ax.grid(axis="y", alpha=0.3)
        path = _resolve_plot_path(output_dir, "step4_blockwise_consistency.png", save=save)
        save_or_show_figure(fig, path, show=show, dpi=dpi)
        if path is not None:
            saved_paths["step4_blockwise_consistency"] = str(path)
    else:
        warnings.warn(
            "Step 4 participant winner table is empty or missing consistency columns.",
            stacklevel=2,
        )

    winner_counts = step4_tables.get("group_winner_counts", pd.DataFrame())
    if _is_non_empty(winner_counts) and {"model_name", "participant_win_count"}.issubset(
        winner_counts.columns
    ):
        count_df = winner_counts.copy()
        count_df["participant_win_count"] = pd.to_numeric(
            count_df["participant_win_count"],
            errors="coerce",
        )
        count_df = count_df.sort_values("participant_win_count", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(count_df["model_name"].astype(str), count_df["participant_win_count"], color="#264653")
        ax.set_xlabel("Model")
        ax.set_ylabel("Participant Wins")
        ax.set_title("Step 4 Group Winner Vote Counts")
        ax.grid(axis="y", alpha=0.3)
        path = _resolve_plot_path(output_dir, "step4_group_winner_counts.png", save=save)
        save_or_show_figure(fig, path, show=show, dpi=dpi)
        if path is not None:
            saved_paths["step4_group_winner_counts"] = str(path)
    else:
        warnings.warn(
            "Step 4 group winner counts table is empty or missing required columns.",
            stacklevel=2,
        )

    return saved_paths


def plot_step5_ppc_hazard_diagnostics(
    step5_tables: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path | None = None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> dict[str, str]:
    """Plot Step 5 posterior-predictive and hazard-signature diagnostics.

    Args:
        step5_tables: Mapping of Step 5 table names to DataFrames.
        output_dir: Directory where PNG files should be written.
        save: Whether to save plot files.
        show: Whether to display plots.
        dpi: Save DPI.

    Returns:
        Mapping of plot labels to written PNG paths.
    """
    saved_paths: dict[str, str] = {}

    ppc_block = step5_tables.get("step5_posterior_predictive_block", pd.DataFrame())
    if _is_non_empty(ppc_block) and {"participant_id", "joint_nll_per_trial"}.issubset(ppc_block.columns):
        ppc_df = ppc_block.copy()
        ppc_df["joint_nll_per_trial"] = pd.to_numeric(
            ppc_df["joint_nll_per_trial"],
            errors="coerce",
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        for participant_id, chunk in ppc_df.groupby("participant_id", sort=True):
            x_vals = np.full(len(chunk), str(participant_id), dtype=object)
            ax.scatter(x_vals, chunk["joint_nll_per_trial"], alpha=0.7, label=str(participant_id))
        ax.set_xlabel("Participant")
        ax.set_ylabel("Joint NLL per Trial")
        ax.set_title("Step 5 PPC Joint NLL per Trial by Participant")
        ax.grid(axis="y", alpha=0.3)
        path = _resolve_plot_path(output_dir, "step5_ppc_joint_nll_per_trial.png", save=save)
        save_or_show_figure(fig, path, show=show, dpi=dpi)
        if path is not None:
            saved_paths["step5_ppc_joint_nll_per_trial"] = str(path)
    else:
        warnings.warn(
            "Step 5 posterior predictive block table is empty or missing required columns.",
            stacklevel=2,
        )

    hazard_block = step5_tables.get("step5_hazard_signature_block", pd.DataFrame())
    required_hazard_cols = {
        "participant_id",
        "h_vs_shrinkage_spearman",
        "change_point_accuracy",
        "stable_accuracy",
    }
    if _is_non_empty(hazard_block) and required_hazard_cols.issubset(hazard_block.columns):
        hazard_df = hazard_block.copy()
        hazard_df["h_vs_shrinkage_spearman"] = pd.to_numeric(
            hazard_df["h_vs_shrinkage_spearman"],
            errors="coerce",
        )
        mean_spearman = (
            hazard_df.groupby("participant_id", as_index=False)["h_vs_shrinkage_spearman"]
            .mean()
            .sort_values("participant_id")
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            mean_spearman["participant_id"].astype(str),
            mean_spearman["h_vs_shrinkage_spearman"],
            color="#457B9D",
        )
        ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Participant")
        ax.set_ylabel("Mean Spearman(H, shrinkage)")
        ax.set_title("Step 5 Hazard Signature by Participant")
        ax.grid(axis="y", alpha=0.3)
        path = _resolve_plot_path(output_dir, "step5_hazard_shrinkage_spearman.png", save=save)
        save_or_show_figure(fig, path, show=show, dpi=dpi)
        if path is not None:
            saved_paths["step5_hazard_shrinkage_spearman"] = str(path)

        accuracy_df = hazard_df.copy()
        accuracy_df["change_point_accuracy"] = pd.to_numeric(
            accuracy_df["change_point_accuracy"],
            errors="coerce",
        )
        accuracy_df["stable_accuracy"] = pd.to_numeric(
            accuracy_df["stable_accuracy"],
            errors="coerce",
        )
        mean_accuracy = (
            accuracy_df.groupby("participant_id", as_index=False)[
                ["change_point_accuracy", "stable_accuracy"]
            ]
            .mean()
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        x = np.arange(len(mean_accuracy))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            x - width / 2,
            mean_accuracy["change_point_accuracy"],
            width=width,
            label="Change-point",
            color="#E76F51",
        )
        ax.bar(
            x + width / 2,
            mean_accuracy["stable_accuracy"],
            width=width,
            label="Stable",
            color="#2A9D8F",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(mean_accuracy["participant_id"].astype(str))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Participant")
        ax.set_ylabel("Mean Accuracy")
        ax.set_title("Step 5 Change-Point vs Stable Accuracy")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        path = _resolve_plot_path(output_dir, "step5_changepoint_vs_stable_accuracy.png", save=save)
        save_or_show_figure(fig, path, show=show, dpi=dpi)
        if path is not None:
            saved_paths["step5_changepoint_vs_stable_accuracy"] = str(path)
    else:
        warnings.warn(
            "Step 5 hazard signature block table is empty or missing required columns.",
            stacklevel=2,
        )

    return saved_paths


def plot_step5_latent_block_diagnostics(
    step5_tables: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path | None = None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> dict[str, str]:
    """Plot Step 5 latent block-level diagnostics.

    Args:
        step5_tables: Mapping of Step 5 table names to DataFrames.
        output_dir: Directory where PNG files should be written.
        save: Whether to save plot files.
        show: Whether to display plots.
        dpi: Save DPI.

    Returns:
        Mapping of plot labels to written PNG paths.
    """
    saved_paths: dict[str, str] = {}

    latent_block = step5_tables.get("step5_latent_quantities_block", pd.DataFrame())
    required_cols = {
        "participant_id",
        "choice_accuracy_excluding_timeout",
        "timeout_rate",
        "mae_rt_ms",
        "mae_belief",
    }
    if not (_is_non_empty(latent_block) and required_cols.issubset(latent_block.columns)):
        warnings.warn(
            "Step 5 latent quantities block table is empty or missing required columns.",
            stacklevel=2,
        )
        return saved_paths

    latent_df = latent_block.copy()
    for col in (
        "choice_accuracy_excluding_timeout",
        "timeout_rate",
        "mae_rt_ms",
        "mae_belief",
    ):
        latent_df[col] = pd.to_numeric(latent_df[col], errors="coerce")

    summary = (
        latent_df.groupby("participant_id", as_index=False)[
            ["choice_accuracy_excluding_timeout", "timeout_rate", "mae_rt_ms", "mae_belief"]
        ]
        .mean()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x - width / 2,
        summary["choice_accuracy_excluding_timeout"],
        width=width,
        label="Choice Accuracy",
        color="#264653",
    )
    ax.bar(
        x + width / 2,
        summary["timeout_rate"],
        width=width,
        label="Timeout Rate",
        color="#E9C46A",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["participant_id"].astype(str))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Rate")
    ax.set_title("Step 5 Latent Accuracy and Timeout Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = _resolve_plot_path(output_dir, "step5_latent_accuracy_timeout.png", save=save)
    save_or_show_figure(fig, path, show=show, dpi=dpi)
    if path is not None:
        saved_paths["step5_latent_accuracy_timeout"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(summary["mae_rt_ms"], summary["mae_belief"], color="#8E6C8A", s=70)
    for row in summary.itertuples(index=False):
        ax.text(float(row.mae_rt_ms), float(row.mae_belief), f" {row.participant_id}", fontsize=9)
    ax.set_xlabel("MAE RT (ms)")
    ax.set_ylabel("MAE Belief")
    ax.set_title("Step 5 Latent Error Tradeoff by Participant")
    ax.grid(alpha=0.3)
    path = _resolve_plot_path(output_dir, "step5_latent_error_tradeoff.png", save=save)
    save_or_show_figure(fig, path, show=show, dpi=dpi)
    if path is not None:
        saved_paths["step5_latent_error_tradeoff"] = str(path)

    return saved_paths


def plot_step5_latent_model_comparison(
    latent_trial_table: pd.DataFrame,
    *,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> dict[str, str]:
    """Render Evan-style model-comparison plot for Step 5 latent trials.

    Args:
        latent_trial_table: Trial-level latent table from Step 5.
        run_id: Optional run identifier for plot annotation.
        output_dir: Directory where PNG file should be written.
        save: Whether to save plot file.
        show: Whether to display plot.
        dpi: Save DPI.

    Returns:
        Mapping with latent comparison plot path when saved.
    """
    required_cols = {
        "choice",
        "predicted_decision",
        "predicted_rt_ms",
        "reaction_time_ms",
        "predicted_belief",
        "belief_L",
        "block_id",
        "LLR",
    }
    if not _is_non_empty(latent_trial_table):
        warnings.warn("Step 5 latent trial table is empty.", stacklevel=2)
        return {}
    missing_cols = sorted(required_cols - set(latent_trial_table.columns))
    if missing_cols:
        warnings.warn(
            f"Cannot run Evan latent comparison plot; missing columns: {missing_cols}",
            stacklevel=2,
        )
        return {}

    from evan.glaze import plot_model_comparison

    save_path = _resolve_plot_path(
        output_dir,
        "step5_latent_model_comparison.png",
        save=save,
    )
    params = {}
    if run_id is not None:
        params["run_id"] = str(run_id)

    plot_model_comparison(
        latent_trial_table.copy(),
        params=params or None,
        show=show,
        save_path=save_path,
        dpi=dpi,
        return_fig=False,
    )

    if save_path is None:
        return {}
    return {"step5_latent_model_comparison": str(save_path)}
