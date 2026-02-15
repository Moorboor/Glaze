"""Step 5 final conclusion and report export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HAZARD_CAVEAT_SENTENCE = (
    "Caveat: Hazard input was fixed to `subjective_h_snapshot` and treated as an externally "
    "provided past-only signal during fit and evaluation; this does not establish endogenous "
    "hazard inference by the models."
)


def _safe_first_bool(df: pd.DataFrame, column: str, default: bool) -> bool:
    if df.empty or column not in df.columns:
        return bool(default)
    return bool(df[column].iloc[0])


def _safe_first_str(df: pd.DataFrame, column: str, default: str) -> str:
    if df.empty or column not in df.columns:
        return str(default)
    return str(df[column].iloc[0])


def build_recovery_aware_conclusion(
    *,
    step3_soft_gate: dict[str, Any],
    step4_group_winner_summary: pd.DataFrame,
    step5_posterior_predictive_block: pd.DataFrame,
    step5_hazard_signature_block: pd.DataFrame,
    step5_latent_quantities_block: pd.DataFrame,
) -> pd.DataFrame:
    """Build one machine-readable Step 5 conclusion row."""
    soft_gate_status = str(step3_soft_gate.get("overall_status", "unknown"))
    step4_vote_tie = _safe_first_bool(step4_group_winner_summary, "vote_tie", False)
    step4_group_winner = _safe_first_str(
        step4_group_winner_summary,
        "group_winner_model_name",
        "unknown",
    )

    if soft_gate_status == "weak":
        conclusion_level = "weak"
    elif soft_gate_status == "caution" or step4_vote_tie:
        conclusion_level = "caution"
    else:
        conclusion_level = "supported"

    ppc_mean_joint_nll_per_trial = (
        float(step5_posterior_predictive_block["joint_nll_per_trial"].mean())
        if not step5_posterior_predictive_block.empty
        and "joint_nll_per_trial" in step5_posterior_predictive_block.columns
        else float(np.nan)
    )
    hazard_mean_h_shrinkage_spearman = (
        float(step5_hazard_signature_block["h_vs_shrinkage_spearman"].mean())
        if not step5_hazard_signature_block.empty
        and "h_vs_shrinkage_spearman" in step5_hazard_signature_block.columns
        else float(np.nan)
    )
    latent_mean_choice_accuracy = (
        float(step5_latent_quantities_block["choice_accuracy_excluding_timeout"].mean())
        if not step5_latent_quantities_block.empty
        and "choice_accuracy_excluding_timeout" in step5_latent_quantities_block.columns
        else float(np.nan)
    )
    latent_mean_timeout_rate = (
        float(step5_latent_quantities_block["timeout_rate"].mean())
        if not step5_latent_quantities_block.empty
        and "timeout_rate" in step5_latent_quantities_block.columns
        else float(np.nan)
    )

    conclusion_text = (
        "Recovery-aware conclusion: "
        f"level={conclusion_level}; "
        f"step3_soft_gate={soft_gate_status}; "
        f"step4_group_winner={step4_group_winner}; "
        f"step4_vote_tie={step4_vote_tie}; "
        f"ppc_mean_joint_nll_per_trial={ppc_mean_joint_nll_per_trial:.6f}; "
        f"hazard_mean_h_shrinkage_spearman={hazard_mean_h_shrinkage_spearman:.6f}; "
        f"latent_mean_choice_accuracy={latent_mean_choice_accuracy:.6f}; "
        f"latent_mean_timeout_rate={latent_mean_timeout_rate:.6f}."
    )

    return pd.DataFrame(
        [
            {
                "conclusion_level": str(conclusion_level),
                "step3_soft_gate_status": str(soft_gate_status),
                "step4_group_winner_model_name": str(step4_group_winner),
                "step4_vote_tie": bool(step4_vote_tie),
                "ppc_mean_joint_nll_per_trial": ppc_mean_joint_nll_per_trial,
                "hazard_mean_h_shrinkage_spearman": hazard_mean_h_shrinkage_spearman,
                "latent_mean_choice_accuracy": latent_mean_choice_accuracy,
                "latent_mean_timeout_rate": latent_mean_timeout_rate,
                "conclusion_text": str(conclusion_text),
                "hazard_input_caveat_sentence": HAZARD_CAVEAT_SENTENCE,
            }
        ]
    )


def write_step5_report_markdown(
    path: str | Path,
    *,
    run_id: str,
    step3_soft_gate: dict[str, Any],
    step4_group_winner_summary: pd.DataFrame,
    conclusion_table: pd.DataFrame,
    step5_posterior_predictive_block: pd.DataFrame,
    step5_hazard_signature_block: pd.DataFrame,
    step5_latent_quantities_block: pd.DataFrame,
) -> Path:
    """Write a concise Markdown report for Step 5 outputs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    step3_status = str(step3_soft_gate.get("overall_status", "unknown"))
    step4_group_winner = _safe_first_str(
        step4_group_winner_summary,
        "group_winner_model_name",
        "unknown",
    )
    step4_vote_tie = _safe_first_bool(step4_group_winner_summary, "vote_tie", False)

    if conclusion_table.empty:
        raise ValueError("conclusion_table must contain one row.")
    conclusion = conclusion_table.iloc[0]

    lines = [
        f"# Step 5 Report ({run_id})",
        "",
        "## Recovery-Aware Conclusion",
        str(conclusion.get("conclusion_text", "")),
        "",
        "## Cross-Step Context",
        f"- Step 3 soft-gate status: `{step3_status}`",
        f"- Step 4 group winner: `{step4_group_winner}`",
        f"- Step 4 vote tie: `{step4_vote_tie}`",
        "",
        "## Step 5 Summary Metrics",
        f"- PPC mean joint NLL per trial: `{conclusion.get('ppc_mean_joint_nll_per_trial', np.nan)}`",
        "- Hazard-signature mean Spearman(H, shrinkage): "
        f"`{conclusion.get('hazard_mean_h_shrinkage_spearman', np.nan)}`",
        "- Latent mean choice accuracy (excluding timeout): "
        f"`{conclusion.get('latent_mean_choice_accuracy', np.nan)}`",
        f"- Latent mean timeout rate: `{conclusion.get('latent_mean_timeout_rate', np.nan)}`",
        "",
        "## Step 5 Artifact Counts",
        f"- Posterior predictive blocks: `{len(step5_posterior_predictive_block)}`",
        f"- Hazard-signature blocks: `{len(step5_hazard_signature_block)}`",
        f"- Latent-summary blocks: `{len(step5_latent_quantities_block)}`",
        "",
        "## Hazard Input Caveat",
        HAZARD_CAVEAT_SENTENCE,
        "",
    ]

    target.write_text("\n".join(lines), encoding="utf-8")
    return target
