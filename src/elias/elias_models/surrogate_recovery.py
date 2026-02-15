from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import SUPPORTED_MODEL_NAMES
from .continuous_models import _prepare_model_input
from .likelihood_scoring import (
    _simulate_continuous_trials_for_likelihood,
    _simulate_ddm_trials_for_likelihood,
)
from .optimizer_runner import fit_model_parameters
from .parameter_space import (
    get_parameter_spec,
    theta_to_eta,
    theta_to_named_params,
    theta_to_scoring_model_params,
)
from .results_store import (
    build_basic_manifest,
    load_json,
    load_table_csv,
    prepare_run_dir,
    resolve_elias_data_root,
    save_json,
    save_table_csv,
)


STEP3_PIPELINE_NAME = "surrogate_recovery"
_STEP3_TABLE_NAMES: tuple[str, ...] = (
    "pseudo_true_table",
    "surrogate_metadata",
    "surrogate_trials",
    "fit_results",
    "model_recovery_joint_counts",
    "model_recovery_joint_rates",
    "model_recovery_bic_counts",
    "model_recovery_bic_rates",
    "parameter_recovery_rows",
    "parameter_recovery_summary",
    "soft_gate_summary",
)

_DEFAULT_SURROGATE_CONFIG: dict[str, object] = {
    "n_draws_per_trial": 128,
    "fixed_model_params": {
        "dt_ms": 1.0,
        "max_duration_ms": 5000.0,
    },
}

_DEFAULT_STEP3_PIPELINE_CONFIG: dict[str, object] = {
    "candidate_models": SUPPORTED_MODEL_NAMES,
    "n_surrogates_per_model": 20,
    "surrogate_n_draws_per_trial": 128,
    "fit_n_starts": 4,
    "fit_n_iterations": 8,
    "fit_n_sims_per_trial": 150,
    "dt_ms": 1.0,
    "max_duration_ms": 5000.0,
    "random_seed": 0,
    "soft_gate_joint_diag_min": 0.60,
    "soft_gate_param_median_r_min": 0.30,
}


def _validate_candidate_models(candidate_models: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(model_name) for model_name in candidate_models)
    if len(normalized) == 0:
        raise ValueError("candidate_models must not be empty.")
    invalid = [model_name for model_name in normalized if model_name not in SUPPORTED_MODEL_NAMES]
    if invalid:
        raise ValueError(
            f"Unsupported candidate models: {invalid}. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )
    return normalized


def _build_surrogate_config(
    surrogate_config: dict[str, object] | None,
) -> dict[str, object]:
    merged = deepcopy(_DEFAULT_SURROGATE_CONFIG)
    if surrogate_config is not None:
        for key, value in surrogate_config.items():
            if key == "fixed_model_params" and isinstance(value, dict):
                merged_fixed = dict(merged.get("fixed_model_params", {}))
                merged_fixed.update(value)
                merged["fixed_model_params"] = merged_fixed
            else:
                merged[key] = value

    if int(merged["n_draws_per_trial"]) <= 0:
        raise ValueError("n_draws_per_trial must be > 0.")

    fixed_params = merged.get("fixed_model_params", {})
    if not isinstance(fixed_params, dict):
        raise ValueError("fixed_model_params must be a dictionary.")
    merged["fixed_model_params"] = dict(fixed_params)
    return merged


def build_step3_pipeline_config(
    *,
    candidate_models: tuple[str, ...] = SUPPORTED_MODEL_NAMES,
    n_surrogates_per_model: int = 20,
    surrogate_n_draws_per_trial: int = 128,
    fit_n_starts: int = 4,
    fit_n_iterations: int = 8,
    fit_n_sims_per_trial: int = 150,
    dt_ms: float = 1.0,
    max_duration_ms: float = 5000.0,
    random_seed: int = 0,
    soft_gate_joint_diag_min: float = 0.60,
    soft_gate_param_median_r_min: float = 0.30,
) -> dict[str, object]:
    """Build one canonical Step 3 pipeline config without notebook-specific split keys."""
    cfg = {
        "candidate_models": _validate_candidate_models(tuple(candidate_models)),
        "n_surrogates_per_model": int(n_surrogates_per_model),
        "surrogate_n_draws_per_trial": int(surrogate_n_draws_per_trial),
        "fit_n_starts": int(fit_n_starts),
        "fit_n_iterations": int(fit_n_iterations),
        "fit_n_sims_per_trial": int(fit_n_sims_per_trial),
        "dt_ms": float(dt_ms),
        "max_duration_ms": float(max_duration_ms),
        "random_seed": int(random_seed),
        "soft_gate_joint_diag_min": float(soft_gate_joint_diag_min),
        "soft_gate_param_median_r_min": float(soft_gate_param_median_r_min),
    }

    if cfg["n_surrogates_per_model"] <= 0:
        raise ValueError("n_surrogates_per_model must be > 0.")
    if cfg["surrogate_n_draws_per_trial"] <= 0:
        raise ValueError("surrogate_n_draws_per_trial must be > 0.")
    if cfg["fit_n_starts"] <= 0:
        raise ValueError("fit_n_starts must be > 0.")
    if cfg["fit_n_iterations"] < 0:
        raise ValueError("fit_n_iterations must be >= 0.")
    if cfg["fit_n_sims_per_trial"] <= 0:
        raise ValueError("fit_n_sims_per_trial must be > 0.")
    if cfg["dt_ms"] <= 0.0:
        raise ValueError("dt_ms must be > 0.")
    if cfg["max_duration_ms"] <= 0.0:
        raise ValueError("max_duration_ms must be > 0.")
    if not (0.0 <= cfg["soft_gate_joint_diag_min"] <= 1.0):
        raise ValueError("soft_gate_joint_diag_min must be in [0, 1].")
    if not (-1.0 <= cfg["soft_gate_param_median_r_min"] <= 1.0):
        raise ValueError("soft_gate_param_median_r_min must be in [-1, 1].")

    return cfg


def _normalize_step3_pipeline_config(config: dict[str, object]) -> dict[str, object]:
    merged = deepcopy(_DEFAULT_STEP3_PIPELINE_CONFIG)
    merged.update(dict(config))
    return build_step3_pipeline_config(
        candidate_models=tuple(merged["candidate_models"]),
        n_surrogates_per_model=int(merged["n_surrogates_per_model"]),
        surrogate_n_draws_per_trial=int(merged["surrogate_n_draws_per_trial"]),
        fit_n_starts=int(merged["fit_n_starts"]),
        fit_n_iterations=int(merged["fit_n_iterations"]),
        fit_n_sims_per_trial=int(merged["fit_n_sims_per_trial"]),
        dt_ms=float(merged["dt_ms"]),
        max_duration_ms=float(merged["max_duration_ms"]),
        random_seed=int(merged["random_seed"]),
        soft_gate_joint_diag_min=float(merged["soft_gate_joint_diag_min"]),
        soft_gate_param_median_r_min=float(merged["soft_gate_param_median_r_min"]),
    )


def _fit_config_from_step3_config(config: dict[str, object]) -> dict[str, object]:
    return {
        "n_starts": int(config["fit_n_starts"]),
        "n_iterations": int(config["fit_n_iterations"]),
        "n_sims_per_trial": int(config["fit_n_sims_per_trial"]),
        "fixed_model_params": {
            "dt_ms": float(config["dt_ms"]),
            "max_duration_ms": float(config["max_duration_ms"]),
        },
    }


def _surrogate_config_from_step3_config(config: dict[str, object]) -> dict[str, object]:
    return {
        "n_draws_per_trial": int(config["surrogate_n_draws_per_trial"]),
        "fixed_model_params": {
            "dt_ms": float(config["dt_ms"]),
            "max_duration_ms": float(config["max_duration_ms"]),
        },
    }


def sample_pseudo_true_thetas(
    model_name: str,
    n_sets: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample pseudo-true theta vectors uniformly within Step 2D bounds."""
    if int(n_sets) <= 0:
        raise ValueError("n_sets must be > 0.")

    spec = get_parameter_spec(model_name)
    lower = np.asarray([float(item["lower"]) for item in spec], dtype=float)
    upper = np.asarray([float(item["upper"]) for item in spec], dtype=float)
    param_names = [str(item["name"]) for item in spec]

    rng = np.random.default_rng(int(random_seed))
    theta_matrix = rng.uniform(
        low=lower,
        high=upper,
        size=(int(n_sets), len(spec)),
    )

    rows: list[dict[str, object]] = []
    for set_idx, theta_vector in enumerate(theta_matrix):
        named = theta_to_named_params(model_name, theta_vector)
        row = {
            "model_name": str(model_name),
            "sample_index": int(set_idx),
            "theta_vector": np.asarray(theta_vector, dtype=float),
            "eta_vector": theta_to_eta(model_name, theta_vector),
        }
        for param_name in param_names:
            row[f"theta_{param_name}"] = float(named[param_name])
        rows.append(row)

    return pd.DataFrame(rows)


def _sample_observed_trials_from_simulations(
    *,
    decisions_by_trial: list[np.ndarray],
    rts_by_trial: list[np.ndarray],
    llr_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    sampled_choices = np.zeros(len(decisions_by_trial), dtype=int)
    sampled_rts = np.zeros(len(decisions_by_trial), dtype=float)
    timeout_fallback_count = 0

    for trial_idx, (decision_samples, rt_samples) in enumerate(
        zip(decisions_by_trial, rts_by_trial, strict=True)
    ):
        decision_samples = np.asarray(decision_samples, dtype=int)
        rt_samples = np.asarray(rt_samples, dtype=float)

        valid_idx = np.where(np.isin(decision_samples, (-1, 1)))[0]
        if valid_idx.size > 0:
            chosen_idx = int(rng.choice(valid_idx))
            sampled_choices[trial_idx] = int(decision_samples[chosen_idx])
            sampled_rts[trial_idx] = float(rt_samples[chosen_idx])
            continue

        timeout_fallback_count += 1
        sampled_choices[trial_idx] = 1 if float(llr_values[trial_idx]) >= 0.0 else -1
        finite_rts = rt_samples[np.isfinite(rt_samples)]
        sampled_rts[trial_idx] = (
            float(np.median(finite_rts)) if finite_rts.size > 0 else 5000.0
        )

    return sampled_choices, sampled_rts, timeout_fallback_count


def simulate_surrogate_dataset(
    df_template: pd.DataFrame,
    model_name: str,
    theta: np.ndarray,
    random_seed: int,
    surrogate_id: str,
    surrogate_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Simulate one surrogate dataset from a generating model and pseudo-true theta."""
    model_name_str = str(model_name)
    if model_name_str not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name '{model_name_str}'. "
            f"Supported models: {list(SUPPORTED_MODEL_NAMES)}"
        )

    config = _build_surrogate_config(surrogate_config)
    model_df = _prepare_model_input(df_template)
    model_params = theta_to_scoring_model_params(model_name_str, np.asarray(theta, dtype=float))
    fixed_params = dict(config["fixed_model_params"])
    fixed_params.update(model_params)

    n_draws_per_trial = int(config["n_draws_per_trial"])
    seed_value = int(random_seed)

    if model_name_str in ("cont_threshold", "cont_asymptote"):
        decisions_by_trial, rts_by_trial = _simulate_continuous_trials_for_likelihood(
            model_df=model_df,
            stop_on_sat=(model_name_str == "cont_asymptote"),
            n_sims_per_trial=n_draws_per_trial,
            max_duration_ms=float(fixed_params.get("max_duration_ms", 5000.0)),
            dt_ms=float(fixed_params.get("dt_ms", 1.0)),
            noise_std=float(fixed_params.get("noise_std", 0.7)),
            decision_time_ms=float(fixed_params.get("decision_time_ms", 50.0)),
            noise_gain=float(fixed_params.get("noise_gain", 3.5)),
            threshold_mode=str(
                fixed_params.get("threshold_mode", "participant_block_mean_abs_belief")
            ),
            random_seed=seed_value,
        )
    else:
        decisions_by_trial, rts_by_trial = _simulate_ddm_trials_for_likelihood(
            model_df=model_df,
            n_sims_per_trial=n_draws_per_trial,
            dt_ms=float(fixed_params.get("dt_ms", 1.0)),
            max_duration_ms=float(fixed_params.get("max_duration_ms", 5000.0)),
            boundary_a=float(fixed_params.get("boundary_a", 1.0)),
            non_decision_time_ms=float(fixed_params.get("non_decision_time_ms", 200.0)),
            llr_to_drift_scale=float(fixed_params.get("llr_to_drift_scale", 1.0)),
            start_k=float(fixed_params.get("start_k", 0.1)),
            diffusion_sigma=float(fixed_params.get("diffusion_sigma", 1.0)),
            random_seed=seed_value,
        )

    rng = np.random.default_rng(seed_value + 17)
    sampled_choices, sampled_rts, timeout_fallback_count = _sample_observed_trials_from_simulations(
        decisions_by_trial=decisions_by_trial,
        rts_by_trial=rts_by_trial,
        llr_values=model_df["LLR"].to_numpy(dtype=float),
        rng=rng,
    )

    surrogate_df = model_df.copy()
    surrogate_df["choice"] = sampled_choices.astype(int)
    surrogate_df["reaction_time_ms"] = sampled_rts.astype(float)
    surrogate_df["surrogate_id"] = str(surrogate_id)
    surrogate_df["generating_model_name"] = model_name_str
    surrogate_df["generating_seed"] = int(seed_value)
    surrogate_df["timeout_fallback_count"] = int(timeout_fallback_count)

    return surrogate_df


def fit_models_on_surrogate(
    df_surrogate: pd.DataFrame,
    candidate_models: tuple[str, ...] = SUPPORTED_MODEL_NAMES,
    fit_config: dict[str, object] | None = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Refit all candidate models on one surrogate dataset and return fit table."""
    models = _validate_candidate_models(tuple(candidate_models))
    surrogate_id = (
        str(df_surrogate["surrogate_id"].iloc[0])
        if "surrogate_id" in df_surrogate.columns
        else "unknown_surrogate"
    )
    generating_model_name = (
        str(df_surrogate["generating_model_name"].iloc[0])
        if "generating_model_name" in df_surrogate.columns
        else "unknown_generator"
    )

    rows: list[dict[str, object]] = []
    for model_idx, candidate_model in enumerate(models):
        seed_value = int(random_seed) + model_idx * 31_337
        fit_output = fit_model_parameters(
            df=df_surrogate,
            model_name=candidate_model,
            fit_config=fit_config,
            random_seed=seed_value,
        )

        n_trials = int(len(df_surrogate))
        n_params = int(fit_output["n_parameters"])
        bic_value = float(2.0 * fit_output["best_joint_score"] + n_params * np.log(max(n_trials, 1)))

        rows.append(
            {
                "surrogate_id": surrogate_id,
                "generating_model_name": generating_model_name,
                "candidate_model_name": str(candidate_model),
                "joint_score": float(fit_output["best_joint_score"]),
                "choice_only_score": float(fit_output["best_choice_only_score"]),
                "rt_only_cond_score": float(fit_output["best_rt_only_cond_score"]),
                "bic_score": bic_value,
                "n_trials": n_trials,
                "n_params": n_params,
                "fit_seed": int(seed_value),
                "best_theta": np.asarray(fit_output["best_theta"], dtype=float),
                "best_eta": np.asarray(fit_output["best_eta"], dtype=float),
                "best_named_params": dict(fit_output["best_named_params"]),
            }
        )

    return pd.DataFrame(rows)


def _build_recovery_tables(
    fit_results: pd.DataFrame,
    *,
    score_column: str,
    candidate_models: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = fit_results.sort_values(
        by=["surrogate_id", score_column, "candidate_model_name"],
        ascending=[True, True, True],
    )
    winner_rows = ranked.groupby(
        ["surrogate_id", "generating_model_name"],
        as_index=False,
    ).first()
    winner_rows = winner_rows.rename(columns={"candidate_model_name": "recovered_model_name"})

    counts = pd.crosstab(
        winner_rows["generating_model_name"],
        winner_rows["recovered_model_name"],
    )
    counts = counts.reindex(index=candidate_models, columns=candidate_models, fill_value=0)
    counts.index.name = "generating_model_name"
    counts.columns.name = "recovered_model_name"

    rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return counts, rates


def _build_parameter_recovery_rows(
    pseudo_true_table: pd.DataFrame,
    fit_results: pd.DataFrame,
) -> pd.DataFrame:
    same_model = fit_results[
        fit_results["candidate_model_name"] == fit_results["generating_model_name"]
    ].copy()
    if same_model.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "surrogate_id",
                "param_name",
                "true_value",
                "recovered_value",
                "error",
            ]
        )

    merged = same_model.merge(
        pseudo_true_table[
            ["surrogate_id", "generating_model_name", "theta_vector"]
        ].rename(columns={"theta_vector": "theta_true"}),
        on=["surrogate_id", "generating_model_name"],
        how="left",
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        model_name = str(row.generating_model_name)
        param_spec = get_parameter_spec(model_name)
        theta_true = np.asarray(row.theta_true, dtype=float)
        theta_fit = np.asarray(row.best_theta, dtype=float)
        for param_idx, spec in enumerate(param_spec):
            param_name = str(spec["name"])
            true_value = float(theta_true[param_idx])
            recovered_value = float(theta_fit[param_idx])
            rows.append(
                {
                    "model_name": model_name,
                    "surrogate_id": str(row.surrogate_id),
                    "param_name": param_name,
                    "true_value": true_value,
                    "recovered_value": recovered_value,
                    "error": float(recovered_value - true_value),
                }
            )

    return pd.DataFrame(rows)


def _summarize_parameter_recovery(parameter_rows: pd.DataFrame) -> pd.DataFrame:
    if parameter_rows.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "param_name",
                "n_pairs",
                "mae",
                "bias",
                "pearson_r",
                "spearman_rho",
            ]
        )

    summary_rows: list[dict[str, object]] = []
    grouped = parameter_rows.groupby(["model_name", "param_name"], sort=False)

    for (model_name, param_name), chunk in grouped:
        true_vals = chunk["true_value"].astype(float)
        fit_vals = chunk["recovered_value"].astype(float)

        if len(chunk) >= 2:
            pearson_r = float(true_vals.corr(fit_vals, method="pearson"))
            spearman_rho = float(true_vals.corr(fit_vals, method="spearman"))
        else:
            pearson_r = np.nan
            spearman_rho = np.nan

        summary_rows.append(
            {
                "model_name": str(model_name),
                "param_name": str(param_name),
                "n_pairs": int(len(chunk)),
                "mae": float(np.mean(np.abs(chunk["error"].to_numpy(dtype=float)))),
                "bias": float(np.mean(chunk["error"].to_numpy(dtype=float))),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
            }
        )

    return pd.DataFrame(summary_rows)


def compute_step3_recovery_from_fit_results(
    fit_results: pd.DataFrame,
    pseudo_true_table: pd.DataFrame,
    *,
    candidate_models: tuple[str, ...] = SUPPORTED_MODEL_NAMES,
) -> dict[str, pd.DataFrame]:
    """Build recovery matrices and parameter-recovery summaries from fit outputs."""
    models = _validate_candidate_models(tuple(candidate_models))

    if fit_results.empty:
        empty = pd.DataFrame()
        return {
            "model_recovery_joint_counts": empty,
            "model_recovery_joint_rates": empty,
            "model_recovery_bic_counts": empty,
            "model_recovery_bic_rates": empty,
            "parameter_recovery_rows": empty,
            "parameter_recovery_summary": empty,
        }

    joint_counts, joint_rates = _build_recovery_tables(
        fit_results,
        score_column="joint_score",
        candidate_models=models,
    )
    bic_counts, bic_rates = _build_recovery_tables(
        fit_results,
        score_column="bic_score",
        candidate_models=models,
    )
    parameter_recovery_rows = _build_parameter_recovery_rows(
        pseudo_true_table=pseudo_true_table,
        fit_results=fit_results,
    )
    parameter_recovery_summary = _summarize_parameter_recovery(parameter_recovery_rows)

    return {
        "model_recovery_joint_counts": joint_counts,
        "model_recovery_joint_rates": joint_rates,
        "model_recovery_bic_counts": bic_counts,
        "model_recovery_bic_rates": bic_rates,
        "parameter_recovery_rows": parameter_recovery_rows,
        "parameter_recovery_summary": parameter_recovery_summary,
    }


def compute_step3_soft_gate(
    *,
    model_recovery_joint_rates: pd.DataFrame,
    parameter_recovery_summary: pd.DataFrame,
    joint_diag_min: float = 0.60,
    param_median_r_min: float = 0.30,
) -> dict[str, object]:
    """Compute soft-gate labels from model- and parameter-recovery summaries."""
    if not (0.0 <= float(joint_diag_min) <= 1.0):
        raise ValueError("joint_diag_min must be in [0, 1].")
    if not (-1.0 <= float(param_median_r_min) <= 1.0):
        raise ValueError("param_median_r_min must be in [-1, 1].")

    model_names = list(model_recovery_joint_rates.index.astype(str))
    if parameter_recovery_summary is not None and not parameter_recovery_summary.empty:
        model_names = sorted(set(model_names) | set(parameter_recovery_summary["model_name"].astype(str)))

    rows: list[dict[str, object]] = []
    statuses: list[str] = []

    for model_name in model_names:
        if (
            model_name in model_recovery_joint_rates.index
            and model_name in model_recovery_joint_rates.columns
        ):
            joint_diag_rate = float(model_recovery_joint_rates.loc[model_name, model_name])
        else:
            joint_diag_rate = np.nan

        chunk = (
            parameter_recovery_summary[
                parameter_recovery_summary["model_name"].astype(str) == model_name
            ]
            if parameter_recovery_summary is not None and not parameter_recovery_summary.empty
            else pd.DataFrame()
        )
        median_pearson = (
            float(chunk["pearson_r"].median())
            if not chunk.empty
            else np.nan
        )

        pass_joint = bool(np.isfinite(joint_diag_rate) and joint_diag_rate >= float(joint_diag_min))
        pass_param = bool(np.isfinite(median_pearson) and median_pearson >= float(param_median_r_min))

        if pass_joint and pass_param:
            model_status = "pass"
        elif pass_joint or pass_param:
            model_status = "caution"
        else:
            model_status = "weak"

        statuses.append(model_status)
        rows.append(
            {
                "model_name": model_name,
                "joint_diag_rate": joint_diag_rate,
                "param_median_pearson_r": median_pearson,
                "joint_diag_min": float(joint_diag_min),
                "param_median_r_min": float(param_median_r_min),
                "pass_joint": bool(pass_joint),
                "pass_param": bool(pass_param),
                "status": model_status,
            }
        )

    if not statuses:
        overall_status = "weak"
    elif all(status == "pass" for status in statuses):
        overall_status = "pass"
    elif any(status == "weak" for status in statuses):
        overall_status = "weak"
    else:
        overall_status = "caution"

    return {
        "soft_gate_summary": pd.DataFrame(rows),
        "overall_status": overall_status,
        "thresholds": {
            "joint_diag_min": float(joint_diag_min),
            "param_median_r_min": float(param_median_r_min),
        },
    }


def run_step3_pipeline(
    df_template: pd.DataFrame,
    *,
    run_id: str,
    output_root: str | Path = "data/elias",
    config: dict[str, object],
    overwrite: bool = False,
) -> dict[str, object]:
    """Run canonical Step 3 pipeline and persist all artifacts to disk."""
    normalized_config = _normalize_step3_pipeline_config(config)
    paths = prepare_run_dir(
        output_root,
        pipeline_name=STEP3_PIPELINE_NAME,
        run_id=run_id,
        overwrite=overwrite,
    )

    fit_config = _fit_config_from_step3_config(normalized_config)
    surrogate_config = _surrogate_config_from_step3_config(normalized_config)

    config_path = save_json(normalized_config, paths["run_dir"] / "config.json")

    model_df = _prepare_model_input(df_template)
    rng = np.random.default_rng(int(normalized_config["random_seed"]))
    candidate_models = tuple(normalized_config["candidate_models"])

    pseudo_rows: list[dict[str, object]] = []
    fit_tables: list[pd.DataFrame] = []
    surrogate_meta_rows: list[dict[str, object]] = []
    surrogate_trials_tables: list[pd.DataFrame] = []

    for generating_model in candidate_models:
        sample_seed = int(rng.integers(0, 2_147_483_647))
        pseudo_theta_df = sample_pseudo_true_thetas(
            generating_model,
            n_sets=int(normalized_config["n_surrogates_per_model"]),
            random_seed=sample_seed,
        )

        for set_row in pseudo_theta_df.itertuples(index=False):
            surrogate_id = f"{generating_model}_s{int(set_row.sample_index):03d}"
            simulation_seed = int(rng.integers(0, 2_147_483_647))
            fitting_seed = int(rng.integers(0, 2_147_483_647))

            surrogate_df = simulate_surrogate_dataset(
                df_template=model_df,
                model_name=generating_model,
                theta=np.asarray(set_row.theta_vector, dtype=float),
                random_seed=simulation_seed,
                surrogate_id=surrogate_id,
                surrogate_config=surrogate_config,
            )
            surrogate_df["sample_index"] = int(set_row.sample_index)
            surrogate_trials_tables.append(surrogate_df)

            fit_table = fit_models_on_surrogate(
                df_surrogate=surrogate_df,
                candidate_models=candidate_models,
                fit_config=fit_config,
                random_seed=fitting_seed,
            )
            fit_tables.append(fit_table)

            pseudo_row = {
                "surrogate_id": surrogate_id,
                "generating_model_name": str(generating_model),
                "sample_index": int(set_row.sample_index),
                "theta_vector": np.asarray(set_row.theta_vector, dtype=float),
                "eta_vector": np.asarray(set_row.eta_vector, dtype=float),
                "sampling_seed": int(sample_seed),
                "simulation_seed": int(simulation_seed),
                "fitting_seed": int(fitting_seed),
            }
            spec = get_parameter_spec(generating_model)
            for param_idx, spec_entry in enumerate(spec):
                pseudo_row[f"theta_{spec_entry['name']}"] = float(set_row.theta_vector[param_idx])
            pseudo_rows.append(pseudo_row)

            surrogate_meta_rows.append(
                {
                    "surrogate_id": surrogate_id,
                    "generating_model_name": str(generating_model),
                    "sample_index": int(set_row.sample_index),
                    "n_rows": int(len(surrogate_df)),
                    "n_unique_choices": int(surrogate_df["choice"].nunique()),
                    "min_rt_ms": float(surrogate_df["reaction_time_ms"].min()),
                    "max_rt_ms": float(surrogate_df["reaction_time_ms"].max()),
                    "timeout_fallback_count": int(surrogate_df["timeout_fallback_count"].iloc[0]),
                }
            )

    pseudo_true_table = pd.DataFrame(pseudo_rows)
    surrogate_metadata = pd.DataFrame(surrogate_meta_rows)
    surrogate_trials = (
        pd.concat(surrogate_trials_tables, ignore_index=True)
        if surrogate_trials_tables
        else pd.DataFrame()
    )
    fit_results = pd.concat(fit_tables, ignore_index=True) if fit_tables else pd.DataFrame()

    recovery_outputs = compute_step3_recovery_from_fit_results(
        fit_results,
        pseudo_true_table,
        candidate_models=candidate_models,
    )
    soft_gate_outputs = compute_step3_soft_gate(
        model_recovery_joint_rates=recovery_outputs["model_recovery_joint_rates"],
        parameter_recovery_summary=recovery_outputs["parameter_recovery_summary"],
        joint_diag_min=float(normalized_config["soft_gate_joint_diag_min"]),
        param_median_r_min=float(normalized_config["soft_gate_param_median_r_min"]),
    )

    tables: dict[str, pd.DataFrame] = {
        "pseudo_true_table": pseudo_true_table,
        "surrogate_metadata": surrogate_metadata,
        "surrogate_trials": surrogate_trials,
        "fit_results": fit_results,
        "model_recovery_joint_counts": recovery_outputs["model_recovery_joint_counts"],
        "model_recovery_joint_rates": recovery_outputs["model_recovery_joint_rates"],
        "model_recovery_bic_counts": recovery_outputs["model_recovery_bic_counts"],
        "model_recovery_bic_rates": recovery_outputs["model_recovery_bic_rates"],
        "parameter_recovery_rows": recovery_outputs["parameter_recovery_rows"],
        "parameter_recovery_summary": recovery_outputs["parameter_recovery_summary"],
        "soft_gate_summary": soft_gate_outputs["soft_gate_summary"],
    }

    table_paths: dict[str, str] = {}
    for table_name in _STEP3_TABLE_NAMES:
        table_path = save_table_csv(tables[table_name], paths["tables_dir"] / f"{table_name}.csv")
        table_paths[table_name] = str(table_path)

    manifest_extra = {
        "n_surrogates_total": int(len(surrogate_metadata)),
        "n_fit_rows": int(len(fit_results)),
        "table_paths": table_paths,
        "soft_gate": {
            "overall_status": str(soft_gate_outputs["overall_status"]),
            **soft_gate_outputs["thresholds"],
        },
    }
    manifest = build_basic_manifest(
        run_id=run_id,
        pipeline_name=STEP3_PIPELINE_NAME,
        output_root=output_root,
        run_dir=paths["run_dir"],
        config_path=config_path,
        status=str(soft_gate_outputs["overall_status"]),
        extra=manifest_extra,
    )
    manifest_path = save_json(manifest, paths["run_dir"] / "manifest.json")

    return {
        "run_id": str(run_id),
        "run_dir": paths["run_dir"],
        "tables_dir": paths["tables_dir"],
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "config": normalized_config,
        "tables": tables,
        "table_paths": table_paths,
    }


def load_step3_run(
    run_id: str,
    *,
    output_root: str | Path = "data/elias",
) -> dict[str, object]:
    """Load a persisted Step 3 run and decode all stored tables."""
    data_root = resolve_elias_data_root(output_root)
    run_dir = data_root / STEP3_PIPELINE_NAME / "runs" / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Step 3 run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    manifest = load_json(manifest_path)
    config = load_json(config_path)

    tables: dict[str, pd.DataFrame] = {}
    tables_dir = run_dir / "tables"
    for table_name in _STEP3_TABLE_NAMES:
        table_path = tables_dir / f"{table_name}.csv"
        if table_path.exists():
            tables[table_name] = load_table_csv(table_path)

    return {
        "run_id": str(run_id),
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "manifest": manifest,
        "config": config,
        "tables": tables,
        "table_paths": {
            name: str((tables_dir / f"{name}.csv")) for name in tables.keys()
        },
    }


def list_step3_runs(
    *,
    output_root: str | Path = "data/elias",
) -> pd.DataFrame:
    """List persisted Step 3 runs with manifest metadata."""
    data_root = resolve_elias_data_root(output_root)
    runs_root = data_root / STEP3_PIPELINE_NAME / "runs"
    if not runs_root.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "created_at_utc",
                "status",
                "n_surrogates_total",
                "n_fit_rows",
                "run_dir",
            ]
        )

    rows: list[dict[str, object]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = load_json(manifest_path)
        rows.append(
            {
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "created_at_utc": manifest.get("created_at_utc", ""),
                "status": manifest.get("status", "unknown"),
                "n_surrogates_total": manifest.get("n_surrogates_total", np.nan),
                "n_fit_rows": manifest.get("n_fit_rows", np.nan),
                "run_dir": str(run_dir),
            }
        )

    runs_df = pd.DataFrame(rows)
    if runs_df.empty:
        return runs_df
    return runs_df.sort_values("created_at_utc", ascending=False).reset_index(drop=True)


def run_surrogate_recovery(
    df_template: pd.DataFrame,
    n_surrogates_per_model: int = 20,
    random_seed: int = 0,
    candidate_models: tuple[str, ...] = SUPPORTED_MODEL_NAMES,
    fit_config: dict[str, object] | None = None,
    surrogate_config: dict[str, object] | None = None,
    *,
    run_id: str | None = None,
    output_root: str | Path = "data/elias",
    overwrite: bool = False,
) -> dict[str, object]:
    """Compatibility wrapper that delegates to the canonical Step 3 pipeline."""
    cfg = build_step3_pipeline_config(
        candidate_models=tuple(candidate_models),
        n_surrogates_per_model=int(n_surrogates_per_model),
        random_seed=int(random_seed),
    )

    if fit_config is not None:
        if "n_starts" in fit_config:
            cfg["fit_n_starts"] = int(fit_config["n_starts"])
        if "n_iterations" in fit_config:
            cfg["fit_n_iterations"] = int(fit_config["n_iterations"])
        if "n_sims_per_trial" in fit_config:
            cfg["fit_n_sims_per_trial"] = int(fit_config["n_sims_per_trial"])
        fixed = fit_config.get("fixed_model_params", {})
        if isinstance(fixed, dict):
            if "dt_ms" in fixed:
                cfg["dt_ms"] = float(fixed["dt_ms"])
            if "max_duration_ms" in fixed:
                cfg["max_duration_ms"] = float(fixed["max_duration_ms"])

    if surrogate_config is not None:
        if "n_draws_per_trial" in surrogate_config:
            cfg["surrogate_n_draws_per_trial"] = int(surrogate_config["n_draws_per_trial"])
        fixed = surrogate_config.get("fixed_model_params", {})
        if isinstance(fixed, dict):
            if "dt_ms" in fixed:
                cfg["dt_ms"] = float(fixed["dt_ms"])
            if "max_duration_ms" in fixed:
                cfg["max_duration_ms"] = float(fixed["max_duration_ms"])

    final_run_id = run_id or f"step3_surrogate_seed{int(random_seed)}"
    return run_step3_pipeline(
        df_template,
        run_id=final_run_id,
        output_root=output_root,
        config=cfg,
        overwrite=overwrite,
    )
