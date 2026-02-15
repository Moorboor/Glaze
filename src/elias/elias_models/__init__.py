"""Public API for Elias model modules."""

from .continuous_models import run_model_a_threshold, run_model_b_asymptote
from .data_loading import load_participant_data, preprocess_loaded_participant_data
from .ddm_model import run_model_c_ddm
from .likelihood_scoring import score_model_simulation_likelihood
from .orchestration import run_all_models_for_participant
from .optimizer_runner import fit_model_parameters
from .parameter_space import (
    eta_to_theta,
    get_parameter_spec,
    theta_to_eta,
    theta_to_named_params,
    theta_to_scoring_model_params,
)
from .reporting import build_step5_pipeline_config, run_step34_pipeline, run_step345_pipeline
from .surrogate_recovery import (
    build_step3_pipeline_config,
    compute_step3_recovery_from_fit_results,
    compute_step3_soft_gate,
    fit_models_on_surrogate,
    list_step3_runs,
    load_step3_run,
    run_surrogate_recovery,
    run_step3_pipeline,
    sample_pseudo_true_thetas,
    simulate_surrogate_dataset,
)
from .train_test_eval import (
    build_step4_pipeline_config,
    list_step4_runs,
    load_step4_run,
    run_step4_pipeline,
)
from .winner_rules import apply_step4_winner_rules

__all__ = [
    "load_participant_data",
    "preprocess_loaded_participant_data",
    "run_model_a_threshold",
    "run_model_b_asymptote",
    "run_model_c_ddm",
    "run_all_models_for_participant",
    "score_model_simulation_likelihood",
    "fit_model_parameters",
    "build_step3_pipeline_config",
    "run_step3_pipeline",
    "load_step3_run",
    "list_step3_runs",
    "compute_step3_recovery_from_fit_results",
    "compute_step3_soft_gate",
    "sample_pseudo_true_thetas",
    "simulate_surrogate_dataset",
    "fit_models_on_surrogate",
    "run_surrogate_recovery",
    "build_step4_pipeline_config",
    "run_step4_pipeline",
    "load_step4_run",
    "list_step4_runs",
    "apply_step4_winner_rules",
    "build_step5_pipeline_config",
    "run_step34_pipeline",
    "run_step345_pipeline",
    "get_parameter_spec",
    "eta_to_theta",
    "theta_to_eta",
    "theta_to_named_params",
    "theta_to_scoring_model_params",
]
