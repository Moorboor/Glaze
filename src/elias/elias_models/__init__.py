# Public package surface for Elias model code.
# Re-exports core model runners, scoring/fitting helpers, and notebook workflow wrappers.

"""Public API for Elias model modules."""

from .continuous_models import run_model_a_threshold, run_model_b_asymptote
from .core_workflow import (
    fit_models_train_split,
    prepare_modeling_data,
    run_model_comparison,
    score_models_test_split,
)
from .data_loading import load_participant_data, preprocess_loaded_participant_data
from .ddm_model import run_model_c_ddm
from .environment import generate_environment_from_template, objective_h_mean_from_template
from .likelihood_scoring import score_model_simulation_likelihood
from .optimizer_runner import fit_model_parameters
from .parameter_space import (
    eta_to_theta,
    get_parameter_spec,
    theta_to_eta,
    theta_to_named_params,
    theta_to_scoring_model_params,
)
from .subjective_h import (
    SubjectiveHGrid,
    attach_subjective_h_from_train,
    build_normative_belief_columns,
    fit_blockwise_subjective_h_choice_only,
    glaze_psi,
)

__all__ = [
    "load_participant_data",
    "preprocess_loaded_participant_data",
    "fit_blockwise_subjective_h_choice_only",
    "attach_subjective_h_from_train",
    "build_normative_belief_columns",
    "SubjectiveHGrid",
    "glaze_psi",
    "generate_environment_from_template",
    "objective_h_mean_from_template",
    "run_model_a_threshold",
    "run_model_b_asymptote",
    "run_model_c_ddm",
    "score_model_simulation_likelihood",
    "fit_model_parameters",
    "get_parameter_spec",
    "eta_to_theta",
    "theta_to_eta",
    "theta_to_named_params",
    "theta_to_scoring_model_params",
    "prepare_modeling_data",
    "fit_models_train_split",
    "score_models_test_split",
    "run_model_comparison",
]
