"""Regression checks for the Glaze-consistent internal-H refactor.

These are lightweight unit tests intended for local/CI validation.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from elias_models.environment import generate_environment_from_template
from elias_models.optimizer_runner import fit_model_parameters
from elias_models.subjective_h import (
    attach_subjective_h_from_train,
    build_normative_belief_columns,
    fit_blockwise_subjective_h_choice_only,
)


class GlazeRefactorTests(unittest.TestCase):
    def test_environment_preserves_trial_structure(self) -> None:
        template = pd.DataFrame(
            {
                "participant_id": ["P01"] * 5 + ["P02"] * 3,
                "block_id": [1] * 5 + [1] * 3,
                "trial_index": [1, 2, 3, 4, 5, 1, 2, 3],
                "hazard_rate": [0.1] * 8,
                "noise_sigma": [0.33] * 8,
            }
        )
        out = generate_environment_from_template(template, random_seed=7)

        self.assertEqual(len(out), len(template))
        self.assertListEqual(
            out[["participant_id", "block_id", "trial_index"]].values.tolist(),
            template[["participant_id", "block_id", "trial_index"]].values.tolist(),
        )
        self.assertIn("true_state", out.columns)
        self.assertIn("signed_distance_from_center", out.columns)
        self.assertIn("LLR", out.columns)

    def test_subjective_h_fit_and_normative_state_columns(self) -> None:
        df = pd.DataFrame(
            {
                "participant_id": ["P01"] * 6,
                "block_id": [1] * 6,
                "trial_index": [1, 2, 3, 4, 5, 6],
                "split": ["TRAIN", "TRAIN", "TRAIN", "TEST", "TEST", "TEST"],
                "LLR": [1.2, 0.7, 1.0, 0.8, 0.9, 1.1],
                "choice": [1, 1, 1, 1, 1, 1],
                "hazard_rate": [0.1] * 6,
                "noise_sigma": [0.33] * 6,
                "belief_L": [0.0] * 6,
                "reaction_time_ms": [500.0] * 6,
                "row_id": list(range(6)),
            }
        )

        h_table = fit_blockwise_subjective_h_choice_only(df)
        self.assertEqual(len(h_table), 1)
        self.assertTrue(0.0 < float(h_table["fitted_subjective_h"].iloc[0]) < 1.0)

        with_h = attach_subjective_h_from_train(df, h_table)
        with_state = build_normative_belief_columns(with_h)
        self.assertIn("H", with_state.columns)
        self.assertIn("prev_normative_belief_L", with_state.columns)
        self.assertIn("normative_belief_L", with_state.columns)

    def test_optimizer_uses_choice_only_objective_when_requested(self) -> None:
        dummy_df = pd.DataFrame({"x": [0]})

        fake_score = {
            "aggregate_scores": {
                "joint_score": 5.0,
                "choice_only_score": 1.0,
                "rt_only_cond_score": 4.0,
            },
            "trial_scores": pd.DataFrame(),
        }

        with patch(
            "elias_models.optimizer_runner.score_model_simulation_likelihood",
            return_value=fake_score,
        ):
            out_choice = fit_model_parameters(
                dummy_df,
                model_name="ddm_dnm",
                fit_config={
                    "n_starts": 1,
                    "n_iterations": 0,
                    "fit_objective": "choice_only",
                },
                random_seed=1,
            )
            out_joint = fit_model_parameters(
                dummy_df,
                model_name="ddm_dnm",
                fit_config={
                    "n_starts": 1,
                    "n_iterations": 0,
                    "fit_objective": "joint",
                },
                random_seed=1,
            )

        self.assertEqual(out_choice["fit_objective"], "choice_only")
        self.assertAlmostEqual(float(out_choice["best_fit_objective_score"]), 1.0)
        self.assertEqual(out_joint["fit_objective"], "joint")
        self.assertAlmostEqual(float(out_joint["best_fit_objective_score"]), 5.0)


if __name__ == "__main__":
    unittest.main()
