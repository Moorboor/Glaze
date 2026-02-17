"""Regression tests for the 6-file Elias workflow surface.

This module contains comprehensive workflow-level tests that mirror the notebook execution path,
ensuring that the core pipeline components (prepare, fit, score) operate correctly end-to-end.

How to run the tests:
    From the repository root, run one of the following commands:
    
    # Run all tests in this module
    python -m pytest src/elias/elias_models/test_core_workflow.py -v
    
    # Or using unittest directly
    python -m unittest src.elias.elias_models.test_core_workflow -v
    
    # Run a specific test
    python -m pytest src/elias/elias_models/test_core_workflow.py::CoreWorkflowTests::test_prepare_modeling_data_creates_required_columns -v
    
    # Run with coverage report
    python -m pytest src/elias/elias_models/test_core_workflow.py --cov=elias_models --cov-report=html

    CoreWorkflowTests.setUpClass:
        Build shared prepare/fit/score fixtures once for all test methods.
        Loads participant data, fits candidate models on training split, 
        and scores them on held-out test split with lightweight configuration.

    CoreWorkflowTests.test_prepare_modeling_data_creates_required_columns:
        Verifies that prepared modeling data includes required H and normative state columns
        with finite numerical values.

    CoreWorkflowTests.test_fit_models_train_split_returns_three_models:
        Verifies that fit process returns exactly three candidate models
        with finite fit objective scores.

    CoreWorkflowTests.test_score_models_test_split_returns_finite_scores_and_winner:
        Verifies that scoring on held-out test split returns finite scores,
        identifies a winner model, and ranks it at the top of results.

    CoreWorkflowTests.test_notebook_has_new_step_sections_and_no_legacy_symbols:
        Verifies notebook structure contains required step sections and
        successfully removes legacy pipeline markers and deprecated symbols.

"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from elias_models import (
    fit_models_train_split,
    prepare_modeling_data,
    score_models_test_split,
)


class CoreWorkflowTests(unittest.TestCase):
    """Workflow-level tests that mirror the notebook execution path."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build reusable prepare/fit/score outputs for all test methods."""
        cls.prep_output = prepare_modeling_data(
            csv_path="data/participants.csv",
            participant_ids=["P01"],
        )

        # Call fit_models_train_split to train candidate models on the training split
        cls.fit_output = fit_models_train_split(
            # Pass the preprocessed modeling dataframe from prepare_modeling_data
            cls.prep_output["df_model"],
            # Configuration dictionary specifying fitting hyperparameters
            fit_config={
                # Keep test runtime short while still exercising all code paths.

                "n_starts": 1,  # Number of optimization starting points
                "n_iterations": 1,  # Number of fitting iterations per start
                "n_sims_per_trial": 8,  # Simulations per trial for likelihood computation
                "fit_objective": "choice_only",  # Objective function: choice accuracy only
                "fixed_model_params": {  # Fixed parameters not optimized during fitting
                    "dt_ms": 10.0,  # Time step in milliseconds
                    "min_duration_ms": 150.0,  # Minimum simulated RT floor in milliseconds
                    "max_duration_ms": 1500.0,  # Maximum trial duration in milliseconds
                },
            },
            # Random seed for reproducibility across fitting runs
            random_seed=11,
        )

        cls.score_output = score_models_test_split(
            cls.prep_output["df_model"],
            fitted_models=cls.fit_output["fit_results"],
            n_sims_per_trial=12,
            rt_bin_width_ms=20.0,
            rt_max_ms=5000.0,
            eps=1e-12,
            random_seed=101,
        )

    def test_prepare_modeling_data_creates_required_columns(self) -> None:
        """Prepared modeling data should include finite H and normative state columns."""
        df_model = self.prep_output["df_model"]
        for column in ("H", "prev_normative_belief_L", "normative_belief_L"):
            self.assertIn(column, df_model.columns)
        self.assertTrue(np.isfinite(df_model["H"].to_numpy(dtype=float)).all())

    def test_fit_models_train_split_returns_three_models(self) -> None:
        """Fit output should contain three candidate models with finite objectives."""
        fit_table = self.fit_output["fit_table"]
        self.assertEqual(len(fit_table), 3)
        self.assertEqual(len(self.fit_output["fit_results"]), 3)
        self.assertTrue(np.isfinite(fit_table["best_fit_objective_score"].to_numpy(dtype=float)).all())

    def test_score_models_test_split_returns_finite_scores_and_winner(self) -> None:
        """Held-out scoring should return finite scores and winner at top-ranked row."""
        score_table = self.score_output["score_table"]
        self.assertEqual(len(score_table), 3)
        self.assertTrue(np.isfinite(score_table["joint_score_test"].to_numpy(dtype=float)).all())

        winner = str(self.score_output["winner_model_name"])
        self.assertIn(winner, tuple(score_table["model_name"].astype(str).tolist()))
        self.assertEqual(winner, str(score_table.iloc[0]["model_name"]))

    def test_notebook_has_new_step_sections_and_no_legacy_symbols(self) -> None:
        """Notebook should use the new step framing and avoid removed pipeline markers."""
        notebook_path = Path("src/elias/elias_notebook.ipynb")
        self.assertTrue(notebook_path.exists(), msg=f"Notebook not found: {notebook_path}")
        notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        notebook_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook_data.get("cells", [])
            if cell.get("cell_type") in {"markdown", "code"}
        )

        required_step_markers = (
            "Step 0: Setup and imports",
            "Step 1: Load and preprocess data",
            "Step 2: Fit subjective hazard and build normative state",
            "Step 3: Fit candidate models on TRAIN",
            "Step 4: Score fitted models on TEST",
            "Step 5: Compact summary and plot",
        )
        for marker in required_step_markers:
            self.assertIn(marker, notebook_text)

        forbidden_markers = (
            "run_step3_pipeline",
            "run_step4_pipeline",
            "run_step345_pipeline",
            "load_step3_artifacts",
            "plot_step3_recovery_diagnostics",
            "surrogate_recovery",
            "participant-run",
            "pipeline-run",
            "analysis_io",
            "analysis_plots",
        )
        for marker in forbidden_markers:
            self.assertNotIn(marker, notebook_text)


if __name__ == "__main__":
    unittest.main()
