"""Regression tests for the 6-file Elias workflow surface.

Function Inventory:
- `CoreWorkflowTests.setUpClass`: Build shared prepare/fit/score fixtures once.
- `CoreWorkflowTests.test_prepare_modeling_data_creates_required_columns`: Verifies H and normative state columns.
- `CoreWorkflowTests.test_fit_models_train_split_returns_three_models`: Verifies three fitted models and finite fit scores.
- `CoreWorkflowTests.test_score_models_test_split_returns_finite_scores_and_winner`: Verifies finite held-out scores and winner consistency.
- `CoreWorkflowTests.test_notebook_has_new_step_sections_and_no_legacy_symbols`: Verifies notebook structure and legacy-symbol removal.
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

        cls.fit_output = fit_models_train_split(
            cls.prep_output["df_model"],
            fit_config={
                # Keep test runtime short while still exercising all code paths.
                "n_starts": 1,
                "n_iterations": 0,
                "n_sims_per_trial": 8,
                "fit_objective": "choice_only",
                "fixed_model_params": {
                    "dt_ms": 10.0,
                    "max_duration_ms": 1500.0,
                },
            },
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
