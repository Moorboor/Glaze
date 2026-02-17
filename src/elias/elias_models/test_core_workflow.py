"""Regression checks for the simplified notebook-first Elias workflow."""

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
    @classmethod
    def setUpClass(cls) -> None:
        cls.prep_output = prepare_modeling_data(
            csv_path="data/participants.csv",
            participant_ids=["P01"],
        )
        cls.fit_output = fit_models_train_split(
            cls.prep_output["df_model"],
            fit_config={
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
        df_model = self.prep_output["df_model"]
        for col in ("H", "prev_normative_belief_L", "normative_belief_L"):
            self.assertIn(col, df_model.columns)
        self.assertTrue(np.isfinite(df_model["H"].to_numpy(dtype=float)).all())

    def test_fit_models_train_split_returns_three_models(self) -> None:
        fit_table = self.fit_output["fit_table"]
        self.assertEqual(len(fit_table), 3)
        self.assertEqual(len(self.fit_output["fit_results"]), 3)
        self.assertTrue(np.isfinite(fit_table["best_fit_objective_score"].to_numpy(dtype=float)).all())

    def test_score_models_test_split_returns_finite_scores_and_winner(self) -> None:
        score_table = self.score_output["score_table"]
        self.assertEqual(len(score_table), 3)
        self.assertTrue(np.isfinite(score_table["joint_score_test"].to_numpy(dtype=float)).all())
        winner = str(self.score_output["winner_model_name"])
        self.assertIn(winner, tuple(score_table["model_name"].astype(str).tolist()))
        self.assertEqual(winner, str(score_table.iloc[0]["model_name"]))

    def test_notebook_import_surface_has_no_legacy_pipeline_symbols(self) -> None:
        notebook_path = Path("src/elias/elias_notebook.ipynb")
        self.assertTrue(notebook_path.exists(), msg=f"Notebook not found: {notebook_path}")
        notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        notebook_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook_data.get("cells", [])
            if cell.get("cell_type") in {"markdown", "code"}
        )

        forbidden_markers = (
            "run_step3_pipeline",
            "run_step4_pipeline",
            "run_step345_pipeline",
            "load_step3_artifacts",
            "plot_step3_recovery_diagnostics",
            "surrogate_recovery",
            "participant-run",
            "pipeline-run",
        )
        for marker in forbidden_markers:
            self.assertNotIn(marker, notebook_text)


if __name__ == "__main__":
    unittest.main()
