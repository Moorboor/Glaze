"""Workflow-only public API for Elias model comparison.

Function Inventory:
- `prepare_modeling_data`: Build model-ready dataframe from real participant data.
- `fit_models_train_split`: Fit all three candidate models on pooled TRAIN rows.
- `score_models_test_split`: Score fitted models on pooled TEST rows.
- `run_model_comparison`: One-call wrapper that executes prepare+fit+score and returns winner tables.

Call-site context:
- Imported by `src/elias/elias_notebook.ipynb` and external scripts.
- Backed internally by `core_workflow.py`.
"""

from .core_workflow import (
    fit_models_train_split,
    prepare_modeling_data,
    run_model_comparison,
    score_models_test_split,
)

__all__ = [
    "prepare_modeling_data",
    "fit_models_train_split",
    "score_models_test_split",
    "run_model_comparison",
]
