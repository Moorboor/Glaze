# Glaze

This repository contains an independent implementation of models inspired by:

Glaze, C. M., Kable, J. W., & Gold, J. I. (2015).  
*A normative account of evidence accumulation in unpredictable environments.*  
Nature Neuroscience, 18(12), 1725-1732.

## Collaboration Workflow

All contributors should work in their own repository copy and keep implementation code inside their own folder in `src/` (for example `src/evan/`, `src/elias/`).

Before merging to `main`:

1. Sync with latest `main`.
2. Keep changes scoped to your folder unless a shared file must change.
3. Run your code end-to-end for your scope.
4. Strip notebook outputs and remove temporary/debug files.
5. Review your diff before commit.

## Project Structure

```text
Glaze/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── participants.csv
│   ├── elias.csv
│   ├── evan.csv
│   └── maik.csv
└── src/
    ├── common_helpers/
    │   ├── combine_participant_data_csvs.py
    │   └── preprocessing.py
    ├── elias/
    │   ├── elias_notebook.ipynb       # notebook-first workflow
    │   └── elias_models/
    │       ├── __init__.py
    │       ├── core_workflow.py
    │       ├── constants.py
    │       ├── data_validation.py
    │       ├── data_loading.py
    │       ├── subjective_h.py
    │       ├── environment.py
    │       ├── continuous_models.py
    │       ├── ddm_model.py
    │       ├── likelihood_scoring.py
    │       ├── parameter_space.py
    │       ├── optimizer_runner.py
    │       └── test_core_workflow.py
    ├── evan/
    │   └── glaze.py
    └── old/
```

## Current Elias Workflow (`src/elias`)

The active Elias path is intentionally simplified and notebook-first:

1. Load and preprocess participant data.
2. Infer blockwise subjective hazard from TRAIN choices.
3. Reconstruct normative belief state recursion.
4. Fit three candidate models on pooled TRAIN rows:
   - `cont_threshold`
   - `cont_asymptote`
   - `ddm_dnm`
5. Score fitted models on pooled TEST rows and compare held-out joint score.

This workflow does **not** use surrogate-data generation or multi-step Step3/Step4/Step5 pipeline orchestration.

## Public Elias API

`src/elias/elias_models/__init__.py` exports a reduced interface centered on:

- Data loading/preprocessing:
  - `load_participant_data`
  - `preprocess_loaded_participant_data`
- Subjective hazard and state construction:
  - `fit_blockwise_subjective_h_choice_only`
  - `attach_subjective_h_from_train`
  - `build_normative_belief_columns`
- Model runners and scoring/fitting:
  - `run_model_a_threshold`
  - `run_model_b_asymptote`
  - `run_model_c_ddm`
  - `score_model_simulation_likelihood`
  - `fit_model_parameters`
- Workflow wrappers:
  - `prepare_modeling_data`
  - `fit_models_train_split`
  - `score_models_test_split`
  - `run_model_comparison`

## Notebook Usage

Primary entrypoint:

- `src/elias/elias_notebook.ipynb`

It performs:

1. data loading/preprocessing,
2. subjective-H and normative-state construction,
3. pooled model fitting,
4. pooled held-out scoring,
5. compact comparison table and figure.

## Optional Script Usage

```python
from elias_models import run_model_comparison

result = run_model_comparison(csv_path="data/participants.csv")
print(result["winner_model_name"])
print(result["overview_table"])
```

Run with:

```bash
PYTHONPATH=src:src/elias python your_script.py
```

## Evan Glaze Backend

`src/evan/glaze.py` is kept intact and remains the backend for Elias continuous-model simulation and scoring paths.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate glz
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

## Notebook Output Stripping (`nbstripout`)

```bash
pip install -r requirements.txt
nbstripout --install
```

If needed once after enabling filters:

```bash
git add --renormalize .
git commit -m "Normalize files after enabling nbstripout"
```
