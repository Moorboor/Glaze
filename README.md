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
│   ├── maik.csv
│   └── elias/
│       └── runs/                 # historical artifacts from old workflow
└── src/
    ├── common_helpers/
    │   ├── combine_participant_data_csvs.py
    │   └── preprocessing.py
    ├── elias/
    │   ├── elias_notebook.ipynb
    │   └── elias_models/
    │       ├── __init__.py
    │       ├── core_workflow.py
    │       ├── data_pipeline.py
    │       ├── model_fitting.py
    │       ├── model_scoring.py
    │       └── test_core_workflow.py
    ├── evan/
    │   └── glaze.py
    └── old/
```

## Elias Workflow (`src/elias`)

The active Elias path is notebook-first and intentionally scoped to real-data fitting:

1. Load and preprocess participant data.
2. Fit blockwise subjective hazard on TRAIN and reconstruct normative state.
3. Fit three candidate models on pooled TRAIN rows:
   - `cont_threshold`
   - `cont_asymptote`
   - `ddm_dnm`
4. Score fitted models on pooled TEST rows.
5. Report a compact comparison table and plot.

This workflow does not include surrogate-data generation or the old multi-step Step3/Step4/Step5 artifact pipeline.

## Public Elias API

`src/elias/elias_models/__init__.py` exports only the workflow surface:

- `prepare_modeling_data`
- `fit_models_train_split`
- `score_models_test_split`
- `run_model_comparison`

## Notebook Usage

Primary entrypoint:

- `src/elias/elias_notebook.ipynb`

Notebook sections:

1. Step 0: Setup and imports
2. Step 1: Load and preprocess data
3. Step 2: Fit subjective hazard and build normative state
4. Step 3: Fit candidate models on TRAIN
5. Step 4: Score fitted models on TEST
6. Step 5: Compact summary and plot

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

`src/evan/glaze.py` is unchanged and remains the backend for Elias continuous-model and DDM simulation paths.

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
