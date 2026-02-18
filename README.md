# Glaze

This repository contains an independent implementation of models inspired by:

Glaze, C. M., Kable, J. W., & Gold, J. I. (2015).  
*A normative account of evidence accumulation in unpredictable environments.*  
Nature Neuroscience, 18(12), 1725-1732.

## Project Structure (Current)

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
│       ├── 2026_02_18_choice_fit_run/
│       └── 2026_02_18_joint_fit_run/
└── src/
    ├── common_helpers/
    │   ├── __init__.py
    │   ├── combine_participant_data_csvs.py
    │   └── preprocessing.py
    ├── elias/
    │   ├── workflow.ipynb
    │   └── elias_models/
    │       ├── __init__.py
    │       ├── core_workflow.py
    │       ├── data_pipeline.py
    │       ├── model_fitting.py
    │       ├── model_scoring.py
    │       └── test_core_workflow.py
    ├── evan/
    │   └── glaze.py
    ├── martin/
    │   └── analysis_iribarren.ipynb
    └── old/
```

## Elias Workflow (`src/elias`)

The active Elias path is notebook-first and scoped to real-data fitting:

1. Load and preprocess participant data.
2. Fit blockwise subjective hazard on TRAIN and reconstruct normative state.
3. Fit three candidate models on pooled TRAIN rows:
   - `cont_threshold`
   - `cont_asymptote`
   - `ddm_dnm`
4. Score fitted models on pooled TEST rows.
5. Report a compact comparison table and plot.

Run artifacts are written/read from `data/elias/*_run/` directories (for example `2026_02_18_joint_fit_run`).

## Public Elias API

`src/elias/elias_models/__init__.py` exports only the workflow surface:

- `prepare_modeling_data`
- `fit_models_train_split`
- `score_models_test_split`
- `run_model_comparison`

## Notebook Usage

Primary entrypoint:

- `src/elias/workflow.ipynb`

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
