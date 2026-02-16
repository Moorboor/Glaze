# Glaze
This repository contains an independent implementation of the models described in:

Glaze, C. M., Kable, J. W., & Gold, J. I. (2015).
*A normative account of evidence accumulation in unpredictable environments.*
Nature Neuroscience, 18(12), 1725–1732.

## Collaboration Workflow

All contributors should work in their own repository copy and keep their implementation inside their personal folder in `src/` (for example: `src/evan/`, `src/martin/`).

Use this workflow:

1. Write and update code only in your own folder: `src/<your-name>/`.
2. Commit and push your work in your own branch.
3. When your task is finished, merge your changes into the `main` branch.

## Before Merging to `main`

Before you merge your work into `main`, complete this checklist:

1. Sync your branch with the latest `main`:
   - `git fetch origin`
   - `git rebase origin/main` (or `git merge origin/main`)
2. Keep your changes inside your own folder (`src/<your-name>/`) unless a shared-file change is required.
3. Run your code and verify it works end-to-end for your part.
4. Make sure notebook outputs are stripped and no temporary/debug files are included.
5. Review your diff (`git diff --stat` and `git diff`) and remove accidental changes.
6. Commit with a clear message and push your branch.
7. Merge into `main` only after conflicts are resolved and the branch is clean.

## Project Structure

```text
Glaze/
├── README.md                         # project setup and workflow notes
├── requirements.txt                  # Python dependencies
├── data/
│   ├── participants.csv              # merged dataset with participant_id column
│   ├── elias.csv                     # participant source CSV (160 rows)
│   ├── evan.csv                      # participant source CSV (short, 147 rows)
│   └── maik.csv                      # participant source CSV (160 rows)
└── src/
    ├── common_helpers/               # shared combine + preprocessing utilities
    │   ├── combine_participant_data_csvs.py
    │   └── preprocessing.py
    ├── elias/                        # participant-wise model comparison workflow
    │   ├── elias_models/             # modular Elias modeling package
    │   │   ├── __init__.py
    │   │   ├── constants.py
    │   │   ├── data_loading.py
    │   │   ├── data_validation.py
    │   │   ├── continuous_models.py
    │   │   ├── ddm_model.py
    │   │   ├── likelihood_scoring.py
    │   │   ├── orchestration.py
    │   │   └── cli.py
    │   └── elias_notebook.ipynb
    ├── evan/                         # Glaze model primitives used by model wrappers
    │   └── glaze.py
    └── old/                          # legacy/reference code and notebook
        ├── Group_9_Glaze_2015.ipynb
        └── group_9_glaze_2015.py
```

## Data Pipeline

The current shared data flow is:

1. Source participant CSVs live in `data/elias.csv`, `data/evan.csv`, and `data/maik.csv`.
2. `src/common_helpers/combine_participant_data_csvs.py` can merge them into `data/participants.csv` with assigned participant IDs (`P01`, `P02`, `P03`).
3. `src/common_helpers/preprocessing.py` provides shared load/preprocessing functions used by `src/elias/elias_models/*` and the Elias notebook.

Current merged dataset expectation:
- `P01` (`elias.csv`): 160 rows
- `P02` (`evan.csv`): 147 rows (short block 1)
- `P03` (`maik.csv`): 160 rows
- Total before exclusions in `participants.csv`: 467 rows

## Modeling Policy (Current)

The active Elias pipeline now follows a Glaze-consistent separation:

1. **Environment layer (objective):**
   - Uses `hazard_rate` to describe or generate hidden-state switches.
   - Produces trial evidence (`LLR`) from a simplified 1D signed-distance setup.
2. **Agent layer (subjective):**
   - Infers **blockwise subjective hazard** from TRAIN choices only (fixed `beta=1`).
   - Reconstructs internal normative beliefs recursively from fitted `H` and `LLR`.
3. **Model fitting/evaluation:**
   - Parameter fitting objective defaults to **choice-only** on TRAIN.
   - Held-out model comparison can still use **TEST joint score** (choice + RT).

Important: Evan’s shorter block is preserved exactly; no missing trials are invented.

## Environment Setup

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate glz
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
```

## Working with `src/elias`

`src/elias` contains the modular modeling package and pipeline CLI.

Main entrypoint:

```bash
PYTHONPATH=src:src/elias python -m elias_models.cli pipeline-run \
  --run-id run_2026_02_15_full \
  --csv-path data/participants.csv \
  --step3-fit-objective choice_only \
  --step4-fit-objective choice_only \
  --output-root data/elias
```

This single command runs Step 3, Step 4, and Step 5.

Worker controls for faster runs:
- Step 3 worker count: `--step3-workers`
- Step 4 worker count: `--step4-workers`
- Step 5 worker count: `--step5-workers`

How this command works:

- `PYTHONPATH=src:src/elias` temporarily adds `src/` and `src/elias/` to Python's import path for this command.
  - `src/` is needed for shared modules like `common_helpers`.
  - `src/elias/` is needed so `elias_models` can be imported as a top-level package.
- `python -m elias_models.cli` runs the module `elias_models/cli.py` as a program.
  - `-m` means "run a module by import name" (instead of running a file path directly).
- `pipeline-run` selects the combined CLI subcommand.
- `--run-id` names the run folder and links Step 3/4/5 outputs.
- `--csv-path` points to the input dataset.
- `--output-root` is the root folder where run artifacts are persisted.


Quick smoke run (reduced simulation counts):

```bash
PYTHONPATH=src:src/elias python -m elias_models.cli pipeline-run \
  --run-id run_2026_02_15_smoke \
  --csv-path data/participants.csv \
  --output-root data/elias \
  --step3-n-surrogates-per-model 1 \
  --step3-surrogate-n-draws-per-trial 16 \
  --step3-fit-n-starts 1 \
  --step3-fit-n-iterations 0 \
  --step3-fit-n-sims-per-trial 20 \
  --step4-fit-n-starts 1 \
  --step4-fit-n-iterations 0 \
  --step4-fit-n-sims-per-trial 20 \
  --step4-eval-n-sims-per-trial 20 \
  --step5-ppc-n-sims-per-trial 20 \
  --step5-ddm-n-samples-per-trial 30 \
  --overwrite
```

Run individual steps (or Step 4+5) with workers:

Step 3 only:

```bash
PYTHONPATH=src:src/elias python -m elias_models.cli surrogate-run \
  --run-id run_2026_02_16_step3_only \
  --csv-path data/participants.csv \
  --output-root data/elias \
  --fit-objective choice_only \
  --workers 10
```

Step 4 only:

```bash
PYTHONPATH=src:src/elias python -m elias_models.cli participant-run \
  --run-id run_2026_02_16_step4_only \
  --csv-path data/participants.csv \
  --output-root data/elias \
  --fit-objective choice_only \
  --workers 10
```

Step 4+5 only (requires existing Step 3 under same `run_id`):

```bash
PYTHONPATH=src:src/elias python -m elias_models.cli pipeline-run-45 \
  --run-id run_2026_02_16_final \
  --csv-path data/participants.csv \
  --output-root data/elias \
  --step4-workers 10 \
  --step5-workers 10
```

Step 3+4+5 together:

```bash
PYTHONPATH=src:src/elias python -m elias_models.cli pipeline-run \
  --run-id run_2026_02_16_full \
  --csv-path data/participants.csv \
  --output-root data/elias \
  --step3-workers 10 \
  --step4-workers 10 \
  --step5-workers 10
```

How many workers make sense:
- Start with `10` workers on a 10-core local machine.
- Do not exceed physical cores: a good upper bound is `os.cpu_count()`.
- Do not exceed available parallel tasks:
  - Step 3 max useful workers is about `n_candidate_models * n_surrogates_per_model`.
  - Step 4 max useful workers is about `n_participants`.
  - Step 5 max useful workers is about `n_participants` (participant-level PPC/latent tasks).
- Above those bounds, overhead usually increases and performance can get worse.

Quick core-count check:

```bash
python -c "import os; print(os.cpu_count())"
```

tmux-friendly long run pattern:

```bash
tmux new -s glaze_run
PYTHONPATH=src:src/elias python -m elias_models.cli pipeline-run \
  --run-id run_2026_02_15_tmux \
  --csv-path data/participants.csv \
  --output-root data/elias
```

Where outputs are saved:

- Unified run root: `data/elias/runs/<run_id>/`
- Pipeline metadata:
  - `data/elias/runs/<run_id>/config.json`
  - `data/elias/runs/<run_id>/manifest.json`
- Step 3 artifacts:
  - `data/elias/runs/<run_id>/step3/config.json`
  - `data/elias/runs/<run_id>/step3/manifest.json`
  - `data/elias/runs/<run_id>/step3/tables/`
- Step 4 artifacts:
  - `data/elias/runs/<run_id>/step4/config.json`
  - `data/elias/runs/<run_id>/step4/manifest.json`
  - `data/elias/runs/<run_id>/step4/tables/`
- Step 5 artifacts:
  - `data/elias/runs/<run_id>/step5/tables/`
  - `data/elias/runs/<run_id>/step5/reports/step5_report.md`
  - `data/elias/runs/<run_id>/step5/logs/step5_error.txt` (only if Step 5 fails)

## Working with `src/evan`

`src/evan` contains the lower-level Glaze primitives used by `src/elias`:

- `psi_function(...)`
- `simulate_trial(...)`

Standalone script mode (for interactive simulation + plots):

```bash
python src/evan/glaze.py
```

Optional block-specific run:

```bash
python src/evan/glaze.py 2
```

These commands load `data/participants.csv` by default and open matplotlib plots for inspection.

## Notebook Output Stripping (`nbstripout`)

This repo tracks notebook files (`*.ipynb`) with the `nbstripout` filter via `.gitattributes`.

On each computer where you clone this repo, run:

```bash
pip install -r requirements.txt
nbstripout --install
```

That installs the git filter in your local git config so notebook outputs are stripped automatically before commit.

If you already cloned this repository before `nbstripout` was added, run the same two commands in your existing local clone after pulling the latest changes.

If notebook files were already tracked with outputs, you can normalize once with:

```bash
git add --renormalize .
git commit -m "Normalize files after enabling nbstripout"
```

To disable it on a machine:

```bash
nbstripout --uninstall
```
