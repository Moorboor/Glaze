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
    │   ├── elias_ddm.py
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
3. `src/common_helpers/preprocessing.py` provides shared load/preprocessing functions used by `src/elias/elias_ddm.py` and the Elias notebook.

Current merged dataset expectation:
- `P01` (`elias.csv`): 160 rows
- `P02` (`evan.csv`): 147 rows (short block 1)
- `P03` (`maik.csv`): 160 rows
- Total before exclusions in `participants.csv`: 467 rows

## Environment Setup

Create and activate a conda environment, then install dependencies:

```bash
conda create -n glaze python=3.11 -y
conda activate glaze
pip install -r requirements.txt
```

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
