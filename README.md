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
│   └── participants.csv              # participant data of all participants
├── triangle-data/                    # input datasets used by implementations
│   ├── evan-short.csv
│   ├── evan-standard.csv
│   └── maik-standard.csv
└── src/
    ├── <name>/                       # each contributor keeps their code here
    │   └── ...                       # personal implementation files
    ├── evan/                         # current contributor implementation example
    │   └── glaze.py
    └── old/                          # legacy/reference code and notebook
        ├── Group_9_Glaze_2015.ipynb
        ├── glaze_group_pipeline_data.py
        └── group_9_glaze_2015.py
```

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
