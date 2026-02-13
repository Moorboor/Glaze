# Glaze
This repository contains an independent implementation of the models described in:

Glaze, C. M., Kable, J. W., & Gold, J. I. (2015).
*A normative account of evidence accumulation in unpredictable environments.*
Nature Neuroscience, 18(12), 1725–1732.

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

To disable it on a machine:

```bash
nbstripout --uninstall
```
