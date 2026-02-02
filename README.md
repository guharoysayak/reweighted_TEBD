# reweighted_TEBD (rTEBD)

Reweighted TEBD (rTEBD) is a Matrix Product Density Operator (MPDO)–based tensor-network method for improved quantum dynamics simulations.

Reference implementation for:
https://arxiv.org/abs/2412.08730

Zenodo DOI:
https://doi.org/10.5281/zenodo.17479681

---

## Repository layout

```text
reweighted_TEBD/
├── rtebd/                 # importable Python package
├── scripts/               # standalone paper scripts
├── params_fermion.yaml    # example parameter file
├── runs/                  # generated output (not tracked)
├── pyproject.toml
└── README.md

## Installation

git clone https://github.com/guharoysayak/reweighted_TEBD.git
cd reweighted_TEBD
pip install -e .
pip install pyyaml
