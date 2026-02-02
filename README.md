# reweighted_TEBD (rTEBD)

Reweighted TEBD (rTEBD) is a Matrix Product Density Operator (MPDO) based tensor-network method for improved quantum dynamics simulations.
This repo contains the reference implementations used in https://arxiv.org/abs/2412.08730.

**Zenodo DOI:** 10.5281/zenodo.17479681

---

## Repository layout

- `rtebd/` — importable Python package (library + CLI entrypoints)
- `scripts/` — standalone scripts used to generate paper figures (do not depend on `rtebd/`)
- `params_fermion.yaml` — example parameter file for a fermion run
- `runs/` — (generated) output directory (created automatically; not tracked in git)

---

## Installation

### Option A: editable install (recommended)
```bash
git clone https://github.com/guharoysayak/reweighted_TEBD.git
cd reweighted_TEBD
pip install -e .


