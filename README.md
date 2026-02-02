# reweighted_TEBD (rTEBD)

Reweighted TEBD (rTEBD) is a Matrix Product Density Operator (MPDO)–based tensor-network method for improved quantum dynamics simulations.

This repository contains the reference implementations used in:
https://arxiv.org/abs/2412.08730

If you use this code, please cite the above paper.

Zenodo DOI:
[https://doi.org/10.5281/zenodo.17479681](https://doi.org/10.5281/zenodo.18452771)

## Development status

The `rtebd` package is under active development.

At present, the packaged implementation supports **fermionic models**. Support for additional models (e.g. spin and bosonic systems) will be added in future releases.

The code is written in a modular way: users can define **custom initial states**, **Hamiltonians**, and **time-evolution gates**, and run them within the same rTEBD framework by extending the relevant modules.

The **spin-model implementation used in the paper** is provided in the `scripts/` directory as a standalone reference and is not yet integrated into the `rtebd` package.

---

## Repository layout

```text
reweighted_TEBD/
├── rtebd/                 # importable Python package (library + CLI)
├── scripts/               # standalone scripts used for paper figures
├── params_fermion.yaml    # example parameter file for fermion runs
├── runs/                  # generated output directory (created at runtime)
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/guharoysayak/reweighted_TEBD.git
cd reweighted_TEBD
pip install -e .
pip install pyyaml
```

---

## Run a fermion simulation (CLI)

```bash
python -m rtebd.cli params_fermion.yaml
```

---

## Output directory structure

```text
runs/
└── YYYYMMDD_HHMMSS_L64_chi32_g1.5_T20.0_N250/
    ├── params.json
    ├── results.npz
    └── run.log
```

---

## Loading results (Python / Jupyter)

```python
import json
import numpy as np
from pathlib import Path

run_dir = Path("runs/YYYYMMDD_HHMMSS_L64_chi32_g1.5_T20.0_N250")

with open(run_dir / "params.json") as f:
    payload = json.load(f)

params = payload["params"]
meta = payload["meta"]

results = np.load(run_dir / "results.npz")
print(results.files)

Et = results["Et_TEBD"]
ni = results["ni_psite"]
```

---

## Example parameter file (`params_fermion.yaml`)

```yaml
L: 64
chi: 32

T: 20.0
N: 250

g: 1.5
J: 1.0
k: 0.7853981633974483

schrodinger_check: true
renyi_cuts: false
sqtrace: true

outdir: runs
tag: fermion_test
```

---

## Parameter definitions

```text
L                  system size
chi                MPDO bond dimension
T                  total evolution time
N                  number of Trotter steps (dt = T / N)
g                  reweighting parameter (g = 1 → standard MPDO-TEBD)
J, k               model parameters

schrodinger_check  exact Schrödinger evolution (small systems only)
renyi_cuts         Rényi entropy across bipartitions
sqtrace            global purity / normalization diagnostics
```

---

## Reproducing paper results

```bash
python scripts/run_<script_name>.py
```

All scripts in `scripts/` are self-contained and reproduce the numerical data shown in the paper.
They do not depend on the `rtebd/` package.

---

## Design philosophy

```text
scripts/  → frozen, standalone paper scripts
rtebd/    → reusable library and CLI
params    → explicit, version-controlled inputs
runs/     → reproducible outputs (parameters + metadata + results)
```

---

## Citation

```bibtex
@misc{roy2025reweightedtimeevolvingblockdecimation,
      title={Reweighted Time-Evolving Block Decimation for Improved Quantum Dynamics Simulations}, 
      author={Sayak Guha Roy and Kevin Slagle},
      year={2025},
      eprint={2412.08730},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2412.08730}, 
}
```

Zenodo:
```text
[https://doi.org/10.5281/zenodo.17479681](https://doi.org/10.5281/zenodo.17479681)
```

---

## License

```text
See LICENSE
```
