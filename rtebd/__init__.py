"""rtebd: a small, self-contained rTEBD/MPDO package (fermion model reference).

This package is generated to match the standalone fermion script behavior,
while exposing an importable API usable from scripts and Jupyter.

Main entrypoints:
  - rtebd.run.run_fermion(params)  -> (run_dir, results)
  - python -m rtebd.cli ...        -> CLI
"""

from .config import RunParams
from .run import run_fermion

__all__ = ["RunParams", "run_fermion"]
