from __future__ import annotations
import sys
from pathlib import Path

from .config import load_params
from .run import run_fermion

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        raise RuntimeError(
            "Usage: python -m rtebd.cli <params.yaml>"
        )

    params_file = Path(argv[0])
    params = load_params(params_file)
    run_fermion(params)

if __name__ == "__main__":
    main()
