from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class RunParams:
    """Parameters for a fermion rTEBD run (matches your standalone script)."""
    L: int = 64
    chi: int = 32
    T: float = 20.0
    N: int = 250
    g: float = 1.5

    # optional checks/extra measurements
    schrodinger_check: bool = False
    renyi_cuts: bool = False
    sqtrace: bool = False

    # output control
    outdir: str = "runs"
    tag: str = ""

    # model params (kept for future extension)
    J: float = 1.0
    k: float = 0.7853981633974483  # pi/4


def load_params(path: str | Path) -> RunParams:
    """
    Load parameters from a YAML file and return a RunParams object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Params file must define a YAML mapping")

    return RunParams(**data)
