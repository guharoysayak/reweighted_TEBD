from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import RunParams

def _git_commit_hash() -> str:
    try:
        import subprocess
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception:
        return ""

def make_run_dir(base: Path, params: RunParams) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{params.tag}" if params.tag else ""
    name = f"{ts}{tag}_L{params.L}_chi{params.chi}_g{params.g}_T{params.T}_N{params.N}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )

def save_run(run_dir: Path, params: RunParams, results: dict) -> None:
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    payload = {"params": asdict(params), "meta": meta}
    (run_dir / "params.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    np.savez_compressed(run_dir / "results.npz", **results)

    # small summary
    summary_lines = []
    if "Et_TEBD" in results:
        Et = np.asarray(results["Et_TEBD"])
        summary_lines.append(f"Et_TEBD: shape={Et.shape}, real[min,max]=({np.real(Et).min():.6g},{np.real(Et).max():.6g})")
    if "tr_TB" in results:
        tr = np.asarray(results["tr_TB"])
        summary_lines.append(f"trace: shape={tr.shape}, real[min,max]=({np.real(tr).min():.6g},{np.real(tr).max():.6g})")
    if summary_lines:
        (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
