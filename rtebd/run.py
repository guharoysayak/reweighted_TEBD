from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .config import RunParams
from .fermion import MPDO
from .io import make_run_dir, save_run, setup_logging

def run_fermion(params: RunParams) -> tuple[Path, dict]:
    """Run a fermion rTEBD simulation and save outputs (same arrays as the script)."""
    base = Path(params.outdir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(base, params)
    setup_logging(run_dir)

    logging.info("Starting rTEBD fermion run")
    logging.info("Run directory: %s", run_dir)
    logging.info("Params: %s", params)

    L, N = int(params.L), int(params.N)

    # Preallocate outputs (matches the standalone script)
    E_psite = np.zeros((L - 1, N), dtype=np.complex128)
    ni_psite = np.zeros((L, N), dtype=np.complex128)

    ni_conn = []
    tr_TB = []
    tr2_TB = []
    Et_TEBD = []
    Ef_TEBD = []
    ck_TEBD = []

    if params.sqtrace:
        norm_trace_TEBD = []
        renyi_F = []
    if params.renyi_cuts:
        renyi_L = []
        renyi_R = []

    if params.schrodinger_check:
        E_psite_sch = np.zeros((L - 1, N), dtype=np.complex128)
        ni_psite_sch = np.zeros((L, N), dtype=np.complex128)
        Et_sch = []
        renyi_L_test = []
        renyi_R_test = []
        Ef_sch = []

    mps = MPDO(
        L=params.L,
        chi=params.chi,
        T=params.T,
        N=params.N,
        g=params.g,
        J=params.J,
        k=params.k,
        schrodinger_check=params.schrodinger_check,
        renyi_cuts=params.renyi_cuts,
        sqtrace=params.sqtrace,
    )

    t_start = time.time()
    for i in range(N):
        step_start = time.time()
        mps.sweepU()
        if i == N - 1:
            mps.measure_Gij()
        step_end = time.time()
        logging.info("Step %d/%d done in %.3fs", i + 1, N, step_end - step_start)

        for j in range(L):
            ni_psite[j, i] = mps.ni_persite[j]
            if j != L - 1:
                E_psite[j, i] = mps.E_persite[j]

        Et_TEBD.append(mps.E_total_TEBD)
        ni_conn.append(mps.ni_connect)
        tr_TB.append(mps.tr_TEBD)
        Ef_TEBD.append(mps.E_fourier)
        ck_TEBD.append(mps.ck_correl)

        if params.sqtrace:
            norm_trace_TEBD.append(mps.norm_trace)
            tr2_TB.append(mps.sq_trace)
            renyi_F.append(mps.renyi_full)

        if params.renyi_cuts:
            renyi_L.append(mps.renyi_left)
            renyi_R.append(mps.renyi_right)

        if params.schrodinger_check:
            for j in range(L):
                ni_psite_sch[j, i] = mps.ni_persite_sch[j]
                if j != L - 1:
                    E_psite_sch[j, i] = mps.E_persite_sch[j]
            Et_sch.append(mps.E_total_sch)
            Ef_sch.append(mps.E_fourier_sch)
            renyi_L_test.append(mps.renyi_test_left)
            renyi_R_test.append(mps.renyi_test_right)

    elapsed = time.time() - t_start
    logging.info("Finished run in %.2fs (%.2fs/step average)", elapsed, elapsed / max(N, 1))

    results: dict[str, np.ndarray] = {
        "E_psite": E_psite,
        "ni_psite": ni_psite,
        "Et_TEBD": np.asarray(Et_TEBD, dtype=np.complex128),
        "ni_conn": np.asarray(ni_conn, dtype=np.complex128),
        "tr_TB": np.asarray(tr_TB, dtype=np.complex128),
        "Ef_TEBD": np.asarray(Ef_TEBD, dtype=np.complex128),
        "ck_TEBD": np.asarray(ck_TEBD, dtype=np.complex128),
        "Gij_final": np.asarray(mps.Gij, dtype=np.complex128),
    }

    if params.sqtrace:
        results["norm_trace_TEBD"] = np.asarray(norm_trace_TEBD, dtype=np.complex128)
        results["tr2_TB"] = np.asarray(tr2_TB, dtype=np.complex128)
        results["renyi_F"] = np.asarray(renyi_F, dtype=np.complex128)

    if params.renyi_cuts:
        results["renyi_L"] = np.asarray(renyi_L, dtype=np.complex128)
        results["renyi_R"] = np.asarray(renyi_R, dtype=np.complex128)

    if params.schrodinger_check:
        results.update(
            {
                "E_psite_sch": E_psite_sch,
                "ni_psite_sch": ni_psite_sch,
                "Et_sch": np.asarray(Et_sch, dtype=np.complex128),
                "Ef_sch": np.asarray(Ef_sch, dtype=np.complex128),
                "renyi_L_test": np.asarray(renyi_L_test, dtype=np.complex128),
                "renyi_R_test": np.asarray(renyi_R_test, dtype=np.complex128),
            }
        )

    save_run(run_dir, params, results)
    logging.info("Saved results to %s", run_dir)
    return run_dir, results
