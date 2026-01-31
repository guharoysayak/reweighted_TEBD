"""
rTEBD for a GHZ initial state

Run from terminal:
  python rtebd_ghz_fermion.py --L 128 --chi 64 --g 1.5 --T 20 --N 250

Run from Jupyter:
  from rtebd_ghz_fermion import main
  main(["--L","8","--chi","16","--g","1.5","--T","2","--N","50"])

Outputs:
  runs/<timestamp>_L{L}_chi{chi}_g{g}_T{T}_N{N}/
    params.json
    results.npz
    summary.txt
    run.log
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import expm, svd


# -----------------------------
# Global knobs (set in main())
# -----------------------------
g: float = 1.0
bound_diff: bool = False
sch_bool: bool = False
_renyi: bool = False
_sqtrace: bool = False


# -----------------------------
# Helpers (linear algebra etc.)
# -----------------------------
def vec_conj(a):
    return np.conjugate(a)

def vec_dot(a, b):
    # second vector is conjugated
    return np.dot(a, vec_conj(b))

def dagger(M):
    return np.conjugate(np.transpose(M))

def mat_dot2(A, B):
    return np.dot(A, B)

def mat_dot4(A, B, C, D):
    return np.dot(A, np.dot(B, np.dot(C, D)))

def normalize(psi):
    return psi / np.sqrt(vec_dot(psi, psi))

def s_x():
    return np.matrix([[0, 1.], [1., 0]])

def s_y():
    return np.matrix([[0, -1j], [1j, 0]])

def s_z():
    return np.matrix([[1., 0], [0, -1.]])

def s_p():
    return np.matrix([[0, 1.], [0, 0]])

def pauli_normal():
    return [np.eye(2), np.array(s_x()), np.array(s_y()), np.array(s_z())]

def pauli_tilde():
    # reweighted basis
    return [np.eye(2), g * np.array(s_x()), g * np.array(s_y()), g * g * np.array(s_z())]

def pauli_bar():
    # inverse-reweighted basis
    return [np.eye(2), (1 / g) * np.array(s_x()), (1 / g) * np.array(s_y()), (1 / (g * g)) * np.array(s_z())]


# ---------------------------------
# Model / init helpers (fermions)
# ---------------------------------
def non_int_fermi(J):
    hi = J * np.array(
        [[0, 0, 0, 0],
         [0, 0, -1, 0],
         [0, -1, 0, 0],
         [0, 0, 0, 0]],
        dtype=np.complex128
    )
    return hi

def U_mat(dt, U):
    # U is 4x4 two-site unitary; returns 4x4x4x4 transfer tensor
    Ud = dagger(U)
    U_all = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    sg1 = np.kron(pauli_bar()[i], pauli_bar()[j])
                    sg2 = np.kron(pauli_tilde()[k], pauli_tilde()[l])
                    U_all[i][j][k][l] = (1 / 4) * np.trace(mat_dot4(sg1, U, sg2, Ud))
    return U_all

def init_fermi_psi_sch(L):
    # cat-like superposition used in your script
    t1 = np.array([1, 0])
    t2 = np.array([0, 1])

    psi1 = 1
    for i in range(int(L)):
        if (i + 1) % 8 in [1, 2, 7, 0]:
            psi1 = np.kron(psi1, t2)
        elif (i + 1) % 8 in [3, 4, 5, 6]:
            psi1 = np.kron(psi1, t1)

    psi2 = 1
    for i in range(L):
        if (i + 1) % 8 in [1, 2, 7, 0]:
            psi2 = np.kron(psi2, t1)
        elif (i + 1) % 8 in [3, 4, 5, 6]:
            psi2 = np.kron(psi2, t2)

    return (1 / np.sqrt(2)) * (psi1 + psi2)

def init_fermi_psi(L):
    t1 = np.array([1, 0])
    t2 = np.array([0, 1])
    psi1 = np.zeros((L, 2), dtype=np.complex128)
    psi2 = np.zeros((L, 2), dtype=np.complex128)

    for i in range(L):
        if (i + 1) % 8 in [1, 2, 7, 0]:
            psi1[i] += t2
        elif (i + 1) % 8 in [3, 4, 5, 6]:
            psi1[i] += t1

    for i in range(L):
        if (i + 1) % 8 in [1, 2, 7, 0]:
            psi2[i] += t1
        elif (i + 1) % 8 in [3, 4, 5, 6]:
            psi2[i] += t2

    return [psi1, psi2]

def generate_MPDO(psi, L):
    A_dict_temp = {}
    for i in range(L):
        A_temp = np.zeros(4, dtype=np.complex128)
        rho = np.outer(psi[i], np.conjugate(psi[i]))
        for j in range(4):
            A_temp[j] = np.trace(mat_dot2(pauli_bar()[j], rho))
        A_dict_temp[f"A{i}"] = A_temp
    return A_dict_temp

def add2_MPDO(psi1, psi2, L):
    A_dict1 = generate_MPDO(psi1, L)
    A_dict2 = generate_MPDO(psi2, L)
    A_dict = {}
    for i in range(L):
        A_temp = np.zeros((2, 4, 2), dtype=np.complex128)
        for j in range(4):
            A_temp[0, j, 0] = A_dict1[f"A{i}"][j]
            A_temp[1, j, 1] = A_dict2[f"A{i}"][j]
        A_dict[f"A{i}"] = A_temp
    A_dict["A0"] = (1 / np.sqrt(2)) * A_dict["A0"]
    return A_dict

def init_MPDO_dict_fermi(L):
    psi1, psi2 = init_fermi_psi(L)
    return add2_MPDO(psi1, psi2, L)

def n_op():
    return np.array([[0, 0], [0, 1]], dtype=np.complex128)


# -----------------------------
# MPDO class
# -----------------------------
class MPDO:
    J = 1
    k = np.pi / 4

    def __init__(self, L, chi, T, N):
        self.L = L
        self.chi = chi
        self.T = T
        self.N = N
        self.dt = self.T / self.N

        Hi = non_int_fermi(self.J)
        self.sU = expm(-1j * self.dt * Hi)
        self.U = U_mat(self.dt, self.sU)

        self.A_dict = init_MPDO_dict_fermi(self.L)
        self.lmbd_position = 0

        if sch_bool:
            self.schrodinger_psi = init_fermi_psi_sch(self.L)
            self.E_persite_sch = np.zeros(self.L - 1, dtype=np.complex128)
            self.ni_persite_sch = np.zeros(self.L, dtype=np.complex128)
            self.E_total_sch = 0
            self.E_fourier_sch = 0
            self.ni_connect_sch = 0

        self.tr_TEBD = 0
        self.norm_trace = 0
        self.renyi_left = 0
        self.renyi_right = 0
        self.renyi_full = 0
        self.E_persite = np.zeros(self.L - 1, dtype=np.complex128)
        self.ni_persite = np.zeros(self.L, dtype=np.complex128)
        self.ni_connect = 0
        self.E_total_TEBD = 0
        self.sq_trace = 0
        self.E_fourier = 0

    def applyU_schrodinger(self, ind, sU):
        self.schrodinger_psi = np.reshape(self.schrodinger_psi, (2**ind, 4, 2**(self.L - 2 - ind)))
        self.schrodinger_psi = np.einsum('ij, ajb -> aib', sU, self.schrodinger_psi, optimize='optimal')
        self.schrodinger_psi = self.schrodinger_psi.flatten()

    def applyU(self, ind, dirc, U, sU, lm=False):
        # relocate lambda
        if not lm:
            if dirc == 'left':
                self.lmbd_relocate(ind[1])
            elif dirc == 'right':
                self.lmbd_relocate(ind[0])

        if (not lm) and sch_bool:
            self.applyU_schrodinger(ind[0], sU)

        A1 = self.A_dict[f"A{ind[0]}"]
        A2 = self.A_dict[f"A{ind[1]}"]
        chi1 = np.shape(A1)[0]
        chi2 = np.shape(A2)[2]

        s1 = np.einsum('ijkl,akb,blc->aijc', U, A1, A2, optimize='optimal')
        s2 = np.reshape(s1, (4 * chi1, 4 * chi2))

        try:
            Lp, lmbd, R = np.linalg.svd(s2, full_matrices=False)
        except np.linalg.LinAlgError as err:
            if "SVD did not converge" in str(err):
                Lp, lmbd, R = svd(s2, full_matrices=False, lapack_driver='gesvd')
                logging.warning("SVD convergence issue (used gesvd fallback).")
            else:
                raise

        chi12 = np.min([4 * chi1, 4 * chi2])
        chi12_p = np.min([self.chi, chi12])
        lmbd = np.diag(lmbd)

        lmbd = lmbd[:chi12_p, :chi12_p]
        Lp = Lp[:, :chi12_p]
        R = R[:chi12_p, :]

        if dirc == 'left':
            A1 = np.reshape(np.dot(Lp, lmbd), (chi1, 4, chi12_p))
            A2 = np.reshape(R, (chi12_p, 4, chi2))
            self.lmbd_position = ind[0]
        elif dirc == 'right':
            A1 = np.reshape(Lp, (chi1, 4, chi12_p))
            A2 = np.reshape(np.dot(lmbd, R), (chi12_p, 4, chi2))
            self.lmbd_position = ind[1]

        self.A_dict[f"A{ind[0]}"] = A1
        self.A_dict[f"A{ind[1]}"] = A2

    def move_lmbd_right(self, ind):
        I = np.reshape(np.eye(16), (4, 4, 4, 4))
        self.applyU([ind, ind + 1], 'right', I, 0, lm=True)

    def move_lmbd_left(self, ind):
        I = np.reshape(np.eye(16), (4, 4, 4, 4))
        self.applyU([ind, ind + 1], 'left', I, 0, lm=True)

    def lmbd_relocate(self, ind):
        step = ind - self.lmbd_position
        for _ in range(np.abs(step)):
            if step > 0:
                self.move_lmbd_right(self.lmbd_position)
            elif step < 0:
                self.move_lmbd_left(self.lmbd_position - 1)

    def sweepU(self):
        even_sites = [[i, i + 1] for i in np.arange(0, self.L - 1, 2)]
        odd_sites = [[i, i + 1] for i in np.arange(1, self.L - 1, 2)]
        odd_sites.reverse()

        for bond in even_sites:
            self.applyU(bond, 'right', self.U, self.sU)

        for bond in odd_sites:
            self.applyU(bond, 'left', self.U, self.sU)

        if sch_bool:
            self.measure_schrodinger()
        self.measure_TEBD()

    # --- measurements ---
    def measure_schrodinger(self):
        self.E_total_sch = 0
        c_ij = np.zeros(self.L - 1, dtype=np.complex128)
        for ind in range(self.L - 1):
            trunc_psi = np.reshape(self.schrodinger_psi, (2**ind, 4, 2**(self.L - 2 - ind)))
            c_ij[ind] = np.einsum(
                'aib,ij,ajb',
                np.conjugate(trunc_psi),
                non_int_fermi(self.J),
                trunc_psi,
                optimize='optimal'
            )
            self.E_persite_sch[ind] = c_ij[ind]
            self.E_total_sch += self.E_persite_sch[ind]

        self.E_fourier_sch = 0
        for i in range(self.L - 1):
            self.E_fourier_sch += np.e**(1j * (i + 1) * self.k) * self.E_persite_sch[i]
        self.E_fourier_sch = -(1 / self.L) * self.E_fourier_sch

        for i in range(self.L):
            trunc_psi = np.reshape(self.schrodinger_psi, (2**i, 2, 2**(self.L - 1 - i)))
            self.ni_persite_sch[i] = np.einsum(
                'aib,ij,ajb',
                np.conjugate(trunc_psi),
                n_op(),
                trunc_psi,
                optimize='optimal'
            )

        ninL_schop = n_op()
        for _ in range(1, self.L - 1):
            ninL_schop = np.kron(ninL_schop, np.eye(2))
        ninL_schop = np.kron(ninL_schop, n_op())

        ninl = np.dot(np.conjugate(self.schrodinger_psi), np.dot(ninL_schop, self.schrodinger_psi))
        self.ni_connect_sch = ninl - self.ni_persite_sch[0] * self.ni_persite_sch[self.L - 1]

    def build_left(self):
        temp = np.diag([1. + 0.j, 1. + 0.j])
        self.left_trace = [temp]
        for i in range(1, self.L):
            temp = np.tensordot(temp, self.A_dict[f"A{i-1}"][:, 0, :], axes=1)
            self.left_trace.append(temp)

    def build_right(self):
        temp = np.diag([1. + 0.j, 1. + 0.j])
        right = [temp]
        for i in np.arange(self.L - 2, -1, -1):
            temp = np.tensordot(self.A_dict[f"A{i+1}"][:, 0, :], temp, axes=1)
            right.append(temp)
        right.reverse()
        self.right_trace = right

    def tensordot_SxSy(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_SySx(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_SxSx(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_SySy(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_Sz(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * g * self.A_dict[f"A{ind}"][:, 3, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind], axes=1)
        return np.trace(temp)

    def tensordot_I(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, self.A_dict[f"A{ind}"][:, 0, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind], axes=1)
        return np.trace(temp)

    def measure_TEBD(self):
        self.build_left()
        self.build_right()

        # trace
        temp = self.A_dict["A0"][:, 0, :]
        for i in range(1, self.L):
            temp = np.tensordot(temp, self.A_dict[f"A{i}"][:, 0, :], axes=1)
        self.tr_TEBD = np.trace(temp)

        # energy per bond from Sx/Sy contractions
        SxSy = np.zeros(self.L - 1, dtype=np.complex128)
        SySx = np.zeros(self.L - 1, dtype=np.complex128)
        SxSx = np.zeros(self.L - 1, dtype=np.complex128)
        SySy = np.zeros(self.L - 1, dtype=np.complex128)
        for ind in range(self.L - 1):
            SxSy[ind] = self.tensordot_SxSy(ind)
            SySx[ind] = self.tensordot_SySx(ind)
            SxSx[ind] = self.tensordot_SxSx(ind)
            SySy[ind] = self.tensordot_SySy(ind)

        self.E_total_TEBD = 0
        for i in range(0, self.L - 1):
            cij_en = (1 / 4) * (SxSx[i] + 1j * SxSy[i] - 1j * SySx[i] + SySy[i])
            self.E_persite[i] = cij_en + np.conjugate(cij_en)
            self.E_total_TEBD += self.E_persite[i]

        self.E_fourier = 0
        for i in range(self.L - 1):
            self.E_fourier += np.e**(1j * (i + 1) * self.k) * self.E_persite[i]
        self.E_fourier = -(1 / self.L) * self.E_fourier

        for i in range(self.L):
            self.ni_persite[i] = (1 / 2) * (self.tensordot_I(i) - self.tensordot_Sz(i))

        # n1-nL connected (your form)
        temp_co = np.tensordot(
            self.left_trace[0],
            (1 / 2) * (self.A_dict["A0"][:, 0, :] - g * g * self.A_dict["A0"][:, 3, :]),
            axes=1
        )
        for i in range(1, self.L - 1):
            temp_co = np.tensordot(temp_co, self.A_dict[f"A{i}"][:, 0, :], axes=1)
        temp_co = np.tensordot(
            temp_co,
            (1 / 2) * (self.A_dict[f"A{self.L-1}"][:, 0, :] - g * g * self.A_dict[f"A{self.L-1}"][:, 3, :]),
            axes=1
        )
        self.ni_connect = np.trace(temp_co)
        self.ni_connect = self.ni_connect / self.tr_TEBD - (
            (self.ni_persite[0] * self.ni_persite[self.L - 1]) / (self.tr_TEBD * self.tr_TEBD)
        )


# -----------------------------
# CLI params + saving
# -----------------------------
@dataclass(frozen=True)
class RunParams:
    L: int = 64
    chi: int = 32
    T: float = 20.0
    N: int = 250
    g: float = 1.5
    outdir: str = "runs"
    tag: str = "cat"
    schrodinger_check: bool = False

def _git_commit_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

def _make_run_dir(base: Path, params: RunParams) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{params.tag}" if params.tag else ""
    name = f"{ts}{tag}_L{params.L}_chi{params.chi}_g{params.g}_T{params.T}_N{params.N}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def _setup_logging(run_dir: Path) -> None:
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

    lines = []
    if "Et_TEBD" in results:
        Et = np.asarray(results["Et_TEBD"])
        lines.append(f"Et_TEBD: shape={Et.shape}, real[min,max]=({np.real(Et).min():.6g},{np.real(Et).max():.6g})")
    if "tr_TB" in results:
        tr = np.asarray(results["tr_TB"])
        lines.append(f"trace: shape={tr.shape}, real[min,max]=({np.real(tr).min():.6g},{np.real(tr).max():.6g})")
    if lines:
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")

def parse_args(argv: list[str] | None = None) -> RunParams:
    p = argparse.ArgumentParser(description="MPDO fermion rTEBD (CLI + Jupyter-safe)")
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--chi", type=int, default=32)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--N", type=int, default=250)
    p.add_argument("--g", type=float, default=1.5)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="cat")
    p.add_argument("--schrodinger-check", action="store_true", help="Also evolve Schrödinger check (slow).")

    if argv is None:
        argv = sys.argv[1:]
    a, _unknown = p.parse_known_args(argv)  # <- key for Jupyter (ignores --f=...)
    return RunParams(
        L=a.L,
        chi=a.chi,
        T=a.T,
        N=a.N,
        g=a.g,
        outdir=a.outdir,
        tag=a.tag,
        schrodinger_check=a.schrodinger_check,
    )


# -----------------------------
# main entrypoint
# -----------------------------
def main(argv: list[str] | None = None) -> Path:
    """
    Run the simulation.

    In terminal: main(None) (uses sys.argv)
    In Jupyter:  main(["--L","8","--chi","16",...])

    Returns
    -------
    Path
        The run directory where results were saved.
    """
    params = parse_args(argv)

    # set globals (keeps your internal function style)
    global g, sch_bool, bound_diff, _renyi, _sqtrace
    g = float(params.g)
    sch_bool = bool(params.schrodinger_check)
    bound_diff = False
    _renyi = False
    _sqtrace = False

    base = Path(params.outdir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(base, params)
    _setup_logging(run_dir)

    logging.info("Starting run")
    logging.info("Run directory: %s", run_dir)
    logging.info("Params: %s", params)

    L = int(params.L)
    N = int(params.N)

    # allocate outputs (same shapes)
    E_psite = np.zeros((L - 1, N), dtype=np.complex128)
    ni_psite = np.zeros((L, N), dtype=np.complex128)
    ni_conn = []
    tr_TB = []
    Et_TEBD = []
    Ef_TEBD = []

    if sch_bool:
        E_psite_sch = np.zeros((L - 1, N), dtype=np.complex128)
        ni_psite_sch = np.zeros((L, N), dtype=np.complex128)
        Et_sch = []
        Ef_sch = []
        ni_conn_sch = []

    mps_evolve = MPDO(L, params.chi, params.T, params.N)

    t0 = time.time()
    for i in range(N):
        step_t0 = time.time()
        mps_evolve.sweepU()
        step_t1 = time.time()

        logging.info("Step %d/%d done in %.3fs", i + 1, N, step_t1 - step_t0)

        for j in range(L):
            ni_psite[j, i] = mps_evolve.ni_persite[j]
            if j != L - 1:
                E_psite[j, i] = mps_evolve.E_persite[j]

        Et_TEBD.append(mps_evolve.E_total_TEBD)
        ni_conn.append(mps_evolve.ni_connect)
        tr_TB.append(mps_evolve.tr_TEBD)
        Ef_TEBD.append(mps_evolve.E_fourier)

        if sch_bool:
            for j in range(L):
                ni_psite_sch[j, i] = mps_evolve.ni_persite_sch[j]
                if j != L - 1:
                    E_psite_sch[j, i] = mps_evolve.E_persite_sch[j]
            Et_sch.append(mps_evolve.E_total_sch)
            Ef_sch.append(mps_evolve.E_fourier_sch)
            ni_conn_sch.append(mps_evolve.ni_connect_sch)

    elapsed = time.time() - t0
    logging.info("Finished in %.2fs (%.3fs/step)", elapsed, elapsed / max(N, 1))

    results = {
        "E_psite": E_psite,
        "ni_psite": ni_psite,
        "Et_TEBD": np.asarray(Et_TEBD, dtype=np.complex128),
        "ni_conn": np.asarray(ni_conn, dtype=np.complex128),
        "tr_TB": np.asarray(tr_TB, dtype=np.complex128),
        "Ef_TEBD": np.asarray(Ef_TEBD, dtype=np.complex128),
    }

    if sch_bool:
        results.update({
            "E_psite_sch": E_psite_sch,
            "ni_psite_sch": ni_psite_sch,
            "Et_sch": np.asarray(Et_sch, dtype=np.complex128),
            "Ef_sch": np.asarray(Ef_sch, dtype=np.complex128),
            "ni_conn_sch": np.asarray(ni_conn_sch, dtype=np.complex128),
        })

    save_run(run_dir, params, results)
    logging.info("Saved results to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
