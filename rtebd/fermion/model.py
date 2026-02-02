from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..utils import dagger, mat_dot4, s_x, s_y, s_z

def pauli_tilde(g: float) -> list[np.ndarray]:
    """Reweighted Pauli basis {I, gX, gY, g^2 Z}."""
    return [np.eye(2, dtype=np.complex128), g * s_x(), g * s_y(), (g * g) * s_z()]

def pauli_bar(g: float) -> list[np.ndarray]:
    """Inverse-reweighted Pauli basis {I, X/g, Y/g, Z/g^2}."""
    return [np.eye(2, dtype=np.complex128), (1 / g) * s_x(), (1 / g) * s_y(), (1 / (g * g)) * s_z()]

def U_mat(U: np.ndarray, g: float) -> np.ndarray:
    """Convert a two-site unitary U (4x4) into its 4x4x4x4 superoperator tensor."""
    Ud = dagger(U)
    U_all = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    pbar = pauli_bar(g)
    ptil = pauli_tilde(g)
    for i in range(4):
        for j in range(4):
            sg1 = np.kron(pbar[i], pbar[j])
            for k in range(4):
                for l in range(4):
                    sg2 = np.kron(ptil[k], ptil[l])
                    U_all[i, j, k, l] = (1 / 4) * np.trace(mat_dot4(sg1, U, sg2, Ud))
    return U_all

def non_int_fermi(J: float) -> np.ndarray:
    """Two-site Hamiltonian for the non-interacting fermion model."""
    return J * np.array(
        [[0, 0, 0, 0],
         [0, 0, -1, 0],
         [0, -1, 0, 0],
         [0, 0, 0, 0]],
        dtype=np.complex128,
    )

def gate_two_site(dt: float, J: float, g: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (superoperator U, physical unitary sU) for a timestep dt."""
    Hi = non_int_fermi(J)
    sU = expm(-1j * dt * Hi)
    U = U_mat(sU, g=g)
    return U, sU


def init_fermi_psi_local(L: int) -> np.ndarray:
    """Local 2-component states used to build the MPDO (matches your 8-site pattern)."""
    t1 = np.array([1, 0], dtype=np.complex128)
    t2 = np.array([0, 1], dtype=np.complex128)
    psi = np.zeros((L, 2), dtype=np.complex128)
    for i in range(L):
        if (i + 1) % 8 in [1, 2, 7, 0]:
            psi[i] = t2
        else:
            psi[i] = t1
    return psi

def init_fermi_psi_sch(L: int) -> np.ndarray:
    """
    Schrödinger-picture product state constructed from the same
    local pattern as init_fermi_psi_local.
    """
    psi_local = init_fermi_psi_local(L)

    psi = np.array([1.0 + 0j], dtype=np.complex128)
    for i in range(L):
        psi = np.kron(psi, psi_local[i])

    return psi


def init_MPDO_dict_fermi(L: int, g: float) -> dict[str, np.ndarray]:
    """Initial MPDO tensors A_i with shape (1,4,1) in Pauli transfer form."""
    psi = init_fermi_psi_local(L)
    pbar = pauli_bar(g)
    A_dict: dict[str, np.ndarray] = {}
    for i in range(L):
        rho = np.outer(psi[i], np.conjugate(psi[i]))
        A_temp = np.zeros((4,), dtype=np.complex128)
        for j in range(4):
            A_temp[j] = np.trace(pbar[j] @ rho)
        A_dict[f"A{i}"] = A_temp.reshape((1, 4, 1))
    return A_dict

def n_op() -> np.ndarray:
    """Local number operator n = |1><1|."""
    return np.array([[0, 0], [0, 1]], dtype=np.complex128)
