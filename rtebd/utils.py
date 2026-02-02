from __future__ import annotations

import numpy as np

def dagger(M: np.ndarray) -> np.ndarray:
    """Conjugate transpose (Hermitian adjoint)."""
    return np.conjugate(np.transpose(M))

def mat_dot2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(A, B)

def mat_dot4(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.dot(A, np.dot(B, np.dot(C, D)))

def s_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

def s_y() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)

def s_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
