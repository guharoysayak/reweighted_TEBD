from __future__ import annotations

import numpy as np
from scipy.linalg import svd

from ..utils import dagger
from .model import gate_two_site, init_MPDO_dict_fermi, init_fermi_psi_sch, non_int_fermi, n_op

class MPDO:
    """MPDO representation and rTEBD evolution for the fermion model.

    This is a refactor of your standalone script's MPDO class:
    - no globals (g/chi/sch flags are instance attributes)
    - same contraction formulas and measurement outputs
    - same sweep order (even bonds then reversed odd bonds)
    """

    def __init__(
        self,
        L: int,
        chi: int,
        T: float,
        N: int,
        g: float,
        *,
        J: float = 1.0,
        k: float = np.pi / 4,
        schrodinger_check: bool = False,
        renyi_cuts: bool = False,
        sqtrace: bool = False,
    ):
        self.L = int(L)
        self.chi = int(chi)
        self.T = float(T)
        self.N = int(N)
        self.dt = self.T / self.N
        self.g = float(g)

        self.J = float(J)
        self.k = float(k)

        self.sch_bool = bool(schrodinger_check)
        self._renyi = bool(renyi_cuts)
        self._sqtrace = bool(sqtrace)

        # gates
        self.U, self.sU = gate_two_site(self.dt, J=self.J, g=self.g)

        # MPDO tensors
        self.A_dict = init_MPDO_dict_fermi(self.L, g=self.g)
        self.lmbd_position = 0

        # optional Schrödinger ref
        if self.sch_bool:
            self.schrodinger_psi = init_fermi_psi_sch(self.L)
            self.E_persite_sch = np.zeros(self.L - 1, dtype=np.complex128)
            self.ni_persite_sch = np.zeros(self.L, dtype=np.complex128)
            self.E_total_sch = 0
            self.renyi_test_left = 0
            self.renyi_test_right = 0
            self.E_fourier_sch = 0

        # measurement holders (match script names)
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
        self.ck_correl = 0
        self.Gij = np.zeros(self.L, dtype=np.complex128)

        self.left_trace: list[np.ndarray] = []
        self.right_trace: list[np.ndarray] = []

    # Schrödinger evolution (debug)
    def applyU_schrodinger(self, ind: int, sU: np.ndarray) -> None:
        self.schrodinger_psi = self.schrodinger_psi.reshape((2**ind, 4, 2 ** (self.L - 2 - ind)))
        self.schrodinger_psi = np.einsum("ij, ajb -> aib", sU, self.schrodinger_psi, optimize="optimal")
        self.schrodinger_psi = self.schrodinger_psi.flatten()

    def applyU(self, ind: list[int], dirc: str, U: np.ndarray, sU: np.ndarray | int, lm: bool = False) -> None:
        # relocate lambda
        if not lm:
            if dirc == "left":
                self.lmbd_relocate(ind[1])
            elif dirc == "right":
                self.lmbd_relocate(ind[0])
            else:
                raise ValueError("dirc must be 'left' or 'right'")

        # Schr check
        if (not lm) and self.sch_bool:
            self.applyU_schrodinger(ind[0], sU)

        A1 = self.A_dict[f"A{ind[0]}"]
        A2 = self.A_dict[f"A{ind[1]}"]
        chi1 = A1.shape[0]
        chi2 = A2.shape[2]

        s1 = np.einsum("ijkl,akb,blc->aijc", U, A1, A2, optimize="optimal")
        s2 = s1.reshape((4 * chi1, 4 * chi2))

        try:
            Lp, lmbd, R = np.linalg.svd(s2, full_matrices=False)
        except np.linalg.LinAlgError as err:
            if "SVD did not converge" in str(err):
                Lp, lmbd, R = svd(s2, full_matrices=False, lapack_driver="gesvd")
            else:
                raise

        chi12 = min(4 * chi1, 4 * chi2)
        chi12_p = min(self.chi, chi12)

        lmbd = np.diag(lmbd)[:chi12_p, :chi12_p]
        Lp = Lp[:, :chi12_p]
        R = R[:chi12_p, :]

        if dirc == "left":
            A1 = (Lp @ lmbd).reshape((chi1, 4, chi12_p))
            A2 = R.reshape((chi12_p, 4, chi2))
            self.lmbd_position = ind[0]
        else:  # right
            A1 = Lp.reshape((chi1, 4, chi12_p))
            A2 = (lmbd @ R).reshape((chi12_p, 4, chi2))
            self.lmbd_position = ind[1]

        self.A_dict[f"A{ind[0]}"] = A1
        self.A_dict[f"A{ind[1]}"] = A2

    def move_lmbd_right(self, ind: int) -> None:
        I = np.eye(16, dtype=np.complex128).reshape((4, 4, 4, 4))
        self.applyU([ind, ind + 1], "right", I, 0, lm=True)

    def move_lmbd_left(self, ind: int) -> None:
        I = np.eye(16, dtype=np.complex128).reshape((4, 4, 4, 4))
        self.applyU([ind, ind + 1], "left", I, 0, lm=True)

    def lmbd_relocate(self, ind: int) -> None:
        step = ind - self.lmbd_position
        for _ in range(abs(step)):
            if step > 0:
                self.move_lmbd_right(self.lmbd_position)
            elif step < 0:
                self.move_lmbd_left(self.lmbd_position - 1)

    def sweepU(self) -> None:
        even_sites = [[i, i + 1] for i in np.arange(0, self.L - 1, 2)]
        odd_sites = [[i, i + 1] for i in np.arange(1, self.L - 1, 2)]
        odd_sites.reverse()

        for pair in even_sites:
            self.applyU(pair, "right", self.U, self.sU)

        for pair in odd_sites:
            self.applyU(pair, "left", self.U, self.sU)

        if self.sch_bool:
            self.measure_schrodinger()
        self.measure_TEBD()
        self.measure_correlations()

    # --- measurements ---
    def measure_schrodinger(self) -> None:
        self.E_total_sch = 0
        c_ij = np.zeros(self.L - 1, dtype=np.complex128)
        for ind in range(self.L - 1):
            trunc_psi = self.schrodinger_psi.reshape((2**ind, 4, 2 ** (self.L - 2 - ind)))
            c_ij[ind] = np.einsum(
                "aib,ij,ajb",
                np.conjugate(trunc_psi),
                non_int_fermi(self.J),
                trunc_psi,
                optimize="optimal",
            )
            self.E_persite_sch[ind] = c_ij[ind]
            self.E_total_sch += self.E_persite_sch[ind]

        self.E_fourier_sch = 0
        for i in range(self.L - 1):
            self.E_fourier_sch += np.e ** (1j * (i + 1) * self.k) * self.E_persite_sch[i]
        self.E_fourier_sch = -(1 / self.L) * self.E_fourier_sch

        for i in range(self.L):
            trunc_psi = self.schrodinger_psi.reshape((2**i, 2, 2 ** (self.L - 1 - i)))
            self.ni_persite_sch[i] = np.einsum(
                "aib,ij,ajb",
                np.conjugate(trunc_psi),
                n_op(),
                trunc_psi,
                optimize="optimal",
            )

        psi_mat = self.schrodinger_psi.reshape((int(2 ** (self.L / 2)), int(2 ** (self.L / 2))))
        rho_left = dagger(psi_mat) @ psi_mat
        rho_right = psi_mat @ dagger(psi_mat)
        tr_left_test = np.trace(rho_left @ rho_left)
        tr_right_test = np.trace(rho_right @ rho_right)
        total_trace = np.trace(rho_left)
        self.renyi_test_left = -np.log2(tr_left_test / (total_trace**2))
        self.renyi_test_right = -np.log2(tr_right_test / (total_trace**2))

    def build_left(self) -> None:
        self.left_trace = []
        temp = np.reshape(1.0 + 0.0j, (1, 1))
        self.left_trace.append(temp)
        for i in range(1, self.L):
            temp = np.tensordot(temp, self.A_dict[f"A{i-1}"][:, 0, :], axes=1)
            self.left_trace.append(temp)

    def build_right(self) -> None:
        self.right_trace = []
        temp = np.reshape(1.0 + 0.0j, (1, 1))
        self.right_trace.append(temp)
        for i in range(self.L - 2, -1, -1):
            temp = np.tensordot(self.A_dict[f"A{i+1}"][:, 0, :], temp, axes=1)
            self.right_trace.append(temp)
        self.right_trace.reverse()

    def tensordot_SxSy(self, ind: int) -> complex:
        g = self.g
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_SySx(self, ind: int) -> complex:
        g = self.g
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_SxSx(self, ind: int) -> complex:
        g = self.g
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 1, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_SySy(self, ind: int) -> complex:
        g = self.g
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, g * self.A_dict[f"A{ind+1}"][:, 2, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind + 1], axes=1)
        return temp.flatten()[0]

    def tensordot_Sz(self, ind: int) -> complex:
        g = self.g
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, (g * g) * self.A_dict[f"A{ind}"][:, 3, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind], axes=1)
        return temp.flatten()[0]

    def tensordot_I(self, ind: int) -> complex:
        temp = self.left_trace[ind]
        temp = np.tensordot(temp, self.A_dict[f"A{ind}"][:, 0, :], axes=1)
        temp = np.tensordot(temp, self.right_trace[ind], axes=1)
        return temp.flatten()[0]

    def measure_TEBD(self) -> None:
        g = self.g
        self.build_left()
        self.build_right()

        # trace
        temp = self.A_dict["A0"][:, 0, :]
        for i in range(1, self.L):
            temp = np.tensordot(temp, self.A_dict[f"A{i}"][:, 0, :], axes=1)
        self.tr_TEBD = temp.flatten()[0]

        # squared trace / full Renyi
        if self._sqtrace:
            temp1 = np.einsum("iaj,kal->ikjl", self.A_dict["A0"][:, :1, :], self.A_dict["A0"][:, :1, :], optimize="optimal")
            temp2 = np.einsum("iaj,kal->ikjl", g * self.A_dict["A0"][:, 1:, :], g * self.A_dict["A0"][:, 1:, :], optimize="optimal")
            temp = temp1 + temp2
            for i in range(1, self.L):
                temp = np.einsum("ikjl,jbm->ikmbl", temp, self.A_dict[f"A{i}"], optimize="optimal")
                temp1 = np.einsum("ikmbl,lbn->ikmn", temp[:, :, :, :1, :], self.A_dict[f"A{i}"][:, :1, :], optimize="optimal")
                temp2 = np.einsum("ikmbl,lbn->ikmn", g * temp[:, :, :, 1:, :], g * self.A_dict[f"A{i}"][:, 1:, :], optimize="optimal")
                temp = temp1 + temp2
            self.sq_trace = (1 / 2**self.L) * temp.flatten()[0]
            self.norm_trace = self.tr_TEBD / np.sqrt(self.sq_trace)
            self.renyi_full = -np.log2(self.sq_trace / (self.tr_TEBD) ** 2)

        # Renyi cuts
        if self._renyi:
            # left cut
            temp1 = np.einsum("iaj,kal->ikjl", self.A_dict["A0"][:, :1, :], self.A_dict["A0"][:, :1, :], optimize="optimal")
            temp2 = np.einsum("iaj,kal->ikjl", g * self.A_dict["A0"][:, 1:, :], g * self.A_dict["A0"][:, 1:, :], optimize="optimal")
            temp = temp1 + temp2
            for i in range(1, int(self.L / 2)):
                temp = np.einsum("ikjl,jbm->ikmbl", temp, self.A_dict[f"A{i}"], optimize="optimal")
                temp1 = np.einsum("ikmbl,lbn->ikmn", temp[:, :, :, :1, :], self.A_dict[f"A{i}"][:, :1, :], optimize="optimal")
                temp2 = np.einsum("ikmbl,lbn->ikmn", g * temp[:, :, :, 1:, :], g * self.A_dict[f"A{i}"][:, 1:, :], optimize="optimal")
                temp = temp1 + temp2
            for i in range(int(self.L / 2), self.L):
                temp1 = np.einsum("ikjl,jm->ikml", temp, self.A_dict[f"A{i}"][:, 0, :], optimize="optimal")
                temp = np.einsum("ikml,ln->ikmn", temp1, self.A_dict[f"A{i}"][:, 0, :], optimize="optimal")
            left_trace_sq = (1 / 2 ** (self.L / 2)) * temp.flatten()[0]
            self.renyi_left = -np.log2(left_trace_sq / (self.tr_TEBD**2))

            # right cut
            temp1 = np.einsum(
                "iaj,kal->ikjl",
                self.A_dict[f"A{self.L-1}"][:, :1, :],
                self.A_dict[f"A{self.L-1}"][:, :1, :],
                optimize="optimal",
            )
            temp2 = np.einsum(
                "iaj,kal->ikjl",
                g * self.A_dict[f"A{self.L-1}"][:, 1:, :],
                g * self.A_dict[f"A{self.L-1}"][:, 1:, :],
                optimize="optimal",
            )
            temp = temp1 + temp2
            for i in np.arange(self.L - 2, (int(self.L / 2)) - 1, -1):
                temp = np.einsum("mbi,ikjl->mbkjl", self.A_dict[f"A{i}"], temp, optimize="optimal")
                temp1 = np.einsum("nbk,mbkjl->mnjl", self.A_dict[f"A{i}"][:, :1, :], temp[:, :1, :, :, :], optimize="optimal")
                temp2 = np.einsum("nbk,mbkjl->mnjl", g * self.A_dict[f"A{i}"][:, 1:, :], g * temp[:, 1:, :, :, :], optimize="optimal")
                temp = temp1 + temp2
            for i in np.arange((int(self.L / 2)) - 1, -1, -1):
                temp1 = np.einsum("ai,ikjl->akjl", self.A_dict[f"A{i}"][:, 0, :], temp, optimize="optimal")
                temp = np.einsum("bk,akjl->abjl", self.A_dict[f"A{i}"][:, 0, :], temp1, optimize="optimal")
            right_trace_sq = (1 / 2 ** (self.L / 2)) * temp.flatten()[0]
            self.renyi_right = -np.log2(right_trace_sq / (self.tr_TEBD**2))

        # energy per site
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
        for i in range(self.L - 1):
            cij_en = (1 / 4) * (SxSx[i] + 1j * SxSy[i] - 1j * SySx[i] + SySy[i])
            self.E_persite[i] = cij_en + np.conjugate(cij_en)
            self.E_total_TEBD += self.E_persite[i]

        # Fourier energy component
        self.E_fourier = 0
        for i in range(self.L - 1):
            self.E_fourier += np.e ** (1j * (i + 1) * self.k) * self.E_persite[i]
        self.E_fourier = -(1 / self.L) * self.E_fourier

        # densities
        for i in range(self.L):
            self.ni_persite[i] = (1 / 2) * (self.tensordot_I(i) - self.tensordot_Sz(i))

        # end-to-end connected density correlator
        temp_co = np.tensordot(self.left_trace[0], (1 / 2) * (self.A_dict["A0"][:, 0, :] - (g * g) * self.A_dict["A0"][:, 3, :]), axes=1)
        for i in range(1, self.L - 1):
            temp_co = np.tensordot(temp_co, self.A_dict[f"A{i}"][:, 0, :], axes=1)
        temp_co = np.tensordot(
            temp_co,
            (1 / 2) * (self.A_dict[f"A{self.L-1}"][:, 0, :] - (g * g) * self.A_dict[f"A{self.L-1}"][:, 3, :]),
            axes=1,
        )
        self.ni_connect = temp_co.flatten()[0] - self.ni_persite[0] * self.ni_persite[self.L - 1]

    def measure_correlations(self) -> None:
        g = self.g
        self.ccdagger = np.zeros((self.L, self.L), dtype=np.complex128)
        self.cdaggerc = np.zeros((self.L, self.L), dtype=np.complex128)

        for i in range(self.L):
            self.cdaggerc[i, i] = (1 / 2) * (self.tensordot_I(i) - self.tensordot_Sz(i))
            self.ccdagger[i, i] = (1 / 2) * (self.tensordot_I(i) - self.tensordot_Sz(i))

            b_temp = np.tensordot(
                self.left_trace[i],
                (1 / 2) * (g * self.A_dict[f"A{i}"][:, 1, :] + 1j * g * self.A_dict[f"A{i}"][:, 2, :]),
                axes=1,
            )
            d_temp = np.tensordot(
                self.left_trace[i],
                (1 / 2) * (g * self.A_dict[f"A{i}"][:, 1, :] - 1j * g * self.A_dict[f"A{i}"][:, 2, :]),
                axes=1,
            )

            for j in range(i + 1, self.L):
                bt1 = np.tensordot(
                    b_temp,
                    (1 / 2) * (g * self.A_dict[f"A{j}"][:, 1, :] - 1j * g * self.A_dict[f"A{j}"][:, 2, :]),
                    axes=1,
                )
                self.ccdagger[i, j] = -1 * np.tensordot(bt1, self.right_trace[j], axes=1).flatten()[0]
                b_temp = np.tensordot(b_temp, (g * g) * self.A_dict[f"A{j}"][:, 3, :], axes=1)

                dt1 = np.tensordot(
                    d_temp,
                    (1 / 2) * (g * self.A_dict[f"A{j}"][:, 1, :] + 1j * g * self.A_dict[f"A{j}"][:, 2, :]),
                    axes=1,
                )
                self.cdaggerc[i, j] = np.tensordot(dt1, self.right_trace[j], axes=1).flatten()[0]
                d_temp = np.tensordot(d_temp, (g * g) * self.A_dict[f"A{j}"][:, 3, :], axes=1)

            for j in np.flip(np.arange(0, i, 1)):
                self.ccdagger[i, j] = -1 * self.cdaggerc[j, i]
                self.cdaggerc[i, j] = -1 * self.ccdagger[j, i]

        self.ck_correl = 0
        for i in range(self.L):
            for j in range(self.L):
                self.ck_correl += np.e ** (-1j * self.k * (i - j)) * self.cdaggerc[i, j]

    def measure_Gij(self) -> None:
        g = self.g
        self.Gij = []
        i0 = int(self.L / 4)
        d_temp = np.tensordot(
            self.left_trace[i0],
            (1 / 2) * (g * self.A_dict[f"A{i0}"][:, 1, :] - 1j * g * self.A_dict[f"A{i0}"][:, 2, :]),
            axes=1,
        )
        for j in range(i0 + 1, self.L):
            dt1 = np.tensordot(
                d_temp,
                (1 / 2) * (g * self.A_dict[f"A{j}"][:, 1, :] + 1j * g * self.A_dict[f"A{j}"][:, 2, :]),
                axes=1,
            )
            self.Gij.append(np.tensordot(dt1, self.right_trace[j], axes=1).flatten()[0])
            d_temp = np.tensordot(d_temp, (g * g) * self.A_dict[f"A{j}"][:, 3, :], axes=1)
