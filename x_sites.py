import numpy as np


def make_local_ops():
    ops = {}
    cdag = np.zeros((2, 2))
    cdag[1, 0] = 1
    ops['c^+'] = cdag
    ops['c^-'] = cdag.transpose()
    ops['I'] = np.identity(2)
    ops['F'] = np.diag((1.0, -1.0))
    return ops


def make_matrix(ss):
    res = np.array([1])
    for s in reversed(ss):
        res = np.kron(res, s)
    return res


def make_G(eigval, eigvec, O0, O1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (eigvec[:, 0].conj() @ O0 @ eigvec[:, j]) * (
            eigvec[:, j].conj() @ O1 @ eigvec[:, 0]) / (E0 - Ej + omega + 0.1j)
    return G


def make_PES(eigval, eigvec, O0, O1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (eigvec[:, 0].conj() @ O0 @ eigvec[:, j]) * (
            eigvec[:, j].conj() @ O1 @ eigvec[:, 0]) / (E0 - Ej + omega + 0.1j)
    return G


def make_BIS(eigval, eigvec, O0, O1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (eigvec[:, 0].conj() @ O0 @ eigvec[:, j]) * (
            eigvec[:, j].conj() @ O1 @ eigvec[:, 0]) / (-E0 + Ej + omega + 0.1j)
    return G


def make_Gij(eigval, eigvec, L, i, j, omega):
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    szi = [s0] * i + [sz] + [s0] * (L - i - 1)
    szj = [s0] * j + [sz] + [s0] * (L - j - 1)
    Si = make_matrix(szi)
    Sj = make_matrix(szj)
    G = make_dynamical_structure_factor(eigval, eigvec, Sj, Si, omega)
    return G
