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
            eigvec[:, j].conj() @ O1 @ eigvec[:, 0]) / (-E0 + Ej + omega + 0.1j)
    return G


def make_BIS(eigval, eigvec, O0, O1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (eigvec[:, 0].conj() @ O0 @ eigvec[:, j]) * (
            eigvec[:, j].conj() @ O1 @ eigvec[:, 0]) / (E0 - Ej + omega + 0.1j)
    return G


def projection(h, n):
    from itertools import product
    index = [i for i, state in enumerate(
        product([0, 1], repeat=4)) if sum(state) == n]
    return h[index, :][:, index]


def make_Gkomega(Gijs, ks):
    omegamesh = Gijs.shape[0]
    L = Gijs.shape[1]
    Gkomegas = []
    for iomega in range(omegamesh):
        Gks = []
        for k in ks:
            Gk = 0+0j
            for ix in range(L):
                Gk += np.exp(-1.0j*k*ix)*Gijs[iomega][ix]
            Gks.append(Gk)
        Gkomegas.append(Gks)
    Gkomegas = np.array(Gkomegas)
    Gkomegas = Gkomegas/np.sqrt(L)
    return Gkomegas
