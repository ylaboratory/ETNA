# Paper: Functional protein representations from biological networks enable diverse cross-species inference
# The original code can be found in: https://github.com/lrgr/munk

import numpy as np
import networkx as nx
import func

def regularized_laplacian(nx_g, lam):
    L = np.array(nx.laplacian_matrix(nx_g).todense())
    D = np.linalg.inv(np.eye(len(L)) + lam * L)
    return D

def rkhs_factor(D):
    e, v = np.linalg.eigh(D)
    C = v.dot(np.diag(np.sqrt(e)))
    return C

def proj(C1, D2, anchor1, anchor2):
    C2_hat = np.linalg.pinv(C1[anchor1,:]).dot(D2[anchor2,:]).T
    return C2_hat

def munk(g1, g2, lam1, lam2, anchor):
    D1 = regularized_laplacian(g1, lam1)
    D2 = regularized_laplacian(g2, lam2)
    C1 = rkhs_factor(D1)
    C2 = rkhs_factor(D2)
    #anchor1, anchor2 = func.anchor_idx(anchor, g1, g2)
    anchor1 = [x[0] for x in anchor]
    anchor2 = [x[1] for x in anchor]
    C2_hat = proj(C1, D2, anchor1, anchor2)
    D12 = C1.dot(C2_hat.T)
    return C1, C2, C2_hat, D12
