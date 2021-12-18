import theano
import torch
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csgraph
from theano import tensor as T


def pairwise_cosimilarity(a, b, eps=1e-8):
    '''
    calculate pairwise cosine similiarty between two matrices
    '''
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def direct_compute_deepwalk_matrix(A, window, b=1):
    '''
    calculate deepwalk matrix for give adjacency matrix
    '''
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L  # D^(-1/2) @ A @ D^(-1/2)
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)
