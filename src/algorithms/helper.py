# helper functions for ETNA algorithm

import numpy as np
import scipy.sparse as sparse
import theano
import torch
from scipy.sparse import csgraph
from theano import tensor as T


def pairwise_cosimilarity(a, b, eps=1e-8):
    '''Calculate pairwise cosine similiarty between two matrices

    Parameters
    ----------
    a : torch.Tensor
        input matrix 1
        Shape `(input_a_size, embed_dim)`

    b : torch.Tensor
        input matrix 2
        Shape `(input_b_size, embed_dim)`

    eps : float
        small value to avoid division by zero.

    Returns
    -------
    sim_mt : torch.Tensor
        pairwise cosimilarity matrix
        Shape `(input_a_size, input_b_size)`
    '''
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def direct_compute_deepwalk_matrix(A, window, b=1):
    '''Calculate deepwalk matrix for give adjacency matrix
    https://github.com/xptree/NetMF/blob/master/netmf.py

    Parameters
    ----------
    A : scipy.sparse.csr.csr_matrix
        adjacency matrix

    window : int
        window size

    b : int
        the number of negative samplin
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
