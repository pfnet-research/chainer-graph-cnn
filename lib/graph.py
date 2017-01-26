#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.sparse as ss
import numpy as np
import scipy.spatial.distance
import sklearn.metrics.pairwise


def create_laplacian(W, normalize=True, minus_identity=False):
    n = W.shape[0]
    W = ss.csr_matrix(W)
    WW_diag = W.dot(ss.csr_matrix(np.ones((n, 1)))).todense()
    if normalize:
        WWds = np.sqrt(WW_diag)
        WWds[WWds == 0] = np.float("inf")  # Let the inverse of zero entries become zero.
        WW_diag_invroot = 1. / WWds
        D_invroot = ss.lil_matrix((n, n))
        D_invroot.setdiag(WW_diag_invroot)
        D_invroot = ss.csr_matrix(D_invroot)
        L = D_invroot.dot(W.dot(D_invroot))
        if minus_identity:
            L = -L
        else:
            I = scipy.sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
            L = I - L
    else:
        D = ss.lil_matrix((n, n))
        D.setdiag(WW_diag)
        D = ss.csr_matrix(D)
        L = D - W
        if minus_identity:
            I = scipy.sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
            L -= I

    return L.astype(W.dtype)


def grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    return d, idx


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def grid_graph(m):
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    z = grid(m)
    dist, idx = distance_sklearn_metrics(z, k=8)
    A = adjacency(dist, idx)
    return A

