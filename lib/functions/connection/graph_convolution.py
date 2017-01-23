#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six

import numpy as np
import scipy.sparse

from chainer import function

class GraphConvolutionFunction(function.Function):

    """
    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.

    This link implements the graph convolution described in
    the following paper:

    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.

    """

    def __init__(self, L, K, use_cudnn=True):
        # NOTE(tommi): It is very important that L
        # is a normalized Graph Laplacian matrix.
        # Otherwise, this will not work.
        self.use_cudnn = use_cudnn
        L = scipy.sparse.csr_matrix(L)
        I = scipy.sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
        self.LmI = L - I
        self.K = K

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        # x.shape = (n_batch, c_in, N)
        n_batch, c_in, N = x.shape
        b = inputs[2] if len(inputs) == 3 else None

        K = self.K
        LmI = self.LmI

        C = np.empty((n_batch, K, N, c_in), dtype=x.dtype)
        C[:, 0] = x.transpose((0, 2, 1)) # (n_batch, N, c_in)
        # NOTE(tommi): Chainer does not support sparse tensordot, so have to use a for loop, although inefficient.
        if K > 1:
            for i in range(n_batch):
                C[i, 1] = LmI.dot(C[i, 0])
        for k in range(2, K):
            for i in range(n_batch):
                C[i, k] = 2 * LmI.dot(C[i, k-1]) - C[i, k-2]

        # C.shape = (n_batch, K, N, c_in)
        C = C.transpose((0,3,1,2))
        # C.shape = (n_batch, c_in, K, N)
        self.C = C
        # W.shape = (c_out, c_in, K)

        y = np.tensordot(C, W, ((1,2), (1,2)))
        # y.shape = (n_batch, N, c_out)

        if b is not None:
            y += b

        return np.rollaxis(y, 2, 1), # y.shape = (n_batch, c_out, N)

    #def forward_gpu(self, inputs):
    #    pass

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]

        n_batch, c_in, N = x.shape
        c_out = gy.shape[1]

        # gy.shape = (n_batch, c_out, N)
        # C.shape = (n_batch, c_in, K, N)
        gW = np.tensordot(gy, self.C, ((0,2), (0,3))).astype(W.dtype, copy=False)
        # gW.shape = (c_out, c_in, K)
        # y0.shape = (n_batch, N, c_out)

        K = self.K
        LmI = self.LmI

        C = np.empty((n_batch, K, N, c_out), dtype=x.dtype)
        C[:, 0] = gy.transpose((0, 2, 1)) # (n_batch, N, c_out)
        # NOTE(tommi): Chainer does not support sparse tensordot, so have to use a for loop, although inefficient.
        if K > 1:
            for i in range(n_batch):
                C[i, 1] = LmI.dot(C[i, 0])
        for k in range(2, K):
            for i in range(n_batch):
                C[i, k] = 2 * LmI.dot(C[i, k-1]) - C[i, k-2]

        # C.shape = (n_batch, K, N, c_out)
        C = C.transpose((0,3,1,2))
        # C.shape = (n_batch, c_out, K, N)
        # W.shape = (c_out, c_in, K)

        gx = np.tensordot(C, W, ((1,2), (0,2)))
        # gx.shape = (n_batch, N, c_in)
        gx = np.rollaxis(gx, 2, 1)
        # gx.shape = (n_batch, c_in, N)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0,2))
            # gb.shape = (c_out,)
            return gx, gW, gb


    #def backward_gpu(self, inputs, grad_outputs):
    #    pass

def graph_convolution(x, W, L, K, b=None, use_cudnn=True):
    """
    Graph convolution function.

    This is an implementation of graph convolution.
    """
    func = GraphConvolutionFunction(L, K, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
