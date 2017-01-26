#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six

import numpy as np
import scipy.sparse

import chainer
from chainer import cuda
from chainer import function
from chainer.cuda import cupy


def chebyshev_matvec_cpu(C, x, K, n_batch, LmI):
    C[:, 0] = x.transpose((0, 2, 1)) # (n_batch, N, c_in)
    # NOTE(tommi): Chainer does not support sparse tensordot, so have to use a for loop, although inefficient.
    if K > 1:
        for i in range(n_batch):
            C[i, 1] = LmI.dot(C[i, 0])
    for k in range(2, K):
        for i in range(n_batch):
            C[i, k] = 2 * LmI.dot(C[i, k-1]) - C[i, k-2]

if chainer.cuda.available:
    # Computes y = Lx
    # x will be flattened in C-order
    # y will be flattened in C-order
    csr_matvec = cupy.ElementwiseKernel(
            'I p, raw T data, raw I indices, raw I indptr, raw T x',
            'T y',
            '''
            y = 0;
            int n_cols = _ind.size() / p;
            int row_idx = i / n_cols;
            int col_idx = i % n_cols;
            for(I j = indptr[row_idx]; j < indptr[(row_idx+1)]; j++) {
                y += data[j] * x[indices[j] * n_cols + col_idx];
            }
            ''',
            'csr_matvec'
            )

    def chebyshev_matvec_gpu(C, x, K, n_batch, LmI_data, LmI_indices, LmI_indptr):
        # C.shape = (K, N, c_in, n_batch)
        C[0] = x.transpose((2, 1, 0)) # (N, c_in, n_batch)
        N = C.shape[1]
        if K > 1:
                csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[0], C[1])
        for k in range(2, K):
            csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[k-1], C[k])
            C[k] = 2 * C[k] - C[k-2]
        #if K > 1:
        #    for i in range(n_batch):
        #        # C[i, 1] = LmI.dot(C[i, 0])
        #        csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[i, 0, :], C[i, 1, :])
        #for k in range(2, K):
        #    for i in range(n_batch):
        #        # C[i, k] = 2 * LmI.dot(C[i, k-1]) - C[i, k-2]
        #        # C[i, k].shape = (N, c_in)
        #        csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[i, k-1, :], C[i, k, :])
        #        C[i, k, :] = 2 * C[i, k, :] - C[i, k-2, :]

class GraphConvolutionFunction(function.Function):

    """
    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.

    This link implements the graph convolution described in
    the following paper:

    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.

    """

    def __init__(self, n_verts, LmI_data, LmI_indices, LmI_indptr, K):
        # NOTE(tommi): It is very important that L
        # is a normalized Graph Laplacian matrix.
        # Otherwise, this will not work.

        # It is assumed here that the diagonal entries of L has
        # already been set to zero.
        self.LmI_tuple = (LmI_data, LmI_indices, LmI_indptr)
        self.LmI_shape = (n_verts, n_verts)

        self.K = K

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        # x.shape = (n_batch, c_in, N)
        n_batch, c_in, N = x.shape
        b = inputs[2] if len(inputs) == 3 else None

        K = self.K
        LmI = scipy.sparse.csr_matrix(self.LmI_tuple, self.LmI_shape)

        C = np.empty((n_batch, K, N, c_in), dtype=x.dtype)
        chebyshev_matvec_cpu(C, x, K, n_batch, LmI)

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

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        # x.shape = (n_batch, c_in, N)
        n_batch, c_in, N = x.shape
        b = inputs[2] if len(inputs) == 3 else None
        xp = cuda.get_array_module(x)

        K = self.K
        LmI_data, LmI_indices, LmI_indptr = self.LmI_tuple

        C = xp.empty((K, N, c_in, n_batch), dtype=x.dtype)
        chebyshev_matvec_gpu(C, x, K, n_batch, LmI_data, LmI_indices, LmI_indptr)

        # C.shape = (K, N, c_in, n_batch)
        C = C.transpose((3,2,0,1))
        # C.shape = (n_batch, c_in, K, N)
        self.C = C
        # W.shape = (c_out, c_in, K)

        y = xp.tensordot(C, W, ((1,2), (1,2)))
        # y.shape = (n_batch, N, c_out)

        if b is not None:
            y += b

        return xp.rollaxis(y, 2, 1), # y.shape = (n_batch, c_out, N)

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
        LmI = scipy.sparse.csr_matrix(self.LmI_tuple, self.LmI_shape)

        C = np.empty((n_batch, K, N, c_out), dtype=x.dtype)
        chebyshev_matvec_cpu(C, gy, K, n_batch, LmI)

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


    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        xp = cuda.get_array_module(x)

        n_batch, c_in, N = x.shape
        c_out = gy.shape[1]

        # gy.shape = (n_batch, c_out, N)
        # C.shape = (n_batch, c_in, K, N)
        gW = xp.tensordot(gy, self.C, ((0,2), (0,3))).astype(W.dtype, copy=False)
        # gW.shape = (c_out, c_in, K)
        # y0.shape = (n_batch, N, c_out)

        K = self.K
        LmI_data, LmI_indices, LmI_indptr = self.LmI_tuple

        C = xp.empty((K, N, c_out, n_batch), dtype=x.dtype)
        chebyshev_matvec_gpu(C, gy, K, n_batch, LmI_data, LmI_indices, LmI_indptr)

        # C.shape = (K, N, c_out, n_batch)
        C = C.transpose((3,2,0,1))
        # C.shape = (n_batch, c_out, K, N)
        # W.shape = (c_out, c_in, K)

        gx = xp.tensordot(C, W, ((1,2), (0,2)))
        # gx.shape = (n_batch, N, c_in)
        gx = xp.rollaxis(gx, 2, 1)
        # gx.shape = (n_batch, c_in, N)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0,2))
            # gb.shape = (c_out,)
            return gx, gW, gb

def graph_convolution(x, W, n_verts, L_data, L_indices, L_indptr, K, b=None):
    """
    Graph convolution function.

    This is an implementation of graph convolution.
    """
    func = GraphConvolutionFunction(n_verts, L_data, L_indices, L_indptr, K)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)

if __name__ == '__main__':
    x = np.random.randn(5)
    A = scipy.sparse.csr_matrix(np.arange(4*5).reshape((4,5)), dtype=x.dtype)
    y = A.dot(x)
    print(y)
    print "x", x.dtype
    print "y", y.dtype
    print "A", A.dtype
    y2 = cupy.empty((4,), dtype=x.dtype)
    print "y2", y2.dtype
    A_data = cuda.to_gpu(A.data)
    A_indices = cuda.to_gpu(A.indices)
    A_indptr = cuda.to_gpu(A.indptr)
    x2 = cuda.to_gpu(x)
    print "x2", x2.dtype
    csr_matvec(y.shape[0], A_data, A_indices, A_indptr, x2, y2)
    print(y2)

    print "--------------------------------"
    x = np.arange(5*4).reshape((5,4)).astype(np.float32)
    #x = np.asfortranarray(x)
    A = scipy.sparse.csr_matrix(np.arange(3*5).reshape((3,5)), dtype=x.dtype)
    y = A.dot(x)
    print(y)
    print "x", x.dtype
    print "y", y.dtype
    print "A", A.dtype
    y2 = cupy.empty((3,4), dtype=x.dtype)
    #y2 = cupy.asfortranarray(y2)
    print "y2", y2.dtype
    A_data = cuda.to_gpu(A.data)
    A_indices = cuda.to_gpu(A.indices)
    A_indptr = cuda.to_gpu(A.indptr)
    x2 = cuda.to_gpu(x)
    #x2 = cupy.asfortranarray(x2)
    print "x2", x2.dtype
    csr_matvec(y.shape[0], A_data, A_indices, A_indptr, x2, y2)
    print(y2)
