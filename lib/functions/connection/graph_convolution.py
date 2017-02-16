#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.cuda import cupy
from chainer.utils import type_check


def chebyshev_matvec_cpu(C, x, K, n_batch, LmI):
    C[:, 0] = x.transpose((0, 2, 1))  # (n_batch, N, c_in)
    # NOTE(tommi): scipy.sparse does not support sparse tensordot,
    # so have to use a for loop, although inefficient.
    if K > 1:
        for i in six.moves.range(n_batch):
            C[i, 1] = LmI.dot(C[i, 0])
    for k in six.moves.range(2, K):
        for i in six.moves.range(n_batch):
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

    def chebyshev_matvec_gpu(C, x, K, n_batch,
                             LmI_data, LmI_indices, LmI_indptr):
        C[0] = x.transpose((2, 1, 0))
        N = C.shape[1]
        if K > 1:
                csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[0], C[1])
        for k in six.moves.range(2, K):
            csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[k-1], C[k])
            C[k] = 2 * C[k] - C[k-2]


class GraphConvolutionFunction(function.Function):

    def __init__(self, L, K):
        # NOTE(tommi): It is very important that L
        # is a normalized Graph Laplacian matrix.
        # Otherwise, this will not work.

        I = scipy.sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
        self.LmI = L - I
        self.LmI_tuple = (self.LmI.data, self.LmI.indices, self.LmI.indptr)

        self.K = K

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 3,
            w_type.ndim == 3,
            x_type.shape[1] == w_type.shape[1],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def to_cpu(self):
        self.LmI_tuple = map(cuda.to_cpu, self.LmI_tuple)

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            self.LmI_tuple = map(cuda.to_gpu, self.LmI_tuple)

    def forward_cpu(self, inputs):
        x, W = inputs[:2]
        n_batch, c_in, N = x.shape
        b = inputs[2] if len(inputs) == 3 else None

        K = self.K
        if x.dtype != self.LmI.dtype:
            self.LmI = self.LmI.astype(x.dtype)

        C = np.empty((n_batch, K, N, c_in), dtype=x.dtype)
        chebyshev_matvec_cpu(C, x, K, n_batch, self.LmI)
        C = C.transpose((0, 3, 1, 2))
        self.C = C
        y = np.tensordot(C, W, ((1, 2), (1, 2)))

        if b is not None:
            y += b

        return np.rollaxis(y, 2, 1),  # y.shape = (n_batch, c_out, N)

    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        n_batch, c_in, N = x.shape
        b = inputs[2] if len(inputs) == 3 else None
        xp = cuda.get_array_module(x)
        with cuda.get_device(x.data):
            K = self.K
            LmI_data, LmI_indices, LmI_indptr = self.LmI_tuple

            if x.dtype != LmI_data.dtype:
                LmI_data = LmI_data.astype(x.dtype)

            C = xp.empty((K, N, c_in, n_batch), dtype=x.dtype)
            chebyshev_matvec_gpu(C, x, K, n_batch,
                                 LmI_data, LmI_indices, LmI_indptr)

            C = C.transpose((3, 2, 0, 1))
            self.C = C
            y = xp.tensordot(C, W, ((1, 2), (1, 2)))

            if b is not None:
                y += b

            return xp.rollaxis(y, 2, 1),  # y.shape = (n_batch, c_out, N)

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]

        n_batch, c_in, N = x.shape
        c_out = gy.shape[1]

        gW = np.tensordot(gy, self.C, ((0, 2), (0, 3))).astype(W.dtype, copy=False)

        K = self.K
        if x.dtype != self.LmI.dtype:
            self.LmI = self.LmI.astype(x.dtype)

        C = np.empty((n_batch, K, N, c_out), dtype=x.dtype)
        chebyshev_matvec_cpu(C, gy, K, n_batch, self.LmI)
        C = C.transpose((0, 3, 1, 2))
        gx = np.tensordot(C, W, ((1, 2), (0, 2)))
        gx = np.rollaxis(gx, 2, 1)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        xp = cuda.get_array_module(x)
        with cuda.get_device(x.data):
            n_batch, c_in, N = x.shape
            c_out = gy.shape[1]

            gW = xp.tensordot(gy, self.C, ((0, 2), (0, 3))).astype(W.dtype, copy=False)

            K = self.K
            LmI_data, LmI_indices, LmI_indptr = self.LmI_tuple

            if x.dtype != LmI_data.dtype:
                LmI_data = LmI_data.astype(x.dtype)

            C = xp.empty((K, N, c_out, n_batch), dtype=x.dtype)
            chebyshev_matvec_gpu(C, gy, K, n_batch,
                                 LmI_data, LmI_indices, LmI_indptr)
            C = C.transpose((3, 2, 0, 1))
            gx = xp.tensordot(C, W, ((1, 2), (0, 2)))
            gx = xp.rollaxis(gx, 2, 1)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2))
            return gx, gW, gb


def graph_convolution(x, W, L, K, b=None):
    """Graph convolution function.

    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.
    This is an implementation the graph convolution described in
    the following paper:

    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.

    Notation:
    - :math:`n_batch` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`n_vertices` is the number of vertices in the graph.

    Args:
        x (~chainer.Variable): Input graph signal.
            Its shape is :math:`(n_batch, c_I, n_vertices)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`c_O, c_I, K`.
        L (scipy.sparse.csr_matrix): Normalized graph Laplacian matrix
            that describes the graph.
        K (int): Polynomial order of the Chebyshev approximation.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional)

    Returns:
        ~chainer.Variable: Output variable.

    If the bias vector is given, it is added to all spatial locations of the
    output of the graph convolution.

    """
    func = GraphConvolutionFunction(L, K)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)


if __name__ == '__main__':
    # for quick testing of csr_matvec
    x = np.random.randn(5)
    A = scipy.sparse.csr_matrix(np.arange(4*5).reshape((4, 5)), dtype=x.dtype)
    y = A.dot(x)
    print(y)
    print("x", x.dtype)
    print("y", y.dtype)
    print("A", A.dtype)
    y2 = cupy.empty((4,), dtype=x.dtype)
    print("y2", y2.dtype)
    A_data = cuda.to_gpu(A.data)
    A_indices = cuda.to_gpu(A.indices)
    A_indptr = cuda.to_gpu(A.indptr)
    x2 = cuda.to_gpu(x)
    print("x2", x2.dtype)
    csr_matvec(y.shape[0], A_data, A_indices, A_indptr, x2, y2)
    print(y2)

    print("--------------------------------")
    x = np.arange(5*4).reshape((5, 4)).astype(np.float32)
    A = scipy.sparse.csr_matrix(np.arange(3*5).reshape((3, 5)), dtype=x.dtype)
    y = A.dot(x)
    print(y)
    print("x", x.dtype)
    print("y", y.dtype)
    print("A", A.dtype)
    y2 = cupy.empty((3, 4), dtype=x.dtype)
    print("y2", y2.dtype)
    A_data = cuda.to_gpu(A.data)
    A_indices = cuda.to_gpu(A.indices)
    A_indptr = cuda.to_gpu(A.indptr)
    x2 = cuda.to_gpu(x)
    print("x2", x2.dtype)
    csr_matvec(y.shape[0], A_data, A_indices, A_indptr, x2, y2)
    print(y2)
