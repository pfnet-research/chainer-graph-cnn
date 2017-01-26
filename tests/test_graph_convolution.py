#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import unittest

import numpy as np
import scipy.sparse

import chainer
from chainer import gradient_check
from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from lib.functions.connection import graph_convolution
from lib import graph


@testing.parameterize(*(testing.product({
    'c_contiguous': [True, False],
    'x_dtype': [np.float32],
    'W_dtype': [np.float32],
}) + testing.product({
    'c_contiguous': [True, False],
    'x_dtype': [np.float64],
    'W_dtype': [np.float64],
})
))
class TestGraphConvolution(unittest.TestCase):

    def setUp(self):
        n_batch = 5
        c_in = 3
        c_out = 2
        N = 4
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
                ]).astype(self.x_dtype)
        self.L = graph.create_laplacian(A, no_diag=True)
        self.K = 25
        self.x = np.random.randn(n_batch,c_in,N).astype(self.x_dtype)
        self.W = np.random.randn(c_out,c_in,self.K).astype(self.W_dtype)
        self.b = np.random.randn(c_out,).astype(self.x_dtype)
        self.gy = np.random.randn(n_batch,c_out,N).astype(self.x_dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': np.float64}
        if self.x_dtype == np.float16 or self.W_dtype == np.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': np.float64, 'atol': 5e-4, 'rtol': 5e-3}

    @attr.gpu
    def test_forward_consistency(self, nobias=False):

        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = graph_convolution.graph_convolution(x_cpu, W_cpu, self.L.shape[0], self.L.data, self.L.indices, self.L.indptr, self.K, b_cpu)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = graph_convolution.graph_convolution(x_gpu, W_gpu, self.L.shape[0], cuda.to_gpu(self.L.data), cuda.to_gpu(self.L.indices), cuda.to_gpu(self.L.indptr), self.K, b_gpu)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data.get(), **self.check_forward_options)

    @attr.gpu
    def test_forward_consistency_nobias(self):
        self.test_forward_consistency(nobias=True)

    def check_backward(self, x_data, W_data, b_data, y_grad, (L_data, L_indices, L_indptr)):
        xp = cuda.get_array_module(x_data)
        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=self.b.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        func = graph_convolution.GraphConvolutionFunction(self.L.shape[0], L_data, L_indices, L_indptr, self.K)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)
        gradient_check.check_backward(
                func, args, y_grad,
                **self.check_backward_options
                )

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy, (self.L.data, self.L.indices, self.L.indptr))

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy, (self.L.data, self.L.indices, self.L.indptr))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy),
                            map(cuda.to_gpu, (self.L.data, self.L.indices, self.L.indptr)))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy),
                            map(cuda.to_gpu, (self.L.data, self.L.indices, self.L.indptr)))

if __name__ == '__main__':
    unittest.main()
