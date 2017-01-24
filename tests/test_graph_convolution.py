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

class TestGraphConvolution(unittest.TestCase):

    def setUp(self):
        dtype = np.float64
        n_batch = 5
        c_in = 3
        c_out = 2
        N = 4
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
                ]).astype(dtype)
        self.L = graph.create_laplacian(A, no_diag=True)
        self.K = 25
        self.x = np.random.randn(n_batch,c_in,N).astype(dtype)
        self.W = np.random.randn(c_out,c_in,self.K).astype(dtype)
        self.b = np.random.randn(c_out,).astype(dtype)
        self.gy = np.random.randn(n_batch,c_out,N).astype(dtype)

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
            y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency_nobias(self):
        self.test_forward_consistency(nobias=True)

    def check_backward(self, x_data, W_data, b_data, y_grad, (L_data, L_indices, L_indptr)):

        func = graph_convolution.GraphConvolutionFunction(self.L.shape[0], L_data, L_indices, L_indptr, self.K)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)
        check_backward_options = {'dtype': np.float64}
        gradient_check.check_backward(
                func, args, y_grad,
                **check_backward_options
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
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy),
                            map(cuda.to_gpu, (self.L.data, self.L.indices, self.L.indptr)))

if __name__ == '__main__':
    unittest.main()
