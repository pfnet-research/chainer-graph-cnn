#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import unittest

import numpy as np
import scipy.sparse

import chainer
# from chainer import testing
from chainer import gradient_check
from chainer import cuda

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

    def test_forward_consistency(self, nobias=False):

        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        func_cpu = graph_convolution.GraphConvolutionFunction(self.L.shape[0], self.L.data, self.L.indices, self.L.indptr, self.K)
        y_cpu = func_cpu(x_cpu, W_cpu, b_cpu)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        func_gpu = graph_convolution.GraphConvolutionFunction(self.L.shape[0], cuda.to_gpu(self.L.data), cuda.to_gpu(self.L.indices), cuda.to_gpu(self.L.indptr), self.K)
        y_gpu = func_gpu(x_gpu, W_gpu, b_gpu)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data.get())

    def test_backward_cpu(self):
        x_data = self.x
        W_data = self.W
        b_data = self.b
        y_grad = self.gy

        func = graph_convolution.GraphConvolutionFunction(self.L.shape[0], self.L.data, self.L.indices, self.L.indptr, self.K)

        args = (x_data, W_data, b_data)
        check_backward_options = {'dtype': np.float64}
        gradient_check.check_backward(
                func, args, y_grad,
                **check_backward_options
                )

if __name__ == '__main__':
    unittest.main()
