#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import unittest

import numpy as np
import scipy.sparse

import chainer
# from chainer import testing
from chainer import gradient_check

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
        self.L = graph.create_laplacian(A)
        self.K = 25
        self.x = np.random.randn(n_batch,c_in,N).astype(dtype)
        self.W = np.random.randn(c_out,c_in,self.K).astype(dtype)
        self.b = np.random.randn(c_out,).astype(dtype)
        self.gy = np.random.randn(n_batch,c_out,N).astype(dtype)

    def test_backward_cpu(self):
        x_data = self.x
        W_data = self.W
        b_data = self.b
        y_grad = self.gy

        func = graph_convolution.GraphConvolutionFunction(self.L, self.K)

        args = (x_data, W_data, b_data)
        check_backward_options = {'dtype': np.float64}
        gradient_check.check_backward(
                func, args, y_grad,
                **check_backward_options
                )

if __name__ == '__main__':
    unittest.main()
