#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import unittest

import numpy as np
import scipy.sparse

import chainer
# from chainer import testing
from chainer import gradient_check

from lib.functions.pooling import graph_max_pooling
from lib import graph

class TestGraphMaxPooling(unittest.TestCase):

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
                ]).astype(np.float32)
        self.L = graph.create_laplacian(A)
        self.K = 25
        self.pooling_inds = np.array([[0,1],[2,2],[3,3]])
        N_coarse = len(self.pooling_inds)
        self.x = np.random.randn(n_batch,c_in,N).astype(np.float32)
        self.gy = np.random.randn(n_batch,c_in,N_coarse).astype(np.float32)

    def test_backward_cpu(self):
        x_data = self.x
        y_grad = self.gy

        func = graph_max_pooling.GraphMaxPoolingFunction(self.pooling_inds)

        args = (x_data,)
        check_backward_options = {'dtype': np.float64}
        gradient_check.check_backward(
                func, args, y_grad,
                **check_backward_options
                )

if __name__ == '__main__':
    unittest.main()
