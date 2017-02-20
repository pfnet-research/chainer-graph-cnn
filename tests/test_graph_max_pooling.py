#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from lib.functions.pooling import graph_max_pooling


@testing.parameterize(*testing.product({
    'dtype': [np.float32, np.float64],
}))
class TestGraphMaxPooling(unittest.TestCase):

    def setUp(self):
        n_batch = 5
        c_in = 3
        N = 4
        self.K = 25
        self.pooling_inds = np.array([[0, 1], [2, 2], [3, 3]])
        N_coarse = len(self.pooling_inds)
        self.x = np.arange(n_batch * c_in * N,
                           dtype=self.dtype).reshape((n_batch, c_in, N))
        self.gy = np.random.randn(n_batch, c_in, N_coarse).astype(self.dtype)
        self.check_backward_options = {'eps': 2.0 ** -8}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        func = graph_max_pooling.GraphMaxPoolingFunction(self.pooling_inds)
        y = func(x)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        n_batch, c_in = x.shape[:2]
        for k in six.moves.range(n_batch):
            for c in six.moves.range(c_in):
                x = self.x[k, c]
                expect = np.array([x[0:2].max(), x[2], x[3]])
                testing.assert_allclose(expect, y_data[k, c])

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):

        func = graph_max_pooling.GraphMaxPoolingFunction(self.pooling_inds)

        args = (x_data,)
        gradient_check.check_backward(
            func, args, y_grad,
            **self.check_backward_options
        )

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
