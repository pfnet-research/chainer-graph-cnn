#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from chainer import function

class GraphMaxPoolingFunction(function.Function):

    def __init__(self, pooling_inds, use_cudnn=True):
        self.pooling_inds = np.array(pooling_inds)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        x = inputs[0]
        n_batch, c, N = x.shape
        # x.shape = (n_batch, c, N)
        x_pairs = x[:, :, self.pooling_inds]
        # x_pairs = (n_batch*c*N_coarse, 2)
        N_coarse = len(self.pooling_inds)
        m = self.pooling_inds[np.arange(N_coarse), x_pairs.argmax(axis=3)]
        x_inds = np.arange(x.size).reshape(x.shape)
        self.max_inds = x_inds[
               np.arange(n_batch)[:,None,None],
               np.arange(c)[None,:,None],
               m]
        # max_inds.shape = (n_batch, c, N_coarse)
        return x_pairs.max(axis=3),

    #def forward_gpu(self, inputs):
    #    pass

    def backward_cpu(self, inputs, grad_outputs):
        x = inputs[0]
        n_batch, c_in, N = x.shape
        # x.shape = (n_batch, c_in, N)
        gy = grad_outputs[0]
        # gy.shape = (n_batch, c_in, N_coarse)
        gx = np.zeros((n_batch*c_in*N), dtype=x.dtype)
        inds = self.max_inds.ravel()
        gx[inds] = gy.ravel()
        gx = gx.reshape(x.shape)
        return gx,

    #def backward_gpu(self, inputs, grad_outputs):
    #    pass

def graph_max_pooling(x, pooling_inds, use_cudnn=True):
    return GraphMaxPoolingFunction(pooling_inds, use_cudnn)(x)
