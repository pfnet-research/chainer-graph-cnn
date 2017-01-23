#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse

from chainer import link
from chainer import initializers
from chainer import cuda

from lib.functions.connection import graph_convolution
from lib import graph

class GraphConvolution(link.Link):

    """
    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.

    This link implements the graph convolution described in
    the paper

    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.

    """

    def __init__(self, in_channels, out_channels, A, K, wscale=1, bias=0,
            nobias=False, initialW=None, initial_bias=None):
        super(GraphConvolution, self).__init__()

        LmI = graph.create_laplacian(A, no_diag=True)
        print("GraphConvolution: Created LmI with {} nodes".format(LmI.shape[0]))

        self.LmI_data = cuda.to_gpu(LmI.data)
        self.LmI_indices = cuda.to_gpu(LmI.indices)
        self.LmI_indptr = cuda.to_gpu(LmI.indptr)
        self.K = K
        self.out_channels = out_channels
        self.n_verts = LmI.shape[0]

        # For backward compatibility
        self.initialW = initialW
        self.wscale = wscale

        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, scale=np.sqrt(wscale))

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=bias_initializer)

    def _initialize_params(self, in_channels):
        W_shape = (self.out_channels, in_channels, self.K)
        self.add_param('W', W_shape, initializer=self._W_initializer)
        print("Initialized W")

    def __call__(self, x, *args):
        """
        Applies the graph convolutional layer.
        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])
        return graph_convolution.graph_convolution(
                x, self.W, self.n_verts, self.LmI_data, self.LmI_indices, self.LmI_indptr, self.K, self.b)

