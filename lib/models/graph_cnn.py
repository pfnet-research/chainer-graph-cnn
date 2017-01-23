#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer import reporter

from lib.links.connection.graph_convolution import GraphConvolution
from lib.functions.pooling.graph_max_pooling import graph_max_pooling
from lib import coarsening

class GraphCNN(chainer.Chain):

    def __init__(self, A, coarsening_levels=3, n_out=10):
        N = A.shape[0]
        # Precompute the coarsened graphs
        graphs, pooling_inds = coarsening.coarsen(A, levels=coarsening_levels)
        self.pooling_inds = pooling_inds
        layers = {
                'gconv0': GraphConvolution(1, 32, graphs[0], K=25),
                'gconv1': GraphConvolution(32, 64, graphs[1], K=25),
                'gconv2': GraphConvolution(64, 128, graphs[2], K=25),
                'fc0': L.Linear(None, 1000),
                'fc1': L.Linear(None, 1000),
                'fc2': L.Linear(None, n_out),
                }
        super(GraphCNN, self).__init__(**layers)

    def __call__(self, x, *args):
        # x.shape = (n_batch, n_channels, h*w)

        h = x
        # Graph convolutional layers
        h = F.relu(self.gconv0(h))
        h = graph_max_pooling(h, self.pooling_inds[0])
        h = F.relu(self.gconv1(h))
        h = graph_max_pooling(h, self.pooling_inds[1])
        h = F.relu(self.gconv2(h))
        h = graph_max_pooling(h, self.pooling_inds[2])

        n_batch = h.shape[0]
        h = F.reshape(h, (n_batch, -1,))
        # Fully connected layers
        h = F.relu(self.fc0(h))
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        if args:
            labels = args[0]
            loss = F.softmax_cross_entropy(h, labels)
            acc = accuracy.accuracy(h, labels)
            reporter.report({
                'loss': loss,
                'accuracy': acc},
                self)

            return loss

        return h
