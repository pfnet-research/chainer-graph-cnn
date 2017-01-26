#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        super(GraphCNN, self).__init__()

        # Precompute the coarsened graphs
        graphs, pooling_inds = coarsening.coarsen(A, levels=coarsening_levels)

        self.graph_layers = []
        sizes = [32, 64, 128]
        for i, (g, p, s) in enumerate(zip(graphs, pooling_inds, sizes)):
            f = GraphConvolution(None, s, g, K=25)
            self.add_link('gconv{}'.format(i), f)
            self.graph_layers.append((f, p))

        self.linear_layers = []
        sizes = [1000, 1000, 10]
        for i, s in enumerate(sizes):
            f = L.Linear(None, s)
            self.add_link('l{}'.format(i), f)
            self.linear_layers.append(f)

    def __call__(self, x, *args):
        # x.shape = (n_batch, n_channels, h*w)
        n_batch = x.shape[0]

        h = x
        # Graph convolutional layers
        for f, p in self.graph_layers:
            h = F.relu(f(h))
            h = graph_max_pooling(h, p)

        h = F.reshape(h, (n_batch, -1,))
        # Fully connected layers
        for f in self.linear_layers:
            h = F.relu(f(h))

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
