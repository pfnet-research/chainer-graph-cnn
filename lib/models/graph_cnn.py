#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.evaluation import accuracy
from chainer import reporter

from lib.links.connection.graph_convolution import GraphConvolution
from lib.functions.pooling.graph_max_pooling import GraphMaxPoolingFunction
from lib import coarsening


class GraphCNN(chainer.Chain):
    """Graph CNN example implementation.

    Uses the GC32-P4-GC64-P4-FC512 architecture as in the original paper.
    """

    def __init__(self, A, n_out=10):
        super(GraphCNN, self).__init__()

        # Precompute the coarsened graphs
        graphs, pooling_inds = coarsening.coarsen(A, levels=4)
        # In order to simulate 2x2 max pooling, combine the 4 levels
        # of graphs into 2 levels by combining pooling indices.
        graphs, pooling_inds = coarsening.combine(graphs, pooling_inds, 2)

        self.graph_layers = []
        sizes = [32, 64]
        for i, (g, inds, s) in enumerate(zip(graphs, pooling_inds, sizes)):
            f = GraphConvolution(None, s, g, K=25)
            self.add_link('gconv{}'.format(i), f)
            p = GraphMaxPoolingFunction(inds)
            self.graph_layers.append((f, p))

        self.linear_layers = []
        sizes = [512]
        for i, s in enumerate(sizes):
            f = L.Linear(None, s)
            self.add_link('l{}'.format(i), f)
            self.linear_layers.append(f)
        self.add_link('cls_layer', L.Linear(None, n_out))

        self.train = True

    def __call__(self, x, *args):
        # x.shape = (n_batch, n_channels, h*w)
        n_batch = x.shape[0]
        dropout_ratio = 0.5

        h = x
        # Graph convolutional layers
        for f, p in self.graph_layers:
            h = p(F.relu(f(h)))

        # Fully connected layers
        for f in self.linear_layers:
            h = F.relu(F.dropout(f(h), dropout_ratio, train=self.train))

        # Linear classification layer
        h = self.cls_layer(h)

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
