#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tempfile

import unittest

import numpy as np

import chainer
from chainer import optimizers
from chainer import testing
from chainer.training import extensions
from chainer.training.updater import ParallelUpdater

from lib.models import graph_cnn

from sklearn.datasets import make_classification


class EasyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, train, with_gt=True, n_classes=2):

        X, y = make_classification(n_samples=1000, n_features=4,
                                   flip_y=0.0,
                                   n_classes=n_classes)
        self.X = X[:, None, :].astype(np.float32)
        self.y = y.astype(np.int32)
        print("X:", self.X.shape)
        print("y:", self.y.shape)

        self.with_gt = with_gt

    def __len__(self):
        return len(self.X)

    def get_example(self, i):
        x = self.X[i]
        if not self.with_gt:
            return x
        label = self.y[i]
        return x, label


class TestGraphCNN(unittest.TestCase):

    def check_train(self, gpu):
        outdir = tempfile.mkdtemp()
        print("outdir: {}".format(outdir))

        n_classes = 2
        batch_size = 32

        devices = {'main': gpu}

        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]).astype(np.float32)
        model = graph_cnn.GraphCNN(A, n_out=n_classes)

        optimizer = optimizers.Adam(alpha=1e-4)
        optimizer.setup(model)
        train_dataset = EasyDataset(train=True, n_classes=n_classes)
        train_iter = chainer.iterators.MultiprocessIterator(
            train_dataset, batch_size)
        updater = ParallelUpdater(train_iter, optimizer, devices=devices)
        trainer = chainer.training.Trainer(updater, (10, 'epoch'), out=outdir)
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/accuracy']))
        trainer.extend(extensions.ProgressBar())
        trainer.run()

    def test_train_cpu(self):
        self.check_train(-1)

    def test_train_gpu(self):
        self.check_train(1)


testing.run_module(__name__, __file__)
