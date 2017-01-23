#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import chainer

from chainer.datasets.mnist import _retrieve_mnist_training, _retrieve_mnist_test

class MNIST(chainer.dataset.DatasetMixin):
    def __init__(self, train, with_gt=True):
        if train:
            raw = _retrieve_mnist_training()
        else:
            raw = _retrieve_mnist_test()
        images = raw['x'].reshape((-1, 1, 28*28))
        images = images.astype(np.float32) # (n_samples, 1, 784)
        images /= 255.
        self.images = images

        self.with_gt = with_gt

        if with_gt:
            self.labels = raw['y'].astype(np.int32)
        print("images:",self.images.shape)
        print("labels:",self.labels.shape)

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        img = self.images[i]
        if not self.with_gt:
            return img
        label = self.labels[i]
        return img, label
