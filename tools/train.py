#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tommi Kerola

import argparse
import json

import numpy as np

import chainer
from chainer import optimizers
from chainer.training import extensions
from chainer.training.updater import ParallelUpdater

from lib.datasets import mnist
from lib.models import graph_cnn
from lib import graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    parser.add_argument('--outdir', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--epoch', '-e', type=int, required=True, help='Number of epochs to train for')
    parser.add_argument('--gpus', '-g', type=int, nargs="*", required=True, help='GPU(s) to use for training')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--snapshot_freq', type=int, default=1, help='Snapshot frequency')
    parser.add_argument('--log_freq', type=int, default=1, help='Log frequency')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    A = graph.grid_graph(28)
    model = graph_cnn.GraphCNN(A)

    optimizer = optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    if 'optimizer' in config:
        optimizer.add_hook(chainer.optimizer.WeightDecay(config['optimizer']['weight_decay']))

    devices = {'main': args.gpus[0]}
    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid

    train_dataset = mnist.MNIST(train=True)
    val_dataset = mnist.MNIST(train=False)

    train_iter = chainer.iterators.MultiprocessIterator(train_dataset, config['batch_size'])
    val_iter = chainer.iterators.SerialIterator(val_dataset, batch_size=1, repeat=False, shuffle=False)

    updater = ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    # Extentions
    trainer.extend(
        extensions.Evaluator(val_iter, model, device=devices['main']),
        trigger=(args.val_freq, 'epoch'))
    trainer.extend(
        extensions.snapshot(trigger=(args.snapshot_freq, 'epoch')))
    trainer.extend(
        extensions.LogReport(trigger=(args.log_freq, 'epoch')))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss', 'main/accuracy',
        'validation/main/loss', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
