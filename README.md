This is a Chainer implementation of 'Convolutional Neural Networks on
Graphs with Fast Localized Spectral Filtering'
(https://arxiv.org/abs/1606.09375)
Published in NIPS 2016.

Based on
https://github.com/mdeff/cnn_graph

Usage
-----
```
# Trains a GraphCNN on MNIST
$ python tools/train.py -c configs/default.json -o results -e 100 -g 0
```
