# Chainer Graph CNN

This is a Chainer implementation of
_Defferrard et al., "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering", NIPS 2016._
(https://arxiv.org/abs/1606.09375)

Disclaimer: PFN provides no warranty or support for this implementation. Use it at your own risk.
See [license](LICENSE) for details.

This is not the original author's implementation. This implementation was based on [https://github.com/mdeff/cnn_graph](https://github.com/mdeff/cnn_graph).

Usage
-----
```
# Trains a GraphCNN on MNIST
$ python tools/train.py -c configs/default.json -o results -e 100 -g 0
```

Prerequisites
-------------
```
pip install -r requirements.txt
```
This implementation has been tested with the following versions.
```
python 2.7.6
chainer (1.19.0)
nose (1.3.7)
numpy (1.11.3)
scikit-learn (0.18.1)
scipy (0.18.1)
```
It may work with other versions; not tested.

Results
-------
```
Using ADAM alpha=1e-4
epoch       iteration   main/loss   main/accuracy  validation/main/loss  validation/main/accuracy
1           600         0.515395    0.854901       0.193552              0.9453
2           1200        0.195267    0.942567       0.122769              0.9652
3           1800        0.139023    0.95875        0.0955012             0.9726
4           2400        0.110456    0.9676         0.0769727             0.9762
5           3000        0.0932845   0.972033       0.0643796             0.9812
6           3600        0.0811693   0.975149       0.0603944             0.9824
7           4200        0.074127    0.978266       0.0556359             0.9831
8           4800        0.0670138   0.980266       0.0509385             0.9839
9           5400        0.0625065   0.980933       0.0496262             0.9839
10          6000        0.0585658   0.982366       0.0493765             0.9838
11          6600        0.0547269   0.983082       0.0444783             0.9859
12          7200        0.050334    0.984582       0.0413585             0.9866
13          7800        0.0493707   0.985032       0.0416611             0.9873
14          8400        0.0459602   0.985999       0.0437013             0.9859
15          9000        0.044378    0.986715       0.0406627             0.987
16          9600        0.0430196   0.986815       0.0394637             0.9866
17          10200       0.0404675   0.988182       0.0385143             0.9877
18          10800       0.0398833   0.988265       0.0366019             0.989
19          11400       0.0371923   0.988998       0.0348309             0.9875
20          12000       0.0361765   0.989215       0.0402662             0.9858
-- snip --
100         60000       0.0157423   0.995832       0.0292472             0.9901
```

```
Using ADAM alpha=1e-3
epoch       iteration   main/loss   main/accuracy  validation/main/loss  validation/main/accuracy
1           600         0.225126    0.930017       0.0767015             0.9768
2           1200        0.0977682   0.969899       0.0606019             0.9801
3           1800        0.0770546   0.976016       0.0513997             0.9838
4           2400        0.0666313   0.979532       0.0424098             0.9866
5           3000        0.06334     0.980782       0.051125              0.9841
6           3600        0.0578026   0.982532       0.0457874             0.985
7           4200        0.0541042   0.983982       0.0405522             0.9875
8           4800        0.0514735   0.984432       0.0443701             0.9867
9           5400        0.0503822   0.984448       0.0557598             0.9812
10          6000        0.0465654   0.985432       0.035589              0.9897
11          6600        0.0455079   0.985932       0.03442               0.988
12          7200        0.0425339   0.986882       0.038998              0.9868
13          7800        0.0427513   0.986899       0.0395496             0.9873
14          8400        0.0431217   0.986815       0.0372915             0.9877
15          9000        0.0420674   0.987432       0.0401286             0.9864
16          9600        0.0408353   0.987482       0.0404751             0.9876
17          10200       0.0401931   0.987515       0.0372056             0.9879
18          10800       0.0388781   0.988315       0.0389307             0.9889
19          11400       0.0391798   0.988198       0.0406604             0.9872
20          12000       0.0380889   0.988298       0.039208              0.9867
-- snip --
100         60000       0.0320832   0.990331       0.0345484             0.9887
```

License
-------
MIT License. Please see the LICENSE file for details.
