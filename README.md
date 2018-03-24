End-To-End Memory Networks in MXNet (Gluon)
========================================

MXNet (using Gluon) implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modelling. The original Tensorflow code from [carpedm20](https://github.com/carpedm20) can be found [here](https://github.com/carpedm20/MemN2N-tensorflow). MXNet Symbolic implementation can be found [here](https://github.com/nicklhy/MemN2N).

![alt tag](http://i.imgur.com/nv89JLc.png)

Before Started
--------------

    $ pip install -r requirements.txt

Usage
--------------

To train a model with 6 hops and memory size of 100, run the following command:

    $ python main.py --nhop 6 --mem_size 100

To test a model with the lastest stored model:
  
    $ python main.py --is_test True

To see all options, run:

    $ python main.py --help
