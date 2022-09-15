import os
import argparse
import logging
import mxnet as mx
import numpy as np
import gzip
import struct
from common import fit
from common import utils
# This example only for mlp network
from symbols import mlp

# Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics
logging.basicConfig(
format="%(asctime)s %(levelname)-8s %(message)s",
datefmt="%Y-%m-%dT%H:%M:%SZ",
level=logging.DEBUG)


def get_mnist_iter(args, kv):
  """
Create data iterator with NDArrayIter
"""
    mnist = mx.test_utils.get_mnist()

# Get MNIST data.
    train_data = mx.io.NDArrayIter(
    mnist['train_data'], mnist['train_label'], args.batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(
    mnist['test_data'], mnist['test_label'], args.batch_size)

    return (train_data, val_data)


if __name__ == '__main__':
  # parse args
  parser = argparse.ArgumentParser(description="train mnist",
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--num-classes', type=int, default=10,
  help='the number of classes')
  parser.add_argument('--num-examples', type=int, default=60000,
  help='the number of training examples')

  parser.add_argument('--add_stn',  action="store_true", default=False,
  help='Add Spatial Transformer Network Layer (lenet only)')
  parser.add_argument('--image_shape', default='1, 28, 28', help='shape of training images')

  fit.add_fit_args(parser)
  parser.set_defaults(
  # network
  network='mlp',
  # train
  gpus=None,
  batch_size=64,
  disp_batches=100,
  num_epochs=10,
  lr=.05,
  lr_step_epochs='10'
  )
  args = parser.parse_args()

  # load mlp network
  sym = mlp.get_symbol(**vars(args))

  # train
  fit.fit(args, sym, get_mnist_iter)