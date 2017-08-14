#!/usr/bin/env python
# MNIST data pre-processing script.
import gzip
import struct
import sys
import functools
import subprocess
from distutils import sysconfig
import imp
import cProfile as profile
import pstats
import io

import cv2
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt

def load_cython(lib):
  subprocess.check_call(
    ['cython', '-a', lib + '.pyx'])

  headers = sysconfig.get_python_inc()
  subprocess.check_call(
    ['gcc', '-shared', '-pthread', '-fPIC', '-fwrapv', '-O2', '-Wall',
     '-fno-strict-aliasing', '-fopenmp', '-I' + headers, '-o', lib + '.so', lib + '.c'])
  return imp.load_dynamic(lib, lib + '.so')

distort_images = load_cython('distort_images')


def read_images(file):
  """Reads MNIST file into numpy tensor."""
  # From https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
  zero, data_type, dims = struct.unpack('>HBB', file.read(4))
  shape = tuple(struct.unpack('>I', file.read(4))[0] for d in range(dims))
  return np.fromstring(file.read(), dtype=np.uint8).reshape(shape)


def save_images(images, file):
  """Save images array to outfile."""
  file.write(b'\x00\x00\x08')  # Header
  file.write(struct.pack('>B', len(images.shape)))
  for dimension in images.shape:
    file.write(struct.pack('>I', dimension))
  file.write(images.tobytes())


def read_labels(file):
  zero, data_type, dims = struct.unpack('>HBB', file.read(4))
  count = struct.unpack('>I', file.read(4))
  return np.fromstring(file.read(), dtype=np.uint8)


def save_labels(labels, file):
  file.write(b'\x00\x00\x08\x01')
  file.write(struct.pack('>I', labels.shape[0]))
  file.write(labels.tobytes())


def elastic_distort(images, sigma=4, alpha=34, kernel_size=33):
  """Distort elastically."""
  # Create distortion vector field.
  shape = images.shape
  randoms = np.random.uniform(
    -1, 1, (shape[0], shape[1], shape[2], 2)) * alpha
  filtered1d = ndimage.filters.gaussian_filter(
    input=randoms,
    sigma=(0, 0, sigma, 0),
    order=0,
    mode='constant',
    cval=0.0)
  filtered2d = ndimage.filters.gaussian_filter(
    input=filtered1d,
    sigma=(0, sigma, 0, 0),
    order=0,
    mode='constant',
    cval=0.0)

  # Distort.
  # plt.imshow(np.concatenate((filtered2d[0], np.zeros((28, 28, 1))), axis=2))
  # plt.show()
  return distort_images.distort_images(images, filtered2d.astype(np.float32))


def display_n_images(images, n):
  for i in range(n):
    figure = plt.figure()
    f1 = figure.add_subplot(1, 2, 1)
    plt.imshow(distorted_images[i], cmap='gray')
    f2 = figure.add_subplot(1, 2, 2)
    plt.imshow(input_images[i], cmap='gray')
    plt.show()


def process_images(images, distort):
  # Elastic distortions.
  if distort:
    images = elastic_distort(images, sigma=4, alpha=34)

  return ndimage.interpolation.zoom(
    input=images,
    zoom=(1, 29.0 / 28.0, 29.0 / 28.0),
    order=1,
    mode='constant',
    cval=0.0)


def augment_dataset(
    input_filename,
    labels_filename,
    output_filename,
    output_labels_filename,
    duplication=1,
    distort=True):
  print('Reading images.')
  with gzip.open(input_filename, 'rb') as infile:
    input_images = read_images(infile)

  # Augment.
  print('Processing images.')
  output_images = np.empty((0, 29, 29), dtype=np.uint8)
  for i in range(duplication):
    output_images = np.concatenate(
      (output_images, process_images(input_images, distort)), axis=0)

  print('Writing images.')
  with open(output_filename, 'wb') as outfile:
    save_images(output_images, outfile)

  # Labels.
  print('Processing and writing labels.')
  with gzip.open(labels_filename, 'rb') as infile:
    labels = read_labels(infile)
  labels = np.tile(labels, duplication)
  with open(output_labels_filename, 'wb') as outfile:
    save_labels(labels, outfile)

  print('Done.')


def main(argv):
  augment_dataset(
    'MNIST_data/train-images-idx3-ubyte.gz',
    'MNIST_data/train-labels-idx1-ubyte.gz',
    'MNIST_data/train-images-processed',
    'MNIST_data/train-labels-processed',
    duplication=8)

  augment_dataset(
    'MNIST_data/t10k-images-idx3-ubyte.gz',
    'MNIST_data/t10k-labels-idx1-ubyte.gz',
    'MNIST_data/t10k-images-processed',
    'MNIST_data/t10k-labels-processed',
    distort=False)

if __name__ == '__main__':
  main(sys.argv)