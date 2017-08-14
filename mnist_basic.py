#!/usr/bin/env python
# Following the tutorial from
# https://www.tensorflow.org/get_started/mnist/pros
import datetime
import random
import math
import gzip

import tensorflow as tf
from tensorflow import flags
from tensorflow.contrib import learn
import tensorflow.contrib.image
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors as tferrors
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
import numpy as np
from scipy import ndimage

import process_data

image_vector_distort = tf.load_op_library('bin/image_vector_distort.so')

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, 'Train model. Evaluates if false.')
flags.DEFINE_integer('epochs', 15, 'Number of steps to train.')
flags.DEFINE_bool(
  'clean', True,
  'Train new model from scratch instead of improving existing model.')
flags.DEFINE_bool('summaries', False, 'Log detailed image summaries.')

# Model parameters.
flags.DEFINE_integer('columns', 7, 'Columns in multi-column convnet.')
flags.DEFINE_bool('dropout', True, 'Enable dropout in training.')

# Enable normal logging.
tf.logging.set_verbosity(tf.logging.INFO)


def with_dependencies(dependencies, op):
  """Returns `op` with dependencies added.

  Arguments:
    dependencies: List of tensorflow ops.
    op: Tensorflow op.
   """
  with tf.control_dependencies(dependencies):
    return tf.identity(op)


def conv_pool_layer(
    inputs,
    filters,
    conv_padding='valid',  # no padding
    conv_sample_size=5,  # Convolve size
    pool_size=2,
    name=None):
  """Convolve2D and max-pool layer."""
  with tf.variable_scope(name):
    conv = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=[conv_sample_size] * 2,
      padding=conv_padding,
      activation=tf.nn.relu,
      bias_initializer=tf.truncated_normal_initializer(stddev=0.1),
      name='conv')

    pool = tf.layers.max_pooling2d(
      inputs=conv,
      pool_size=[pool_size] * 2,
      strides=pool_size,
      padding='same',
      name='pool')

    return pool


def cnn_model_fn(features, labels, mode):
  """Model definition."""
  input_layer = tf.reshape(
    features['images'], [-1, 29, 29, 1], name='input/layer')

  # Conv-pool layers.
  conv_pool1 = conv_pool_layer(
    inputs=input_layer,
    filters=32,
    conv_sample_size=4,
    pool_size=2,
    name='conv_pool1')  # Output 26x26x32 -> 13x13x32
  conv_pool2 = conv_pool_layer(
    inputs=conv_pool1,
    filters=64,
    conv_sample_size=5,
    pool_size=3,
    name='conv_pool2')  # Output 9x9x64 -> 3x3x64

  # Final Fully/densely-connected layer
  conv_pool2_flat = tf.reshape(conv_pool2, [-1, 3 * 3 * 64])
  fc1 = tf.layers.dense(
    inputs=conv_pool2_flat,
    units=150,
    activation=tf.nn.relu,
    name='fc1')
  if FLAGS.dropout:
    dropout = tf.layers.dropout(
      inputs=fc1,
      rate=0.5,
      training=(mode == learn.ModeKeys.TRAIN),
      name='dropout')
  else:
    dropout = fc1

  # Final connected layer
  logits = tf.layers.dense(dropout, units=10, name='logits')

  loss = None
  train_op = None

  # Configure loss function.
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(labels, depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Training op
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer=tf.train.AdamOptimizer)
    tf.summary.scalar('loss', loss)

  # Predict values.
  predictions = {
    'classes': tf.argmax(logits, axis=1, name='classes'),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
  }

  return model_fn_lib.ModelFnOps(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op
  )

def main(unused_argv):
  """Load data and train."""
  # Read MNIST data.
  with open('MNIST_data/train-images-processed', 'rb') as file:
    train_data = process_data.read_images(file).astype(np.float32)
  with open('MNIST_data/train-labels-processed', 'rb') as file:
    train_labels = process_data.read_labels(file).astype(np.int32)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=FLAGS.epochs,
    shuffle=True)

  with open('MNIST_data/t10k-images-processed', 'rb') as file:
    eval_data = process_data.read_images(file).astype(np.float32)
  with open('MNIST_data/t10k-labels-processed', 'rb') as file:
    eval_labels = process_data.read_labels(file).astype(np.int32)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': eval_data},
    y=eval_labels,
    batch_size=100,
    shuffle=False)

  # Create classifiers.
  model_dirs = ['model/model_%d' % i for i in range(FLAGS.columns)]
  mnist_classifiers = [
    learn.Estimator(
      model_fn=cnn_model_fn,
      # config=learn.RunConfig(save_checkpoints_steps=600),
      model_dir=model_dir)
    for model_dir in model_dirs]

  # Train.
  if FLAGS.train:
    # Delete existing model if it exists.
    if FLAGS.clean:
      try:
        tf.gfile.DeleteRecursively('model')
      except tferrors.NotFoundError:
        pass

    monitors = [
      # learn.monitors.ValidationMonitor(
      #   input_fn=eval_input_fn,
      #   every_n_steps=600,
      #   metrics={
      #     'accuracy': learn.MetricSpec(
      #       metric_fn=tf.metrics.accuracy, prediction_key='classes')
      #   })
    ]
    for classifier in mnist_classifiers:
      classifier.fit(
        input_fn=train_input_fn,
        monitors=monitors)

  # Evaluate.
  predictions = []
  for classifier in mnist_classifiers:
    probabilities = classifier.predict(
      input_fn=eval_input_fn,
      outputs=['probabilities'])
    prediction = [x['probabilities'] for x in probabilities]
    predictions.append(prediction)

  # Print results.
  errors = []
  for i in range(len(predictions)):
    prediction_numbers = np.argmax(predictions[i], axis=1)
    accuracy = np.mean(np.equal(prediction_numbers, eval_labels)) * 100.0
    errors.append(100 - accuracy)
  print('errors %s%%' % str(errors))
  print('average %g stddev %g' % (np.mean(errors), np.std(errors)))
  average_predictions = np.argmax(np.mean(predictions, axis=0), axis=1)
  accuracy = np.mean(np.equal(average_predictions, eval_labels)) * 100.0
  print('multi-column error %g%%' % (100 - accuracy))


if __name__ == '__main__':
  tf.app.run()
