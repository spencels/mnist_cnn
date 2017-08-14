import math

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

image_vector_distort = tf.load_op_library('bin/image_vector_distort.so')

def expand_to_4d(image):
  """Expands 2d image tensor to 4d."""
  return tf.expand_dims(tf.expand_dims(image, 0), 3)

class ImageVectorDistortTest(test.TestCase):
  def test_Identity(self):
    images = tf.reshape(tf.range(0, 9, dtype=tf.float32), [1, 3, 3, 1])
    distortions = tf.zeros([1, 3, 3, 2])
    with tf.Session():
      distorted = image_vector_distort.image_vector_distort(
        images=images,
        distortion_fields=distortions)
      
      self.assertAllEqual(images.eval(), distorted.eval())

  def test_DiagonalShift(self):
    images = expand_to_4d(
      tf.convert_to_tensor(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        dtype=tf.float32))
    distortions = tf.expand_dims(
      tf.convert_to_tensor(
        [[[1, 1], [1, 1], [1, 1]],
         [[1, 1], [1, 1], [1, 1]],
         [[1, 1], [1, 1], [1, 1]]],
        dtype=tf.float32), 0)
    expected = expand_to_4d(
      tf.convert_to_tensor(
        [[5, 6, 0],
         [8, 9, 0],
         [0, 0, 0]],
        dtype=tf.float32))

    with tf.Session():
      distorted = image_vector_distort.image_vector_distort(
        images=images,
        distortion_fields=distortions)

      self.assertAllEqual(expected.eval(), distorted.eval())

  def test_BilinearInterpolation(self):
    images = expand_to_4d(
      tf.convert_to_tensor(
        [[1, 0],
         [0, 0]],
        dtype=tf.float32))
    distortions = tf.fill([1, 2, 2, 2], -0.5)
    expected = expand_to_4d(
      tf.convert_to_tensor(
        [[0.25, 0.25],
         [0.25, 0.25]]))
    
    with tf.Session():
      distorted = image_vector_distort.image_vector_distort(
        images=images,
        distortion_fields=distortions)

      self.assertAllEqual(expected.eval(), distorted.eval())

if __name__ == '__main__':
  test.main()