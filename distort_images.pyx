# Sampling 
from libc cimport math
from libc cimport stdio

from cython cimport parallel
cimport cython.view.array as cvarray 
import numpy as np
cimport numpy as np
cimport cython
cimport openmp


@cython.boundscheck(False)
cdef unsigned char sample(
    unsigned char[:, :, :] images,
    int image, int x, int y) nogil:
  """Sample a pixel from given coordinates.
  
  Performs out-of-bounds checks.
  """
  cdef int x_in_range = x >= 0 and x < images.shape[2]
  cdef int y_in_range = y >= 0 and y < images.shape[1]
  cdef int index
  if x_in_range and y_in_range:
    return images[image, y, x]
  else:
    return 0


@cython.boundscheck(False)
cdef inline void distort_image(
    unsigned char[:, :, :] images,
    float[:, :, :, :] distortions,
    int image,
    unsigned char[:, :, :] output) nogil:
  """Applies distortion for single image."""
  cdef int x, y, x_low, x_high, y_low, y_high, index
  cdef int distortion_index
  cdef float x_new, y_new, high_sample, low_sample, pixel

  for y in range(images.shape[1]):
    for x in range(images.shape[2]):
      x_new = distortions[image, y, x, 0] + x
      x_low = <int>x_new  # Floor
      x_high = x_low + 1
      y_new = distortions[image, y, x, 1] + y
      y_low = <int>y_new  # Floor
      y_high = y_low + 1

      high_sample = (
        (1.0 - (x_new - x_low)) * sample(images, image, x_low, y_high) +
        (1.0 - (x_high - x_new)) * sample(images, image, x_high, y_high))
      low_sample = (
        (1.0 - (x_new - x_low)) * sample(images, image, x_low, y_low) +
        (1.0 - (x_high - x_new)) * sample(images, image, x_high, y_low))

      pixel = math.round(
        (y_high - y_new) * low_sample + (y_new - y_low) * high_sample)
      output[image, y, x] = <unsigned char>pixel


@cython.boundscheck(False)
def distort_images_impl(
    np.ndarray images not None,
    np.ndarray distortions not None):
  """Distort images according to vector field distortions."""
  shape = images.shape
  output = np.empty((shape[0], shape[1], shape[2]), images.dtype)

  for i in range(images.shape[0]):
    distort_image(images, distortions, i, output)

  return output


def distort_images(np.ndarray images, np.ndarray distortions):
  return distort_images_impl(images, distortions)