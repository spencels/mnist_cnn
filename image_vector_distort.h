#pragma once

#include <iostream>

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include <unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h>

#define EIGEN_USE_THREADS

namespace mnist {

// Define eigen generator: shared kernel code for CPU/GPU.
// T is tensor type.
template <typename Device, typename T>
class ImageVectorDistortGenerator {
public:
  typedef typename tensorflow::TTypes<T, 4>::ConstTensor InputType;

  EIGEN_DEVICE_FUNC
  ImageVectorDistortGenerator(InputType input, InputType distortion_field)
    : input_(input), distortion_field_(distortion_field) {}

  EIGEN_DEVICE_FUNC
  T operator()(const Eigen::array<Eigen::DenseIndex, 4>& coords) const {
    using tensorflow::int64;
    
    const int64 output_y = coords[1];
    const int64 output_x = coords[2];
    
    // Fetch transforms.
    size_t index = distortion_field_.dimension(3) * coords[2];
    index += distortion_field_.dimension(2) * distortion_field_.dimension(3)
      * coords[1];
    index += distortion_field_.dimension(1) * distortion_field_.dimension(2)
      * distortion_field_.dimension(3) * coords[0];
    const float* distortion = &distortion_field_.data()[index];

    const float input_x = distortion[0] + output_x;
    const float input_y = distortion[1] + output_y;

    auto val = bilinear_interpolate(coords[0], input_y, input_x);
    return val;
  }

private:
  InputType input_;
  InputType distortion_field_;

  // Copied from
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/image/kernels/image_ops.h
  // See tensorflow license in pip package.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  T bilinear_interpolate(
      const Eigen::DenseIndex batch,
      float y,
      float x) const {
    const float y_floor = std::floor(y);
    const float x_floor = std::floor(x);
    const float y_ceil = y_floor + 1;
    const float x_ceil = x_floor + 1;
    const float value_yfloor =
        (x_ceil - x) * read_with_fill_value(batch, Eigen::DenseIndex(y_floor),
                                            Eigen::DenseIndex(x_floor),
                                            0.0f) +
        (x - x_floor) * read_with_fill_value(batch, Eigen::DenseIndex(y_floor),
                                             Eigen::DenseIndex(x_ceil),
                                             0.0f);
    const float value_yceil =
        (x_ceil - x) * read_with_fill_value(batch, Eigen::DenseIndex(y_ceil),
                                            Eigen::DenseIndex(x_floor),
                                            0.0f) +
        (x - x_floor) * read_with_fill_value(batch, Eigen::DenseIndex(y_ceil),
                                             Eigen::DenseIndex(x_ceil),
                                             0.0f);
    return T((y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil); 
  }

  // Copied from
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/image/kernels/image_ops.h
  // See tensorflow license in pip package.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T read_with_fill_value(
      const Eigen::DenseIndex batch, const Eigen::DenseIndex y,
      const Eigen::DenseIndex x, const T fill_value) const {
    // batch and channel must be correct, because they are passed unchanged from
    // the input.
    return (0 <= y && y < input_.dimension(1) && 0 <= x &&
            x < input_.dimension(2))
               ? input_(Eigen::array<Eigen::DenseIndex, 4>{batch, y, x, 0})
               : fill_value;
  }
};

template <typename Device, typename T>
struct VectorDistortTransform {
  typedef typename tensorflow::TTypes<T, 4>::ConstTensor InputType;
  typedef typename tensorflow::TTypes<T, 4>::Tensor OutputType;

  void operator()(
      const Device& device, OutputType* output,
      const InputType& images, const InputType& vector_field) const {
    // Call device kernel.
    output->device(device) = images.generate(
      ImageVectorDistortGenerator<Device, T>(images, vector_field));
  }
};

};  // namespace mnist