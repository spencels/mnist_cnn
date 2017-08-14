#define EIGEN_USE_GPU

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/platform/types.h>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h>
//#include <unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h>

#include "image_vector_distort.h"

namespace mnist {

typedef Eigen::GpuDevice GPUDevice;

template struct VectorDistortTransform<GPUDevice, float>;

};