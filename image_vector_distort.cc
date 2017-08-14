// Tensorflow op for implementing

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/platform/types.h> 
#include <unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h>

#include "image_vector_distort.h"


using Eigen::DenseIndex;
using tensorflow::DEVICE_CPU;
using tensorflow::DEVICE_GPU;
using tensorflow::errors::InvalidArgument;
using tensorflow::int64;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::Status;
using tensorflow::Tensor;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace mnist {

// 1. Register op.
REGISTER_OP("ImageVectorDistort")
  .Attr("T: {float}")
  .Input("images: T")
  .Input("distortion_fields: T")
  .Output("distorted: T")
  .SetShapeFn([](InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

// 2. Define operation.
template <typename Device, typename T>
class ImageVectorDistortOp : public OpKernel {
public:
  explicit ImageVectorDistortOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get and validate input.
    const Tensor& images = context->input(0);
    const Tensor& distortion_fields = context->input(1);
    OP_REQUIRES(
      context, images.shape().dims() == 4,
      InvalidArgument("Input images must have rank 4"));
    OP_REQUIRES(
      context,
      images.dim_size(0) == distortion_fields.dim_size(0)
      && images.dim_size(1) == distortion_fields.dim_size(1)
      && images.dim_size(2) == distortion_fields.dim_size(2),
      InvalidArgument(
        "First 3 dimensions of distortion_field must match images."));

    // Allocate output tensor.
    Tensor* output_tensor;
    OP_REQUIRES_OK(
      context, context->allocate_output(0, images.shape(), &output_tensor));
    auto images_tensor = images.tensor<T, 4>();
    auto distortion_tensor = distortion_fields.tensor<T, 4>();
    auto output = output_tensor->tensor<T, 4>();
    (VectorDistortTransform<Device, T>()(
      context->eigen_device<Device>(), &output, images_tensor,
      distortion_tensor));
  }
};

// Declare kernels.
template struct VectorDistortTransform<CPUDevice, float>;
template <>
void VectorDistortTransform<GPUDevice, float>::operator()(
      const GPUDevice& device, OutputType* output, const InputType& images,
      const InputType& vector_field) const;
extern template struct VectorDistortTransform<GPUDevice, float>;

// Register ops.
#define REGISTER_CPU(TYPE) \
  REGISTER_KERNEL_BUILDER( \
    Name("ImageVectorDistort") \
      .Device(DEVICE_CPU) \
      .template TypeConstraint<TYPE>("T"), \
    ImageVectorDistortOp<CPUDevice, TYPE>)

REGISTER_CPU(float);

#define REGISTER_GPU(TYPE) \
  REGISTER_KERNEL_BUILDER( \
    Name("ImageVectorDistort") \
      .Device(DEVICE_GPU) \
      .template TypeConstraint<TYPE>("T"), \
    ImageVectorDistortOp<GPUDevice, TYPE>)
    
REGISTER_GPU(float);

};  // namespace mnist