#include "internal.h"
#if defined(IMPL_CUDA)
#include <cuda_runtime.h>

static __global__ void
kernel_fwd_default_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
		  const float *__restrict__ input,
		  const float *__restrict__ weight,
		  const float *__restrict__ bias,
		  float *__restrict__ output);

static __global__ void
kernel_fwd_no_bias_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			  const float *__restrict__ input,
			  const float *__restrict__ weight,
			  float *__restrict__ output);

static __global__ void
kernel_fwd_fuse_relu_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			    const float *__restrict__ input,
			    const float *__restrict__ weight,
			    const float *__restrict__ bias,
			    float *__restrict__ output);

static __global__ void
kernel_fwd_default_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
		     const float *__restrict__ input,
		     const float *__restrict__ weight,
		     const float *__restrict__ bias,
		     float *__restrict__ output);

static __global__ void
kernel_fwd_no_bias_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			     const float *__restrict__ input,
			     const float *__restrict__ weight,
			     float *__restrict__ output);

static __global__ void
kernel_fwd_fuse_relu_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			       const float *__restrict__ input,
			       const float *__restrict__ weight,
			       const float *__restrict__ bias,
			       float *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
		  	  const double *__restrict__ input,
		  	  const double *__restrict__ weight,
		  	  const double *__restrict__ bias,
		  	  double *__restrict__ output);

static __global__ void
kernel_fwd_no_bias_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			  const double *__restrict__ input,
			  const double *__restrict__ weight,
			  double *__restrict__ output);

static __global__ void
kernel_fwd_fuse_relu_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			    const double *__restrict__ input,
			    const double *__restrict__ weight,
			    const double *__restrict__ bias,
			    double *__restrict__ output);

static __global__ void
kernel_fwd_default_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			     const double *__restrict__ input,
		     	     const double *__restrict__ weight,
		     	     const double *__restrict__ bias,
		     	     double *__restrict__ output);

static __global__ void
kernel_fwd_no_bias_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			     const double *__restrict__ input,
			     const double *__restrict__ weight,
			     double *__restrict__ output);

static __global__ void
kernel_fwd_fuse_relu_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			       const double *__restrict__ input,
			       const double *__restrict__ weight,
			       const double *__restrict__ bias,
			       double *__restrict__ output);

extern "C" {

void fwd_default_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			const float *__restrict__ input,
			const float *__restrict__ weight,
			const float *__restrict__ bias,
			float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_f32<<<grid_dim, block_dim>>>(batch_size, in_dim, out_dim,
			  			   input, weight, bias, output);
}

void fwd_no_bias_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			const float *__restrict__ input,
			const float *__restrict__ weight,
			float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_no_bias_ow_f32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, output);
}

void fwd_fuse_relu_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			  const float *__restrict__ input,
			  const float *__restrict__ weight,
			  const float *__restrict__ bias,
			  float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_fuse_relu_ow_f32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, bias, output);
}

void fwd_default_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
		   	   const float *__restrict__ input,
		   	   const float *__restrict__ weight,
		   	   const float *__restrict__ bias,
		   	   float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_accum_f32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, bias, output);
}

void fwd_no_bias_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			   const float *__restrict__ input,
			   const float *__restrict__ weight,
			   float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_no_bias_accum_f32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, output);
}

void fwd_fuse_relu_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			     const float *__restrict__ input,
			     const float *__restrict__ weight,
			     const float *__restrict__ bias,
			     float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_fuse_relu_accum_f32<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, bias, output);
}

void fwd_default_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			const double *__restrict__ input,
			const double *__restrict__ weight,
			const double *__restrict__ bias,
			double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_f64<<<grid_dim, block_dim>>>(batch_size, in_dim, out_dim,
			  			   input, weight, bias, output);
}

void fwd_no_bias_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			const double *__restrict__ input,
			const double *__restrict__ weight,
			double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_no_bias_ow_f64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, output);
}

void fwd_fuse_relu_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			  const double *__restrict__ input,
			  const double *__restrict__ weight,
			  const double *__restrict__ bias,
			  double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_fuse_relu_ow_f64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, bias, output);
}

void fwd_default_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
		   	   const double *__restrict__ input,
		   	   const double *__restrict__ weight,
		   	   const double *__restrict__ bias,
		   	   double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_accum_f64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, bias, output);
}

void fwd_no_bias_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			   const double *__restrict__ input,
			   const double *__restrict__ weight,
			   double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_no_bias_accum_f64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, output);
}

void fwd_fuse_relu_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			     const double *__restrict__ input,
			     const double *__restrict__ weight,
			     const double *__restrict__ bias,
			     double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((out_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_fuse_relu_accum_f64<<<grid_dim, block_dim>>>(
		batch_size, in_dim, out_dim, input, weight, bias, output);
}

}

static __global__ void
kernel_fwd_default_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
		  	  const float *__restrict__ input,
			  const float *__restrict__ weight,
		  	  const float *__restrict__ bias,
		  	  float *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] = sum;
	}
}

static __global__ void
kernel_fwd_no_bias_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			  const float *__restrict__ input,
			  const float *__restrict__ weight,
			  float *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = 0.0f;
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] = sum;
	}
}

static __global__ void
kernel_fwd_fuse_relu_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			    const float *__restrict__ input,
			    const float *__restrict__ weight,
			    const float *__restrict__ bias,
			    float *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] = sum;
	}
}

static __global__ void
kernel_fwd_default_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
		     	     const float *__restrict__ input,
		     	     const float *__restrict__ weight,
		     	     const float *__restrict__ bias,
		     	     float *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] += sum;
	}
}

static __global__ void
kernel_fwd_no_bias_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			     const float *__restrict__ input,
			     const float *__restrict__ weight,
			     float *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = 0.0f;
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] += sum;
	}
}

static __global__ void
kernel_fwd_fuse_relu_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			       const float *__restrict__ input,
			       const float *__restrict__ weight,
			       const float *__restrict__ bias,
			       float *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		float sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] += sum;
	}
}

static __global__ void
kernel_fwd_default_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			  const double *__restrict__ input,
		  	  const double *__restrict__ weight,
		  	  const double *__restrict__ bias,
		  	  double *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] = sum;
	}
}

static __global__ void
kernel_fwd_no_bias_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			  const double *__restrict__ input,
			  const double *__restrict__ weight,
			  double *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = 0.0;
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] = sum;
	}
}

static __global__ void
kernel_fwd_fuse_relu_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			    const double *__restrict__ input,
			    const double *__restrict__ weight,
			    const double *__restrict__ bias,
			    double *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		sum = (sum > 0.0) * sum;
		output[b * out_dim + o] = sum;
	}
}

static __global__ void
kernel_fwd_default_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			     const double *__restrict__ input,
			     const double *__restrict__ weight,
		     	     const double *__restrict__ bias,
		     	     double *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] += sum;
	}
}

static __global__ void
kernel_fwd_no_bias_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			     const double *__restrict__ input,
			     const double *__restrict__ weight,
			     double *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = 0;
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		output[b * out_dim + o] += sum;
	}
}

static __global__ void
kernel_fwd_fuse_relu_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			       const double *__restrict__ input,
			       const double *__restrict__ weight,
			       const double *__restrict__ bias,
			       double *__restrict__ output)
{
	size_t o = blockIdx.x * blockDim.x + threadIdx.x;
	size_t b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b < batch_size && o < out_dim) {
		double sum = bias[o];
		for (size_t i = 0; i < in_dim; i++) {
			sum += input[b * in_dim + i] * weight[o * in_dim + i];
		}
		sum = (sum > 0.0f) * sum;
		output[b * out_dim + o] += sum;
	}
}
#endif