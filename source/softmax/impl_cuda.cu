#include "internal.h"
#if defined(IMPL_CUDA)
#include <cuda_runtime.h>

static __global__ void
kernel_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			  const float *__restrict__ input,
			  float *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				   float *inout);

static __global__ void
kernel_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			     const float *__restrict__ input,
			     float *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			  const double *__restrict__ input,
			  double *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				   double *inout);

static __global__ void
kernel_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			     const double *__restrict__ input,
			     double *__restrict__ output);

extern "C" {

void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *__restrict__ input,
			float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

void fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				 float *inout)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_in_place_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, inout);
}

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			   const float *__restrict__ input,
			   float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_accum_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *__restrict__ input,
			double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

void fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				 double *inout)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_in_place_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, inout);
}

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			  const double *__restrict__ input,
			  double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_accum_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, output);
}

}

static __global__ void
kernel_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			  const float *__restrict__ input,
			  float *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		float sum = 0.0f;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += expf(input[k_b * inout_dim + io]);
		}
		size_t idx = k_b * inout_dim + k_io;
		output[idx] = expf(input[idx]) / sum;
	}
}

static __global__ void
kernel_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				   float *inout)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		float sum = 0.0f;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += expf(inout[k_b * inout_dim + io]);
		}
		size_t idx = k_b * inout_dim + k_io;
		inout[idx] = expf(inout[idx]) / sum;
	}
}

static __global__ void
kernel_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			     const float *__restrict__ input,
			     float *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		float sum = 0.0f;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += expf(input[k_b * inout_dim + io]);
		}
		size_t idx = k_b * inout_dim + k_io;
		output[idx] += expf(input[idx]) / sum;
	}
}

static __global__ void
kernel_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			  const double *__restrict__ input,
			  double *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		double sum = 0.0;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += exp(input[k_b * inout_dim + io]);
		}
		size_t idx = k_b * inout_dim + k_io;
		output[idx] = exp(input[idx]) / sum;
	}
}

static __global__ void
kernel_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				   double *inout)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		double sum = 0.0;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += exp(inout[k_b * inout_dim + io]);
		}
		size_t idx = k_b * inout_dim + k_io;
		inout[idx] = exp(inout[idx]) / sum;
	}
}

static __global__ void
kernel_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			     const double *__restrict__ input,
			     double *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		double sum = 0.0;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += exp(input[k_b * inout_dim + io]);
		}
		size_t idx = k_b * inout_dim + k_io;
		output[idx] += exp(input[idx]) / sum;
	}
}
#endif