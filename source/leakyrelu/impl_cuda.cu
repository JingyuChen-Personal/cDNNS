#include "internal.h"
#if defined(IMPL_CUDA)
#include <cuda_runtime.h>

static __global__ void
kernel_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			  const float *__restrict__ input,
			  const float *__restrict__ alpha,
			  float *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				   const float *alpha,
				   float *inout);

static __global__ void
kernel_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			     const float *__restrict__ input,
			     const float *__restrict__ alpha,
			     float *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			  const double *__restrict__ input,
			  const double *__restrict__ alpha,
			  double *__restrict__ output);

static __global__ void
kernel_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				   const double *alpha,
				   double *inout);

static __global__ void
kernel_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			     const double *__restrict__ input,
			     const double *__restrict__ alpha,
			     double *__restrict__ output);

extern "C" {

void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *__restrict__ input,
			const float *__restrict__ alpha,
			float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, alpha, output);
}

void fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				 const float *alpha,
				 float *inout)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_in_place_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, alpha, inout);
}

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			   const float *__restrict__ input,
			   const float *__restrict__ alpha,
			   float *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_accum_f32<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, alpha, output);
}

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *__restrict__ input,
			const double *__restrict__ alpha,
			double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, alpha, output);
}

void fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				 const double *alpha,
				 double *inout)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_ow_in_place_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, alpha, inout);
}

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			  const double *__restrict__ input,
			  const double *__restrict__ alpha,
			  double *__restrict__ output)
{
	dim3 block_dim(16, 16);
	dim3 grid_dim((inout_dim - 1) / block_dim.x + 1,
		      (batch_size - 1) / block_dim.y + 1);
	kernel_fwd_default_accum_f64<<<grid_dim, block_dim>>>(
		batch_size, inout_dim, input, alpha, output);
}

}

static __global__ void
kernel_fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			  const float *__restrict__ input,
			  const float *__restrict__ alpha,
			  float *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b  *inout_dim + k_io;
		bool is_greater_than_zero = (input[idx] > 0.0f);
		output[idx] = is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) *
				input[idx];
	}
}

static __global__ void
kernel_fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				   const float *alpha,
				   float *inout)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b  *inout_dim + k_io;
		bool is_greater_than_zero = (inout[idx] > 0.0f);
		inout[idx] = is_greater_than_zero * inout[idx] +
				!is_greater_than_zero * (*alpha) *
				inout[idx];
	}
}

static __global__ void
kernel_fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			     const float *__restrict__ input,
			     const float *__restrict__ alpha,
			     float *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b  *inout_dim + k_io;
		bool is_greater_than_zero = (input[idx] > 0.0f);
		output[idx] += is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) *
				input[idx];
	}
}

static __global__ void
kernel_fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			  const double *__restrict__ input,
			  const double *__restrict__ alpha,
			  double *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b  *inout_dim + k_io;
		bool is_greater_than_zero = (input[idx] > 0.0);
		output[idx] = is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) *
				input[idx];
	}
}

static __global__ void
kernel_fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				   const double *alpha,
				   double *inout)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b  *inout_dim + k_io;
		bool is_greater_than_zero = (inout[idx] > 0.0);
		inout[idx] = is_greater_than_zero * inout[idx] +
				!is_greater_than_zero * (*alpha) *
				inout[idx];
	}
}

static __global__ void
kernel_fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			     const double *__restrict__ input,
			     const double *__restrict__ alpha,
			     double *__restrict__ output)
{
	size_t k_io = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k_b = blockIdx.y * blockDim.y + threadIdx.y;

	if (k_b < batch_size && k_io < inout_dim) {
		size_t idx = k_b  *inout_dim + k_io;
		bool is_greater_than_zero = (input[idx] > 0.0);
		output[idx] += is_greater_than_zero * input[idx] +
				!is_greater_than_zero * (*alpha) *
				input[idx];
	}
}
#endif