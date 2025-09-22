#include "internal.h"
#if defined(IMPL_BASIC)
void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *restrict input, float *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		float sum = 0.0f;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += expf(input[b * inout_dim + io]);
		}
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			output[idx] = expf(input[idx]) / sum;
		}
	}
}

void fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				 float *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
		float sum = 0.0f;
		for (size_t io = 0; io < inout_dim; io++) {
			sum = expf(inout[b * inout_dim + io]);
		}
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			inout[idx] = expf(inout[idx]) / sum;
		}
	}
}

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
		   	   const float *restrict input, float *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		float sum = 0.0f;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += expf(input[b * inout_dim + io]);
		}
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			output[idx] += expf(input[idx]) / sum;
		}
	}
}

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *restrict input, double *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		double sum = 0.0;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += exp(input[b * inout_dim + io]);
		}
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			output[idx] = exp(input[idx]) / sum;
		}
	}
}

void fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				 double *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
		double sum = 0.0;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += exp(inout[b * inout_dim + io]);
		}
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			inout[idx] = exp(inout[idx]) / sum;
		}
	}
}

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			   const double *restrict input, double *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		double sum = 0.0;
		for (size_t io = 0; io < inout_dim; io++) {
			sum += exp(input[b * inout_dim + io]);
		}
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			output[idx] += exp(input[idx]) / sum;
		}
	}
}
#endif

