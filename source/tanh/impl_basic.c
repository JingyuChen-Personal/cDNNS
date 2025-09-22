#include "internal.h"
#if defined(IMPL_BASIC)
void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *restrict input, float *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			float exp_p = expf(input[idx]);
			float exp_n = expf(-input[idx]);
			output[idx] = (exp_p - exp_n) / (exp_p + exp_n);
		}
	}
}

void fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				 float *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			float exp_p = expf(inout[idx]);
			float exp_n = expf(-inout[idx]);
			inout[idx] = (exp_p - exp_n) / (exp_p + exp_n);
		}
	}
}

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
			   const float *restrict input, float *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			float exp_p = expf(input[idx]);
			float exp_n = expf(-input[idx]);
			output[idx] += (exp_p - exp_n) / (exp_p + exp_n);
		}
	}
}

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *restrict input, double *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			double exp_p = exp(input[idx]);
			double exp_n = exp(-input[idx]);
			output[idx] = (exp_p - exp_n) / (exp_p + exp_n);
		}
	}
}

void fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				 double *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			double exp_p = exp(inout[idx]);
			double exp_n = exp(-inout[idx]);
			inout[idx] = (exp_p - exp_n) / (exp_p + exp_n);
		}
	}
}

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			   const double *restrict input,
			   double *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b * inout_dim + io;
			double exp_p = exp(input[idx]);
			double exp_n = exp(-input[idx]);
			output[idx] += (exp_p - exp_n) / (exp_p + exp_n);
		}
	}
}
#endif

