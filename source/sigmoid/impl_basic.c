#include "internal.h"
#if defined(IMPL_BASIC)
void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *restrict input, float *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b  *inout_dim + io;
			output[idx] = 1.0f / (1.0f + expf(-input[idx]));
		}
	}
}

void fwd_default_ow_in_place_f32(size_t batch_size, size_t inout_dim,
				 float *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b  *inout_dim + io;
			inout[idx] = 1.0f / (1.0f + expf(-inout[idx]));
		}
	}
}

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
		   	   const float *restrict input, float *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b  *inout_dim + io;
			output[idx] += 1.0f / (1.0f + expf(-input[idx]));
		}
	}
}

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *restrict input, double *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b  *inout_dim + io;
			output[idx] = 1.0 / (1.0 + exp(-input[idx]));
		}
	}
}

void fwd_default_ow_in_place_f64(size_t batch_size, size_t inout_dim,
				 double *inout)
{
	for (size_t b = 0; b < batch_size; b++) {
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b  *inout_dim + io;
			inout[idx] = 1.0 / (1.0 + exp(-inout[idx]));
		}
	}
}

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			   const double *restrict input,
			   double *restrict output)
{
	for (size_t b = 0; b < batch_size; b++) {
	 	for (size_t io = 0; io < inout_dim; io++) {
			size_t idx = b  *inout_dim + io;
			output[idx] += 1.0 / (1.0 + exp(-input[idx]));
		}
	}
}
#endif

