#ifndef INTERNAL_H
#define INTERNAL_H
#include <cdnns_sigmoid.h>

#ifdef __cplusplus
extern "C" {
#endif

void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *RESTRICT input,
			float *RESTRICT output);

void fwd_in_place_ow_f32(size_t batch_size, size_t inout_dim,
			 float *inout);

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
		   	   const float *RESTRICT input,
			   float *RESTRICT output);

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *RESTRICT input,
			double *RESTRICT output);

void fwd_in_place_ow_f64(size_t batch_size, size_t inout_dim,
			 double *inout);

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
			   const double *RESTRICT input,
			   double *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif
