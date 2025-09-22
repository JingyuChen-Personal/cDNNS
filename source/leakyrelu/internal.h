#ifndef INTERNAL_H
#define INTERNAL_H
#include <cdnns_softmax.h>

#ifdef __cplusplus
extern "C" {
#endif

void fwd_default_ow_f32(size_t batch_size, size_t inout_dim,
			const float *RESTRICT input,
			const float *RESTRICT alpha,
			float *RESTRICT output);

void fwd_in_place_ow_f32(size_t batch_size, size_t inout_dim,
			 const float *alpha, float *inout);

void fwd_default_accum_f32(size_t batch_size, size_t inout_dim,
		   	   const float *RESTRICT input,
			   const float *RESTRICT alpha,
		   	   float *RESTRICT output);

void fwd_default_ow_f64(size_t batch_size, size_t inout_dim,
			const double *RESTRICT input,
			const double *RESTRICT alpha,
			double *RESTRICT output);

void fwd_in_place_ow_f64(size_t batch_size, size_t inout_dim,
			 const float *alpha, double *inout);

void fwd_default_accum_f64(size_t batch_size, size_t inout_dim,
		   	   const double *RESTRICT input,
			   const double *RESTRICT alpha,
		   	   double *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif
