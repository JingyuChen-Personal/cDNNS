#ifndef INTERNAL_H
#define INTERNAL_H
#include <cdnns_linear.h>

#ifdef __cplusplus
extern "C" {
#endif

void fwd_default_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			const float *RESTRICT input,
			const float *RESTRICT weight,
			const float *RESTRICT bias,
			float *RESTRICT output);

void fwd_no_bias_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			const float *RESTRICT input,
			const float *RESTRICT weight,
			float *RESTRICT output);

void fwd_fuse_relu_ow_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			  const float *RESTRICT input,
			  const float *RESTRICT weight,
			  const float *RESTRICT bias,
			  float *RESTRICT output);

void fwd_default_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
		   	   const float *RESTRICT input,
			   const float *RESTRICT weight,
		   	   const float *RESTRICT bias,
			   float *RESTRICT output);

void fwd_no_bias_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			   const float *RESTRICT input,
			   const float *RESTRICT weight,
			   float *RESTRICT output);

void fwd_fuse_relu_accum_f32(size_t batch_size, size_t in_dim, size_t out_dim,
			     const float *RESTRICT input,
			     const float *RESTRICT weight,
			     const float *RESTRICT bias,
			     float *RESTRICT output);

void fwd_default_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			const double *RESTRICT input,
			const double *RESTRICT weight,
			const double *RESTRICT bias,
			double *RESTRICT output);

void fwd_no_bias_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			const double *RESTRICT input,
			const double *RESTRICT weight,
			double *RESTRICT output);

void fwd_fuse_relu_ow_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			  const double *RESTRICT input,
			  const double *RESTRICT weight,
			  const double *RESTRICT bias,
			  double *RESTRICT output);

void fwd_default_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
		   	   const double *RESTRICT input,
			   const double *RESTRICT weight,
		   	   const double *RESTRICT bias,
			   double *RESTRICT output);

void fwd_no_bias_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			   const double *RESTRICT input,
			   const double *RESTRICT weight,
			   double *RESTRICT output);

void fwd_fuse_relu_accum_f64(size_t batch_size, size_t in_dim, size_t out_dim,
			     const double *RESTRICT input,
			     const double *RESTRICT weight,
			     const double *RESTRICT bias,
			     double *RESTRICT output);

#ifdef __cplusplus
}
#endif

#endif

