#ifndef CDNNS_LINEAR_H
#define CDNNS_LINEAR_H

#include <cdnns.h>

void cdnns_linear_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
			  size_t batch_size, size_t in_dim, size_t out_dim,
			  const void *input, const void *weight, const void *bias, void *output);
/*
void cdnns_linear_backward(enum cdnns_type type, enum cdnns_mode mode,
			   enum cdnns_reduction reduction,
			   size_t batch_size, size_t in_dim, size_t out_dim,
			   const void *restrict input,
			   void *restrict input_grad,
			   const void *restrict weight,
			   void *restrict weight_grad,
			   void *restrict bias_grad,
			   const void *restrict output_grad);
*/
#endif
