#ifndef CDNNS_LEAKYRELU_H
#define CDNNS_LEAKYRELU_H

#include "cdnns.h"

void cdnns_leakyrelu_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
		       	     size_t batch_size, size_t inout_dim,
		             const void *input, const void* alpha, void *output);

/*
void CNeural_leakyrelu_backward_f64(
double *input,
double *input_grad,
double *output_grad,
size_t batch_size, size_t inout_dim,
double alpha
);
*/
#endif