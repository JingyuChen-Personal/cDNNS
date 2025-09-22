#ifndef CDNNS_TANH_H
#define CDNNS_TANH_H
#include "cdnns.h"

void cdnns_tanh_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
		        size_t batch_size, size_t inout_dim,
		        const void *input, void *output);


/*
void CNeural_tanh_backward_f64(
double *input,
double *input_grad,
double *output_grad,
size_t batch_size, size_t inout_dim
);
*/
#endif
