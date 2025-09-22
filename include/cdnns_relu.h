#ifndef CDNNS_RELU_H
#define CDNNS_RELU_H

#include "cdnns.h"

void cdnns_relu_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
		        size_t batch_size, size_t inout_dim,
		        const void *input, void *output);


/*
// NOT TESTED!
void CNeural_relu_backward(
	const float *input,
	float *input_grad,
	const float *output_grad,
	size_t batch_size, size_t inout_dim
);
*/
#endif


