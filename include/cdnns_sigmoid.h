#ifndef CDNNS_SIGMOID_H
#define CDNNS_SIGMOID_H
#include <cdnns.h>

void cdnns_sigmoid_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
			   size_t batch_size, size_t inout_dim,
		           const void *input, void *output);

/*
void CNeural_sigmoid_backward_f32(
float *input,
float *input_grad,
float *output_grad,
size_t batch_size, size_t inout_dim
);
*/
#endif
