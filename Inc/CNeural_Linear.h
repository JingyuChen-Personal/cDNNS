#ifndef CNEURAL_LINEAR_H
#define CNEURAL_LINEAR_H

#include "CNeural.h"

#include <stddef.h>
#include <stdint.h>

extern CNeural_Status CNeural_Linear_F32_Forward(
	float* input, float* weights, float* bias, float* output, 
	size_t batch_size, size_t input_dim, size_t output_dim
);

extern CNeural_Status CNeural_Linear_F64_Forward(
	double* input, double* weights, double* bias, double* output, 
	size_t batch_size, size_t input_dim, size_t output_dim
);

extern CNeural_Status CNeural_Linear_F32_Backward(
	float* input, float* weights, 
	float* input_grad, float* weights_grad, float* bias_grad, float* output_grad, 
	size_t batch_size, size_t input_dim, size_t output_dim, 
	CNeural_Reduction reduction
);

extern CNeural_Status CNeural_Linear_F64_Backward(
	double* input, double* weights, 
	double* input_grad, double* weights_grad, double* bias_grad, double* output_grad, 
	size_t batch_size, size_t input_dim, size_t output_dim, 
	CNeural_Reduction reduction
);

#endif
