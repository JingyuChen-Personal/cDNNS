#ifndef CNEURAL_RELU_H
#define CNEURAL_RELU_H
#include "CNeural.h"

#include <stddef.h>

/**
 * Performs forward pass for a single-precision (float32) ReLU layer.
 * Math: Output = max(0, input)
 *
 * @param input      Pointer to input tensor of shape [batch_size, input_dim]
 * @param output     Pointer to output tensor of shape [batch_size, output_dim]
 * @param batch_size Number of samples in the current batch
 * @param inout_dim  Dimensionality of input and output features
 * @return CNeural_Status Execution status code (success/error)
 */
extern CNeural_Status CNeural_ReLU_F32_Forward(
	float* input,
	float* output,
	size_t batch_size, size_t inout_dim
);

/**
 * Performs forward pass for a single-precision (float64) ReLU layer.
 * Math: Output = max(0, input)
 *
 * @param input      Pointer to input tensor of shape [batch_size, input_dim]
 * @param output     Pointer to output tensor of shape [batch_size, output_dim]
 * @param batch_size Number of samples in the current batch
 * @param inout_dim  Dimensionality of input and output features
 * @return CNeural_Status Execution status code (success/error)
 */
extern CNeural_Status CNeural_ReLU_F64_Forward(
	double* input,
	double* output,
	size_t batch_size, size_t inout_dim
);

extern CNeural_Status CNeural_ReLU_F32_Backward(
	float* input,
	float* input_grad,
	float* output_grad,
	size_t batch_size, size_t inout_dim
);

extern CNeural_Status CNeural_ReLU_F64_Backward(
	double* input,
	double* input_grad,
	double* output_grad,
	size_t batch_size, size_t inout_dim
);

#endif

