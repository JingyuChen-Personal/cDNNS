#ifndef CNEURAL_LINEAR_H
#define CNEURAL_LINEAR_H
#include "CNeural.h"

#include <stddef.h>

/**
 * Performs forward pass for a single-precision (float32) linear layer.
 * Math: Output = input * weights_T + bias
 *
 * @param input      Pointer to input tensor of shape [batch_size, input_dim]
 * @param weights    Pointer to weight tensor of shape [output_dim, input_dim]
 * @param bias       Pointer to bias tensor of shape [output_dim]
 * @param output     Pointer to output tensor of shape [batch_size, output_dim]
 * @param batch_size Number of samples in the current batch
 * @param input_dim  Dimensionality of input features
 * @param output_dim Dimensionality of output features
 * @return CNeural_Status Execution status code (success/error)
 */
extern CNeural_Status CNeural_Linear_F32_Forward(
	const float* input,
	const float* weights,
	const float* bias,
	float* output, 
	size_t batch_size, size_t input_dim, size_t output_dim
);

/**
 * Performs forward pass for a double-precision (float64) linear layer.
 * Math: Output = input * weights_T + bias
 *
 * @param input      Pointer to input tensor of shape [batch_size, input_dim]
 * @param weights    Pointer to weight matrix of shape [output_dim, input_dim]
 * @param bias       Pointer to bias vector of shape [output_dim]
 * @param output     Pointer to output tensor of shape [batch_size, output_dim]
 * @param batch_size Number of samples in the current batch
 * @param input_dim  Dimensionality of input features
 * @param output_dim Dimensionality of output features
 * @return CNeural_Status Execution status code (success/error)
 */
extern CNeural_Status CNeural_Linear_F64_Forward(
	const double* input,
	const double* weights,
	const double* bias,
	double* output, 
	size_t batch_size, size_t input_dim, size_t output_dim
);

extern CNeural_Status CNeural_Linear_F32_Backward(
	const float* input,
	const float* weights,
	float* input_grad,
	float* weights_grad,
	float* bias_grad,
	const float* output_grad,
	size_t batch_size, size_t input_dim, size_t output_dim, 
	CNeural_Reduction reduction
);

extern CNeural_Status CNeural_Linear_F64_Backward(
	const double* input,
	const double* weights,
	double* input_grad,
	double* weights_grad,
	double* bias_grad,
	const double* output_grad,
	size_t batch_size, size_t input_dim, size_t output_dim, 
	CNeural_Reduction reduction
);

#endif
