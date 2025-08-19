#include "CNeural_ReLU.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * Performs forward pass for a single-precision (float32) ReLU layer.
 * Math:    Output = max(0, input)
 *
 * @param   input           Pointer to input tensor of shape [batch_size, input_dim]
 * @param   output          Pointer to output tensor of shape [batch_size, output_dim]
 * @param   batch_size      Number of samples in the current batch
 * @param   inout_dim       Dimensionality of input and output features
 * @return  CNeural_Status  Execution status code (success/error)
 */
CNeural_Status CNeural_ReLU_F32_Forward(
    const float* input,
    float* output,
    size_t batch_size, size_t inout_dim
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || output == NULL)
    {
        return CNeural_Error_NullPointer;
    }
    if (batch_size == 0 || inout_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif
    // Perform ReLU activation
    for (size_t io = 0; io < inout_dim; io++)
    {
        for (size_t b = 0; b < batch_size; b++)
        {
            output[io * batch_size + b] += (input[io * batch_size + b] > 0.0f) * input[io * batch_size + b];
        }
    }

    return CNeural_Success;
}

/**
 * Performs forward pass for a double-precision (float64) ReLU layer.
 * Math:    Output = max(0, input)
 *
 * @param   input           Pointer to input tensor of shape [batch_size, input_dim]
 * @param   output          Pointer to output tensor of shape [batch_size, output_dim]
 * @param   batch_size      Number of samples in the current batch
 * @param   inout_dim       Dimensionality of input and output features
 * @return  CNeural_Status  Execution status code (success/error)
 */
CNeural_Status CNeural_ReLU_F64_Forward(
    const double* input,
    double* output,
    size_t batch_size, size_t inout_dim
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || output == NULL)
    {
        return CNeural_Error_NullPointer;
    }
    if (batch_size == 0 || inout_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif
    // Perform ReLU activation
    for (size_t io = 0; io < inout_dim; io++)
    {
        for (size_t b = 0; b < batch_size; b++)
        {
            output[io * batch_size + b] += (input[io * batch_size + b] > 0) * input[io * batch_size + b];
        }
    }

    return CNeural_Success;
}

// NOT TESTED!
CNeural_Status CNeural_ReLU_F32_Backward(
    const float* input,
    float* input_grad,
    const float* output_grad,
    size_t batch_size, size_t inout_dim
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || input_grad == NULL || output_grad == NULL)
    {
        return CNeural_Error_NullPointer;
    }
    if (batch_size == 0 || inout_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif
    // Compute the gradient of the input
    for (size_t io = 0; io < inout_dim; io++)
    {
        for (size_t b = 0; b < batch_size; b++)
        {
            input_grad[io * batch_size + b] += output_grad[io * batch_size + b] * ((input[batch_size + b] > 0.0f) * 1.0f);
        }
    }

    return CNeural_Success;
}

// NOT TESTED!
CNeural_Status CNeural_ReLU_F64_Backward(
    const double* input,
    double* input_grad,
    const double* output_grad,
    size_t batch_size, size_t inout_dim
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || input_grad == NULL || output_grad == NULL)
    {
        return CNeural_Error_NullPointer;
    }
    if (batch_size == 0 || inout_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif
    // Compute the gradient of the input
    for (size_t io = 0; io < inout_dim; io++)
    {
        for (size_t b = 0; b < batch_size; b++)
        {
            input_grad[io * batch_size + b] += output_grad[io * batch_size + b] * ((input[batch_size + b] > 0) * 1);
        }
    }

    return CNeural_Success;
}
