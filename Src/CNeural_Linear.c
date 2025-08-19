#include "CNeural_Linear.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * Performs forward pass for a single-precision (float32) linear layer.
 * Math:    Output = input * weights_T + bias
 *
 * @param   input           Pointer to input tensor of shape [batch_size, input_dim]
 * @param   weights         Pointer to weight tensor of shape [output_dim, input_dim]
 * @param   bias            Pointer to bias tensor of shape [output_dim]
 * @param   output          Pointer to output tensor of shape [batch_size, output_dim]
 * @param   batch_size      Number of samples in the current batch
 * @param   input_dim       Dimensionality of input features
 * @param   output_dim      Dimensionality of output features
 * @return  CNeural_Status  Execution status code (success/error)
 */
CNeural_Status CNeural_Linear_F32_Forward(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t batch_size, size_t input_dim, size_t output_dim
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || weights == NULL || bias == NULL || output == NULL)
    {
        return CNeural_Error_NullPointer;
    }
    if (batch_size == 0 || input_dim == 0 || output_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif

#if defined USE_CBLAS_API
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans, CblasTrans,
        batch_size, output_dim, input_dim,
        1.0f,
        input, input_dim,
        weights, input_dim,
        0.0f,
        output, output_dim
    );

    for (size_t b = 0; b < batch_size; b++)
    {
        cblas_saxpy(
            output_dim,
            1.0f,
            bias, 1,
            output + b * output_dim, 1
        );
    }
#else
    // Perform matrix multiplication and add bias
    float* input_iter   = input;
    float* weights_iter = weights;
    float* bias_iter    = bias;
    float* output_iter  = output;
    float  output_reg;
    for (size_t b = 0; b < batch_size; b++)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            output_reg = 0;
            for (size_t i = 0; i < input_dim; i++)
            {
                output_reg += input_iter[i] * (*weights_iter);

                weights_iter++;
            }
            output_reg   += *bias_iter;;
            *output_iter += output_reg;

            bias_iter++;
            output_iter++;
        }
        input_iter  += input_dim;
        weights_iter = weights;
        bias_iter    = bias;
    }
#endif

    return CNeural_Success;
}


/**
 * Performs forward pass for a double-precision (float64) linear layer.
 * Math:    Output = input * weights_T + bias
 *
 * @param   input           Pointer to input tensor of shape [batch_size, input_dim]
 * @param   weights         Pointer to weight matrix of shape [output_dim, input_dim]
 * @param   bias            Pointer to bias vector of shape [output_dim]
 * @param   output          Pointer to output tensor of shape [batch_size, output_dim]
 * @param   batch_size      Number of samples in the current batch
 * @param   input_dim       Dimensionality of input features
 * @param   output_dim      Dimensionality of output features
 * @return  CNeural_Status  Execution status code (success/error)
 */
CNeural_Status CNeural_Linear_F64_Forward(
    const double* input,
    const double* weights,
    const double* bias,
    double* output,
    size_t batch_size,  size_t input_dim, size_t output_dim
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || weights == NULL || bias == NULL || output == NULL)
    {
        return CNeural_Error_NullPointer;
    }
    if (batch_size == 0 || input_dim == 0 || output_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif

#if defined USE_CBLAS_API
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, CblasTrans,
        batch_size, output_dim, input_dim,
        1.0,
        input, input_dim,
        weights, input_dim,
        0.0,
        output, output_dim
    );

    for (size_t b = 0; b < batch_size; b++)
    {
        cblas_daxpy(
            output_dim,
            1.0,
            bias, 1,
            output + b * output_dim, 1
        );
    }
#else
    // Perform matrix multiplication and add bias
    double* input_iter   = input;
    double* weights_iter = weights;
    double* bias_iter    = bias;
    double* output_iter  = output;
    double  output_reg;
    for (size_t b = 0; b < batch_size; b++)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            output_reg = 0;
            for (size_t i = 0; i < input_dim; i++)
            {
                output_reg += input_iter[i] * (*weights_iter);

                weights_iter++;
            }
            output_reg   += *bias_iter;;
            *output_iter += output_reg;

            bias_iter++;
            output_iter++;
        }
        input_iter  += input_dim;
        weights_iter = weights;
        bias_iter    = bias;
    }
#endif

    return CNeural_Success;
}

// NOT TESTED!
CNeural_Status CNeural_Linear_F32_Backward(
    const float* input,
    const float* weights,
    float* input_grad,
    float* weights_grad,
    float* bias_grad,
    const float* output_grad,
    size_t batch_size, size_t input_dim, size_t output_dim,
    CNeural_Reduction reduction
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || weights == NULL || input_grad == NULL || weights_grad == NULL || bias_grad == NULL || output_grad == NULL)
    {
        return CNeural_Error_NullPointer;
    }

    if (batch_size == 0 || input_dim == 0 || output_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif
    // Compute the gradient of the input
    for (size_t i = 0; i < input_dim; i++)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t b = 0; b < batch_size; b++)
            {
                input_grad[i * batch_size + b] += output_grad[o * batch_size + b] * weights[o * input_dim + i];
            }
        }
    }

    // Compute the gradient of the weights
    if (reduction == CNeural_Reduction_Mean)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t i = 0; i < input_dim; i++)
            {
                weights_grad[o * input_dim + i] = 0.0f;
                for (size_t b = 0; b < batch_size; b++)
                {
                    weights_grad[o * input_dim + i] += output_grad[o * batch_size + b] * input[i * batch_size + b];
                }
            }
        }
    }
    else if (reduction == CNeural_Reduction_Sum)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t i = 0; i < input_dim; i++)
            {
                weights_grad[o * input_dim + i] = 0.0f;
                for (size_t b = 0; b < batch_size; b++)
                {
                    weights_grad[o * input_dim + i] += output_grad[o * batch_size + b] * input[i * batch_size + b];
                }
                weights_grad[o * input_dim + i] /= batch_size;
            }
        }
    }
    else if (CNeural_Reduction_None)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t i = 0; i < input_dim; i++)
            {
                for (size_t b = 0; b < batch_size; b++)
                {
                    weights_grad[(o * input_dim + i) * batch_size + b] = output_grad[o * batch_size + b] * input[i * batch_size + b];
                }
            }
        }
    }
    else
    {
        return CNeural_Error_InvalidParameter;
    }

    // Compute the gradient of the bias
    if (reduction == CNeural_Reduction_Mean)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            bias_grad[o] = 0.0f;
            for (size_t b = 0; b < batch_size; b++)
            {
                bias_grad[o] += output_grad[o * batch_size + b];
            }
            bias_grad[o] /= batch_size;
        }
    }
    else if (reduction == CNeural_Reduction_Sum)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            bias_grad[o] = 0.0f;
            for (size_t b = 0; b < batch_size; b++)
            {
                bias_grad[o] += output_grad[o * batch_size + b];
            }
        }
    }
    else if (CNeural_Reduction_None)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t b = 0; b < batch_size; b++)
            {
                bias_grad[o * batch_size + b] = output_grad[o * batch_size + b];
            }
        }
    }
    else
    {
        return CNeural_Error_InvalidParameter;
    }

    return CNeural_Success;
}

// NOT TESTED!
CNeural_Status CNeural_Linear_F64_Backward(
    const double* input,
    const double* weights,
    double* input_grad,
    double* weights_grad,
    double* bias_grad,
    const double* output_grad,
    size_t batch_size, size_t input_dim, size_t output_dim,
    CNeural_Reduction reduction
)
{
#ifndef ONLY_NECCESSARY_CHECK
    if (input == NULL || weights == NULL || input_grad == NULL || weights_grad == NULL || bias_grad == NULL || output_grad == NULL)
    {
        return CNeural_Error_NullPointer;
    }

    if (batch_size == 0 || input_dim == 0 || output_dim == 0)
    {
        return CNeural_Error_InvalidParameter;
    }
#endif
    // Compute the gradient of the input
    for (size_t i = 0; i < input_dim; i++)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t b = 0; b < batch_size; b++)
            {
                input_grad[i * batch_size + b] += output_grad[o * batch_size + b] * weights[o * input_dim + i];
            }
        }
    }

    // Compute the gradient of the weights
    if (reduction == CNeural_Reduction_Mean)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t i = 0; i < input_dim; i++)
            {
                weights_grad[o * input_dim + i] = 0;
                for (size_t b = 0; b < batch_size; b++)
                {
                    weights_grad[o * input_dim + i] += output_grad[o * batch_size + b] * input[i * batch_size + b];
                }
            }
        }
    }
    else if (reduction == CNeural_Reduction_Sum)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t i = 0; i < input_dim; i++)
            {
                weights_grad[o * input_dim + i] = 0;
                for (size_t b = 0; b < batch_size; b++)
                {
                    weights_grad[o * input_dim + i] += output_grad[o * batch_size + b] * input[i * batch_size + b];
                }
                weights_grad[o * input_dim + i] /= batch_size;
            }
        }
    }
    else if (CNeural_Reduction_None)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t i = 0; i < input_dim; i++)
            {
                for (size_t b = 0; b < batch_size; b++)
                {
                    weights_grad[(o * input_dim + i) * batch_size + b] = output_grad[o * batch_size + b] * input[i * batch_size + b];
                }
            }
        }
    }
    else
    {
        return CNeural_Error_InvalidParameter;
    }


    // Compute the gradient of the bias
    if (reduction == CNeural_Reduction_Mean)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            bias_grad[o] = 0;
            for (size_t b = 0; b < batch_size; b++)
            {
                bias_grad[o] += output_grad[o * batch_size + b];
            }
            bias_grad[o] /= batch_size;
        }
    }
    else if (reduction == CNeural_Reduction_Sum)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            bias_grad[o] = 0;
            for (size_t b = 0; b < batch_size; b++)
            {
                bias_grad[o] += output_grad[o * batch_size + b];
            }
        }
    }
    else if (CNeural_Reduction_None)
    {
        for (size_t o = 0; o < output_dim; o++)
        {
            for (size_t b = 0; b < batch_size; b++)
            {
                bias_grad[o * batch_size + b] = output_grad[o * batch_size + b];
            }
        }
    }
    else
    {
        return CNeural_Error_InvalidParameter;
    }

    return CNeural_Success;
}