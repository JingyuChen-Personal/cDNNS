#ifndef CNEURAL_CONV2D_H
#define CNEURAL_CONV2D_H

#include "CNeural.h"

enum CNeural_padding_mode
{
CNeural_PADDING_MODE_ZERO
};

/**
  *Performs forward pass for a single-precision (float32) 2D convolution operation.
 *
  *@param   input   Pointer to input tensor of shape [batch_size, in_channels, in_height, in_width]
  *@param   weight  Pointer to weight tensor of shape [out_channels, in_channels / groups, kernel_height, kernel_width]
  *@param   biasPointer to bias tensor of shape [out_channels]
  *@param   output  Pointer to output tensor of shape [batch_size, out_channels, out_height, out_width]
  *@param   batch_size  Number of samples in the current batch
  *@param   in_channels Number of input channels
  *@param   out_channelsNumber of output channels
  *@param   in_height   Height of input feature maps
  *@param   in_widthWidth of input feature maps
  *@param   kernel_height   Height of convolution kernel
  *@param   kernel_widthWidth of convolution kernel
  *@param   stride_hVertical stride of convolution
  *@param   stride_wHorizontal stride of convolution
  *@param   dilation_h  Vertical dilation rate of convolution kernel
  *@param   dilation_w  Horizontal dilation rate of convolution kernel
  *@param   groups  Number of groups for grouped convolution
  *@return  CNeural_status  Execution status code (success/error)
 */
void CNeural_conv2d_forward_f32(
	const float *input,
	const float *weight,
	const float *bias,
	float *output,
	size_t batch_size,
	size_t in_channels, size_t out_channels,
	size_t in_height, size_t in_width,
	size_t kernel_height, size_t kernel_width,
	size_t stride_h, size_t stride_w,
	size_t dilation_h, size_t dilation_w,
	size_t groups
);

void CNeural_conv2d_forward_f64(
	const double *input,
	const double *weight,
	const double *bias,
	double *output,
	size_t batch_size,
	size_t in_channels, size_t out_channels,
	size_t in_height, size_t in_width,
	size_t kernel_height, size_t kernel_width,
	size_t stride_h, size_t stride_w,
	size_t dilation_h, size_t dilation_w,
	size_t groups
);

void CNeural_conv2d_backward_f32(float *value, float *label, float *input_grad, size_t batch_size, size_t in_dim, float *loss);

void CNeural_conv2d_backward_f64(double *value, double *label, double *input_grad, size_t batch_size, size_t in_dim, double *loss);

#endif#
