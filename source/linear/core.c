#include "internal.h"

void cdnns_linear_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
			  size_t batch_size, size_t in_dim, size_t out_dim,
			  const void *input, const void *weight, const void *bias, void *output)
{
	switch (type) {
	case CDNNS_TYPE_FP32:

		switch(mode) {
		case CDNNS_MODE_OVERWRITE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_ow_f32(batch_size, in_dim, out_dim,
						   input, weight, bias, output);
				break;
			case CDNNS_OPTION_NO_BIAS:
				fwd_no_bias_ow_f32(batch_size, in_dim, out_dim,
						   input, weight, output);
				break;
			case CDNNS_OPTION_FUSE_RELU:
				fwd_fuse_relu_ow_f32(batch_size, in_dim,
						     out_dim, input,
					   	     weight, bias, output);
				break;
			}
			break;
		case  CDNNS_MODE_ACCUMULATE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_accum_f32(batch_size, in_dim,
						      out_dim, input, weight,
						      bias, output);
				break;
			case CDNNS_OPTION_NO_BIAS:
				fwd_no_bias_accum_f32(batch_size, in_dim,
						      out_dim, input, weight,
						      output);
				break;
			case CDNNS_OPTION_FUSE_RELU:
				fwd_fuse_relu_accum_f32(batch_size, in_dim,
							out_dim, input,
					   	   	weight, bias, output);
				break;
			}
			break;
		}
		break;
	case CDNNS_TYPE_FP64:

		switch(mode) {
		case CDNNS_MODE_OVERWRITE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_ow_f64(batch_size, in_dim, out_dim, input,
					   weight, bias, output);
				break;
			case CDNNS_OPTION_NO_BIAS:
				fwd_no_bias_ow_f64(batch_size, in_dim, out_dim,
						   input, weight, output);
				break;
			case CDNNS_OPTION_FUSE_RELU:
				fwd_fuse_relu_ow_f64(batch_size, in_dim,
						     out_dim, input,
					   	     weight, bias, output);
				break;
			}
			break;
		case  CDNNS_MODE_ACCUMULATE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_accum_f64(batch_size, in_dim, out_dim,
					      input, weight, bias, output);
				break;
			case CDNNS_OPTION_NO_BIAS:
				fwd_no_bias_accum_f64(batch_size, in_dim,
						      out_dim, input, weight,
						      output);
				break;
			case CDNNS_OPTION_FUSE_RELU:
				fwd_fuse_relu_accum_f64(batch_size, in_dim,
							out_dim, input,
					   	   	weight, bias, output);
				break;
			}
			break;
		}
		break;
	}
}

/*
void cdnns_linear_backward(enum cdnns_type type, enum cdnns_mode mode,
			   enum cdnns_reduction reduction,
			   size_t batch_size, size_t in_dim, size_t out_dim,
			   const void *restrict input,
			   void *restrict input_grad,
			   const void *restrict weight,
			   void *restrict weight_grad,
			   void *restrict bias_grad,
			   const void *restrict output_grad)
{
#if defined USE_CBLAS_API
	// Compute the gradient of the input
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size,
		    in_dim, out_dim, 1.0f, output_grad, out_dim, weight,
		    in_dim, 0.0f, input_grad, in_dim );

	// Branch on reduction to compute the gradients of the weight and bias
	if (reduction == CNeural_REDUCTION_MEAN) {
		// Compute the gradient of the weight
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim,
			    in_dim, batch_size, 1.0f / batch_size, output_grad,
			    out_dim, input, in_dim, 1.0f, weight_grad, in_dim);

		// Compute the gradient of the bias
		for (size_t b = 0; b < batch_size; b++) {
			cblas_saxpy(out_dim, 1.0f / batch_size, 
				    output_grad + b * out_dim, 1,
				    bias_grad, 1);
		}
	}
	else if (reduction == CNeural_REDUCTION_SUM) {
		// Compute the gradient of the weight
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim,
			    in_dim, batch_size, 1.0f, output_grad, out_dim,
			    input, in_dim, 1.0f, weight_grad, in_dim);

		// Compute the gradient of the bias
		for (size_t b = 0; b < batch_size; b++) {
			const float *og_b = output_grad + b * out_dim;
			cblas_saxpy(out_dim, 1.0f, og_b, 1, bias_grad, 1);
		}
	}
	else if (CNeural_REDUCTION_NONE) {
		// Compute the gradient of the weight
		CBLAS_TRANSPOSE TransA  = CblasTrans;
		CBLAS_TRANSPOSE TransB  = CblasNoTrans;
		cblasint M = out_dim;
		cblasint N = in_dim;
		cblasint K = 1;
		cblasint lda = out_dim;
		cblasint ldb = in_dim;;
		cblasint ldc = in_dim;
		float beta = 0.0f;
		float alpha = 1.0f;

		float **A_array = malloc(batch_size * sizeof(float*));
		if (A_array == NULL) {
			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(float*));

			fflush(stderr);
			abort();
		}
		float **B_array = malloc(batch_size * sizeof(float*));
		if (B_array == NULL) {
			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(float*));

			fflush(stderr);
			abort();
		}
		float **C_array = malloc(batch_size * sizeof(float*));
		if (C_array == NULL) {
			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(float*));

			fflush(stderr);
			abort();
		}

		for (size_t b = 0; b < batch_size; b++) {
			A_array[b] = output_grad + b * out_dim;
			B_array[b] = input   + b * in_dim;   
			C_array[b] = weight  + b * out_dim * in_dim;
		}

		cblas_sgemm_batch(
			CblasRowMajor,
			&TransA, &TransB,
			&M, &N, &K,
			&alpha,
			A_array, &lda,
			B_array, &ldb,
			&beta,
			C_array, &ldc,
			1, &batch_size
		);
		free(A_array);
		free(B_array);
		free(C_array);

		// Compute the gradient of the bias
		cblas_saxpy(out_dim * batch_size, 1.0f, output_grad, 1,
			    bias_grad, 1);
	}
#else
	// Compute the gradient of the input
	float *in_g = input_grad;
	const float *out_g = output_grad;
	for (size_t b = 0; b < batch_size; b++) {
		const float *w = weight;
		for (size_t o = 0; o < out_dim; o++) {
			const float og = out_g[o];

			for (size_t i = 0; i < in_dim; i++)
				in_g[i] += og * w[i];
			w += in_dim;
		}

		in_g += in_dim;
		out_g += out_dim;
	}

	// Branch on reduction to compute the gradients of the weight and bias
	if (reduction == CDNNS_REDUCTION_MEAN) {
		// Compute the gradient of the weight and bias
		const float *in = input;
		const float *out_g = output_grad;
		for (size_t b = 0; b < batch_size; b++) {
			float *w_g = weight_grad;
			for (size_t o = 0; o < out_dim; o++) {
				for (size_t i = 0; i < in_dim; i++) {
					w_g[i] += (out_g[o] * in[i]) / batch_size;
				}
				bias_grad[o] += out_g[o] / batch_size;
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (reduction == CDNNS_REDUCTION_SUM) {
		// Compute the gradient of the weight and bias
		const float *in = input;
		const float *out_g = output_grad;
		for (size_t b = 0; b < batch_size; b++) {
			float *w_g = weight_grad;
			for (size_t o = 0; o < out_dim; o++) {
				for (size_t i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				bias_grad[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (CDNNS_REDUCTION_NONE) {
		// Compute the gradient of the weight and bias
		const float *in = input;
		const float *out_g = output_grad;
		float *w_g = weight_grad;
		float *b_g = bias_grad;
		for (size_t b = 0; b < batch_size; b++) {
			for (size_t o = 0; o < out_dim; o++) {
				for (size_t i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				b_g[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			b_g += out_dim;
			out_g += out_dim;
		}
	}
#endif
}

// NOT TESTED!
void cdnns_linear_backward_f64(
	const double *restrict input, double *restrict input_grad,
	const double *restrict weight, double *restrict weight_grad,
	double *restrict bias_grad,
	const double *restrict output_grad,
	size_t batch_size,
	size_t in_dim, size_t out_dim,
	enum CNeural_reduction reduction)
{
#if defined USE_CBLAS_API
	// Compute the gradient of the input
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans, CblasNoTrans,
		batch_size, in_dim, out_dim,
		1.0,
		output_grad, out_dim,
		weight, in_dim,
		0.0,
		input_grad, in_dim
	);
	// Branch on reduction to compute the gradients of the weight and bias
	if (reduction == CNeural_REDUCTION_MEAN) {
		// Compute the gradient of the weight
		cblas_dgemm(
			CblasRowMajor,
			CblasTrans, CblasNoTrans,
			out_dim, in_dim, batch_size,
			1.0 / batch_size,
			output_grad, out_dim,
			input, in_dim,
			0.0,
			weight_grad, in_dim
		);
		// Compute the gradient of the bias
		for (size_t b = 0; b < batch_size; b++) {
			cblas_daxpy(
				out_dim,
				1.0 / batch_size,
				output_grad + b * out_dim, 1,
				bias_grad, 1
			);
		}
	}
	else if (reduction == CNeural_REDUCTION_SUM) {
		// Compute the gradient of the weight
		cblas_dgemm(
			CblasRowMajor,
			CblasTrans, CblasNoTrans,
			out_dim, in_dim, batch_size,
			1.0,
			output_grad, out_dim,
			input, in_dim,
			0.0,
			weight_grad, in_dim
		);
		// Compute the gradient of the bias
		for (size_t b = 0; b < batch_size; b++) {
			cblas_daxpy(
				out_dim,
				1.0,
				output_grad + b * out_dim, 1,
				bias_grad, 1
			);
		}
	}
	else if (CNeural_REDUCTION_NONE) {
		// Compute the gradient of the weight
		CBLAS_TRANSPOSE TransA = CblasTrans;
		CBLAS_TRANSPOSE TransB = CblasNoTrans;
		cblasint M = out_dim;
		cblasint N = in_dim;
		cblasint K = 1;
		cblasint lda = out_dim;
		cblasint ldb = in_dim;;
		cblasint ldc = in_dim;
		double alpha = 1.0;
		double beta = 0.0;

		double **A_array = malloc(batch_size * sizeof(double*));
		double **B_array = malloc(batch_size * sizeof(double*));
		double **C_array = malloc(batch_size * sizeof(double*));
		if (A_array == NULL || B_array == NULL || C_array == NULL) {
			size_t requested_num = 0;
			requested_num += (A_array == NULL);
			requested_num += (B_array == NULL);
			requested_num += (C_array == NULL);
	
			free(A_array);
			free(B_array);
			free(C_array);

			fprintf(stderr,
			"CNeural ERROR: Failed to allocate memory. "
			"Requested size: %zu bytes. Aborting.\n",
			batch_size * sizeof(double*));

			fflush(stderr);
			abort();
		}

		for (size_t b = 0; b < batch_size; b++) {
			A_array[b] = output_grad + b * out_dim;
			B_array[b] = input   + b * in_dim;
			C_array[b] = weight + b * out_dim * in_dim;
		}
		cblas_dgemm_batch(
			CblasRowMajor, &TransA, &TransB,
			&M, &N, &K,
			&alpha,
			A_array, &lda,
			B_array, &ldb,
			&beta,
			C_array, &ldc,
			1, &batch_size
		);

		free(A_array);
		free(B_array);
		free(C_array);

		// Compute the gradient of the bias
		cblas_daxpy(out_dim * batch_size,
			1.0,
			output_grad, 1,
			bias_grad, 1
		);
	}
#else
	// Compute the gradient of the input
	double *in_g = input_grad;
	const double *out_g = output_grad;
	for (size_t b = 0; b < batch_size; b++) {
		const double *w = weight;
		for (size_t o = 0; o < out_dim; o++) {
			const double og = out_g[o];

			for (size_t i = 0; i < in_dim; i++)
				in_g[i] += og * w[i];
			w += in_dim;
		}
		in_g += in_dim;
		out_g += out_dim;
	}

	// Branch on reduction to compute the gradients of the weight and bias
	if (reduction == CDNNS_REDUCTION_MEAN) {
		// Compute the gradient of the weight and bias
		const double *in = input;
		const double *out_g = output_grad;
		for (size_t b = 0; b < batch_size; b++) {
			double *w_g = weight_grad;
			for (size_t o = 0; o < out_dim; o++) {
				for (size_t i = 0; i < in_dim; i++) {
					w_g[i] += (out_g[o] * in[i]) / batch_size;
				}
				bias_grad[o] += out_g[o] / batch_size;
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (reduction == CDNNS_REDUCTION_SUM) {
		// Compute the gradient of the weight and bias
		const double *in = input;
		const double *out_g = output_grad;
		for (size_t b = 0; b < batch_size; b++) {
			double *w_g = weight_grad;
			for (size_t o = 0; o < out_dim; o++) {
				for (size_t i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				bias_grad[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			out_g += out_dim;
		}
	}
	else if (CDNNS_REDUCTION_NONE) {
		// Compute the gradient of the weight and bias
		const double *in = input;
		const double *out_g = output_grad;
		double *w_g = weight_grad;
		double *b_g = bias_grad;
		for (size_t b = 0; b < batch_size; b++) {
			for (size_t o = 0; o < out_dim; o++) {
				for (size_t i = 0; i < in_dim; i++) {
					w_g[i] += out_g[o] * in[i];
				}
				b_g[o] += out_g[o];
				w_g += in_dim;
			}
			in += in_dim;
			b_g += out_dim;
			out_g += out_dim;
		}
	}
#endif
}

*/
