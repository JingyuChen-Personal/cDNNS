#include "internal.h"

void cdnns_sigmoid_forward(enum cdnns_type type, enum cdnns_mode mode, enum cdnns_option option,
			   size_t batch_size, size_t inout_dim,
			   const void *input, void *output)
{
	switch (type) {
	case CDNNS_TYPE_FP32:

		switch(mode) {
		case CDNNS_MODE_OVERWRITE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_ow_f32(batch_size, inout_dim,
						   input, output);
				break;
			}
			break;
		case CDNNS_MODE_OVERWRITE_IN_PLACE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_ow_in_place_f32(batch_size, inout_dim,
						      	    output);
				break;
			}
			break;
		case CDNNS_MODE_ACCUMULATE:
			
			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_accum_f32(batch_size, inout_dim,
						      input, output);
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
				fwd_default_ow_f64(batch_size, inout_dim,
						   input, output);
				break;
			}
			break;
		case CDNNS_MODE_OVERWRITE_IN_PLACE:

			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_ow_in_place_f64(batch_size, inout_dim,
						      	    output);
				break;
			}
			break;
		case CDNNS_MODE_ACCUMULATE:
			
			switch (option) {
			case CDNNS_OPTION_DEFAULT:
				fwd_default_accum_f64(batch_size, inout_dim,
						      input, output);
				break;
			}
			break;
		}
		break;
	}
}



/*
// NOT TESTED!
void CNeural_relu_backward_f64(const double *input, double *input_grad,
			       const double *output_grad, size_t batch_size,
			       size_t inout_dim)
{
	// Compute the gradient of the input
	for (size_t b = 0; b < batch_size; b++) {
		for (size_t io = 0; io < inout_dim; io++) {
			input_grad[b  *inout_dim + io] += output_grad[b  *inout_dim + io]  *(input[b  *inout_dim + io] > 0.0);
		}
	}
}
*/