#ifndef CDNNS_H
#define CDNNS_H

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
	#define RESTRICT __restrict__
#else
	#define RESTRICT restrict
#endif

#ifdef USE_ONEMKL
	#include "mkl_cblas.h"
	#define IMPL_CBLAS
	typedef MKL_INT cblasint;
#endif

#ifdef USE_OPENBLAS
	#include "cblas.h"
	#define IMPL_CBLAS
	typedef blasint cblasint;
#endif


enum cdnns_type {
	CDNNS_TYPE_FP32,
    	CDNNS_TYPE_FP64,
};

enum cdnns_mode {
	CDNNS_MODE_OVERWRITE,
	CDNNS_MODE_OVERWRITE_IN_PLACE,
    	CDNNS_MODE_ACCUMULATE
};

enum cdnns_option {
	CDNNS_OPTION_DEFAULT,
	CDNNS_OPTION_NO_BIAS,
	CDNNS_OPTION_FUSE_RELU,
	CDNNS_OPTION_IN_PLACE
};

enum cdnns_reduction {
	CDNNS_REDUCTION_MEAN,
	CDNNS_REDUCTION_SUM,
	CDNNS_REDUCTION_NONE
};
#endif