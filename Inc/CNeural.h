#ifndef CNEURAL_H
#define CNEURAL_H

// #define ONLY_NECCESSARY_CHECK

// #define USE_OPENBLAS
// #define USE_ONEMKL

#if defined USE_ONEMKL
    #include "mkl_cblas.h"
    #define  USE_CBLAS_API
typedef MKL_INT cblas_int;
#endif

#if defined USE_OPENBLAS
    #include "cblas.h"
    #define USE_CBLAS_API
    typedef blasint cblasint;
#endif

typedef enum {
    CNeural_Success,
    CNeural_Error_NullPointer,
    CNeural_Error_InvalidParameter,
    CNeural_Error_AllocationFailed
} CNeural_Status;


typedef enum {
    CNeural_Reduction_None,
    CNeural_Reduction_Mean,
    CNeural_Reduction_Sum
} CNeural_Reduction;

#endif