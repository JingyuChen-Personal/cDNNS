#ifndef CNEURAL_H
#define CNEURAL_H

// #define ONLY_NECCESSARY_CHECK

// #define  USE_CBLAS_API


#ifdef  USE_CBLAS_API

#include "cblas.h"

#endif

typedef enum
{
	CNeural_Success,
	CNeural_Error_NullPointer,
	CNeural_Error_InvalidParameter,
	CNeural_Error_AllocationFailed
} CNeural_Status;


typedef enum
{
    CNeural_Reduction_None,
    CNeural_Reduction_Mean,
    CNeural_Reduction_Sum
} CNeural_Reduction;

#endif