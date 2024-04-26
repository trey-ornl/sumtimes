#pragma once

#include <cublas_v2.h>
#include <string>

#define HIPBLAS_POINTER_MODE_HOST CUBLAS_POINTER_MODE_HOST
#define HIPBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS

#define hipblasCreate cublasCreate
#define hipblasDaxpy cublasDaxpy
#define hipblasHandle_t cublasHandle_t
#define hipblasIdamax cublasIdamax
#define hipblasSetPointerMode cublasSetPointerMode
#define hipblasStatusToString std::to_string
#define hipblasStatus_t cublasStatus_t

