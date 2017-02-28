#include <cublas_v2.h>

cublasHandle_t cublas_handle;

cublasStatus_t initializeCublas()
{
	return cublasCreate(&cublas_handle);
}

cublasStatus_t destroyCublas()
{
	return cublasDestroy(cublas_handle);
}