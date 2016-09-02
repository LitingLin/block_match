#include "block_match_internal.h"

#include "block_match.h"

extern "C"
enum ErrorCode arrayMatchFinalize(void *instance)
{
	ErrorCode errorCode = LibMatchErrorOk;
	ArrayMatchContext *context = (ArrayMatchContext *)instance;
	cudaError_t cudaError = cudaFree(context->deviceBufferA);
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorInternal;
		setCudaLastErrorString(cudaError, "Internal Error: in calling cudaFree");
	}
	cudaError = cudaFreeHost(context->result);
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorInternal;
		setCudaLastErrorString(cudaError, "Internal Error: in calling cudaFreeHost");
	}
	free(context);

	return errorCode;
}