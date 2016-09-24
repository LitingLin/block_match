#include "lib_match_internal.h"

void blockMatchFinalize(void *_instance)
{
	cudaError_t cuda_error;
	BlockMatchContext *instance = static_cast<BlockMatchContext *>(_instance);

	const unsigned numberOfThreads = globalContext.numberOfThreads;

	for (unsigned i = 0; i < numberOfThreads * 2; ++i) {
		cuda_error = cudaStreamDestroy(instance->stream[i]);
		if (cuda_error != cudaSuccess)
			logger.warn("cudaStreamDestroy failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));
	}

	cuda_error = cudaFreeHost(instance->matrixA_buffer);
	if (cuda_error != cudaSuccess)
		logger.warn("cudaFreeHost failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));
	cuda_error = cudaFree(instance->matrixA_deviceBuffer);
	if (cuda_error != cudaSuccess)
		logger.warn("cudaFree failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));

	delete[] instance->stream;
	delete[] instance->index_x;

	free(instance);
}