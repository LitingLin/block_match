#include "block_match_internal.h"

extern "C"
void finalize(void *_instance)
{
	cudaError_t cuda_error;
	Context *instance = (Context *)_instance;

	const unsigned numberOfThreads = globalContext.numberOfThreads;

	for (unsigned i = 0; i < numberOfThreads * 2; ++i) {
		cuda_error = cudaStreamDestroy(instance->stream[i]);
		if (cuda_error != cudaSuccess)
			logger.warn("cudaStreamDestroy failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));
	}

	cuda_error = cudaFreeHost(instance->buffer_A);
	if (cuda_error != cudaSuccess)
		logger.warn("cudaFreeHost failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));
	cuda_error = cudaFree(instance->device_buffer_A);
	if (cuda_error != cudaSuccess)
		logger.warn("cudaFree failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));

	delete[] instance->stream;
	delete[] instance->index_x;

	free(instance);
}