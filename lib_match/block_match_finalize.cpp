#include "lib_match_internal.h"

template <typename Type>
void blockMatchFinalize(void *_instance)
{
	cudaError_t cuda_error;
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(_instance);

	const unsigned numberOfThreads = instance->numberOfThreads;

	for (unsigned i = 0; i < numberOfThreads * 2; ++i) {
		cuda_error = cudaStreamDestroy(instance->stream[i]);
		if (cuda_error != cudaSuccess)
			logger.warn("cudaStreamDestroy failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));
	}
	
	cuda_error = cudaFreeHost(instance->buffer.matrixA_buffer);
	if (cuda_error != cudaSuccess)
		logger.warn("cudaFreeHost failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));
	cuda_error = cudaFree(instance->buffer.matrixA_deviceBuffer);
	if (cuda_error != cudaSuccess)
		logger.warn("cudaFree failed with {}, message {}", cuda_error, cudaGetErrorString(cuda_error));

	free(instance->workerContext.numberOfIteration);
	free(instance->stream);
	free(instance->buffer.index_x_sorting_buffer);
		
	free(instance);
}

template
void blockMatchFinalize<float>(void *_instance);
template
void blockMatchFinalize<double>(void *_instance);