#include "block_match.h"

#include "block_match_internal.h"
#include <cuda_runtime.h>
#include <cstdlib>

bool initialize(void **_instance, int matA_M, int matA_N, int matB_M, int matB_N, int block_M, int block_N, int neighbour_M, int neighbour_N, int stride_M, int stride_N)
{
	struct Context * instance = (struct Context *)malloc(sizeof(struct Context));
	if (!instance)
		return false;

	instance->type = COMBILE;

	instance->matA_M = matA_M;
	instance->matA_N = matA_N;
	instance->matB_M = matB_M;
	instance->matB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->neighbour_M = neighbour_M;
	instance->neighbour_N = neighbour_N;

	instance->stride_M = stride_M;
	instance->stride_N = stride_N;

	int result_dim0 = matA_M - block_M + 1;
	int result_dim1 = matA_N - block_N + 1;
	int result_dim2 = (neighbour_M + stride_M - 1) / stride_M;
	int result_dim3 = (neighbour_N + stride_N - 1) / stride_N;
	instance->result_dim0 = result_dim0;
	instance->result_dim1 = result_dim1;
	instance->result_dim2 = result_dim2;
	instance->result_dim3 = result_dim3;

	int numberOfThreads = globalContext.numberOfThreads;
	int	numDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;


	cudaError_t cuda_error = cudaDeviceGetAttribute(&numDeviceMultiProcessor, cudaDevAttrMultiProcessorCount, 0);
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	for (uint32_t i = 0; i < numberOfGPUProcessorThread; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
			return false;
	}

	cuda_error = cudaMallocHost(&instance->buffer_A, numberOfThreads * numDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->buffer_B, numberOfThreads * numDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer, result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A, numberOfThreads * numDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B, numberOfThreads * numDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer, numberOfThreads * numDeviceMultiProcessor * numberOfGPUProcessorThread * sizeof(float));

	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_B;
	}
	*_instance = instance;

	return true;

release_device_buffer_B:

	cudaFree(instance->device_buffer_B);
release_device_buffer_A:

	cudaFree(instance->device_buffer_A);
release_result_buffer:

	cudaFreeHost(instance->result_buffer);
release_buffer_B:

	cudaFreeHost(instance->buffer_B);
release_buffer_A:

	cudaFreeHost(instance->buffer_A);
release_instance:

	free(instance);
	return false;
}

extern "C"
bool initialize(void **_instance, int matA_M, int matA_N, int matB_M, int matB_N, int block_M, int block_N, int neighbour_M, int neighbour_N, int stride_M, int stride_N)
{
	if (neighbour_M == 0)
	{
		return initialize_TypeA(_instance, matA_M, matA_N, matB_M, matB_N, block_M, block_N);
	}
	else
	{
		return initialize_async_submit(_instance, matA_M, matA_N, matB_M, matB_N, block_M, block_N, neighbour_M, neighbour_N, stride_M, stride_N);
	}
}