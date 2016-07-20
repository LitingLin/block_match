#include "block_match.h"

#include "block_match_internal.h"
#include <cuda_runtime.h>
#include <cstdlib>

bool initialize(void **_instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N, size_t neighbour_M, size_t neighbour_N, size_t stride_M, size_t stride_N)
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

	size_t result_dim0 = matA_M - block_M + 1;
	size_t result_dim1 = matA_N - block_N + 1;
	size_t result_dim2 = (neighbour_M + stride_M - 1) / stride_M;
	size_t result_dim3 = (neighbour_N + stride_N - 1) / stride_N;
	instance->result_dim0 = result_dim0;
	instance->result_dim1 = result_dim1;
	instance->result_dim2 = result_dim2;
	instance->result_dim3 = result_dim3;

	int numDeviceMultiProcessor;
	const int numProcessorThread = 512;

	cudaError_t cuda_error = cudaDeviceGetAttribute(&numDeviceMultiProcessor, cudaDevAttrMultiProcessorCount, 0);
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	instance->numDeviceMultiProcessor = numDeviceMultiProcessor;
	instance->numProcessorThread = numProcessorThread;

	instance->pool = pool;

	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
			return false;
	}

	cuda_error = cudaMallocHost(&instance->buffer_A, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->buffer_B, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer, result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * sizeof(float));

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
bool initialize(void **_instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N, size_t neighbour_M, size_t neighbour_N, size_t stride_M, size_t stride_N)
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