#include "block_match.h"

#include "block_match_internal.h"
#include <cuda_runtime.h>
#include <cstdlib>

bool initialize_local(void **_instance, 
	int matA_M, int matA_N, int matB_M, int matB_N, 
	int block_M, int block_N, int neighbour_M, int neighbour_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N)
{
	struct Context * instance = (struct Context *)malloc(sizeof(struct Context));
	if (!instance)
		return false;
	
	instance->matA_M = matA_M;
	instance->matA_N = matA_N;
	instance->matB_M = matB_M;
	instance->matB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->neighbour_M = neighbour_M;
	instance->neighbour_N = neighbour_N;

	instance->strideA_M = strideA_M;
	instance->strideA_N = strideA_N;
	instance->strideB_M = strideB_M;
	instance->strideB_N = strideB_N;

	int result_dim0 = (matA_M + 2 * paddingA_M - block_M + 1) / strideA_M;
	int result_dim1 = (matA_N + 2 * paddingA_N - block_N + 1) / strideA_N;
	int result_dim2 = (neighbour_M + strideB_M - 1) / strideB_M;
	int result_dim3 = (neighbour_N + strideB_N - 1) / strideB_N;

	instance->result_dims[0] = result_dim0;
	instance->result_dims[1] = result_dim1;
	instance->result_dims[2] = result_dim2;
	instance->result_dims[3] = result_dim3;

	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;
	
	cudaError_t cuda_error;

	int numberOfThreads = globalContext.numberOfThreads;

	instance->stream = new cudaStream_t[numberOfThreads];
	if (!instance->stream)
		goto release_cuda_stream;

	for (int i = 0; i < numberOfThreads; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
		{
			for (int j=i-1; j>=0;j--)
			{
				cudaStreamDestroy(instance->stream[j]);
			}
			goto release_cuda_stream;
		}
	}

	cuda_error = cudaMallocHost(&instance->buffer_A, 
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->buffer_B,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer,
		result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * sizeof(float));

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
release_cuda_stream:
	delete [] instance->stream;

release_instance:

	free(instance);
	return false;
}

bool initialize_full(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N)
{
	struct Context * instance = (struct Context *)malloc(sizeof(struct Context));
	if (!instance)
		return false;

	instance->matA_M = matA_M;
	instance->matA_N = matA_N;
	instance->matB_M = matB_M;
	instance->matB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->strideA_M = strideA_M;
	instance->strideA_N = strideA_N;
	instance->strideB_M = strideB_M;
	instance->strideB_N = strideB_N;

	int result_dim0 = (matA_M + 2 * paddingA_M - block_M + strideA_M - 1) / strideA_M;
	int result_dim1 = (matA_N + 2 * paddingA_N - block_N + strideA_N - 1) / strideA_N;
	int result_dim2 = (matB_M + 2 * paddingB_M - block_M + strideB_M - 1) / strideB_M;
	int result_dim3 = (matA_N + 2 * paddingB_N - block_N + strideB_N - 1) / strideB_N;

	instance->result_dims[0] = result_dim0;
	instance->result_dims[1] = result_dim1;
	instance->result_dims[2] = result_dim2 * result_dim3;

	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	cudaError_t cuda_error;

	int numberOfThreads = globalContext.numberOfThreads;

	instance->stream = new cudaStream_t[numberOfThreads];
	if (!instance->stream)
		goto release_cuda_stream;

	for (int i = 0; i < numberOfThreads; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
		{
			for (int j = i - 1; j >= 0; j--)
			{
				cudaStreamDestroy(instance->stream[j]);
			}
			goto release_cuda_stream;
		}
	}

	cuda_error = cudaMallocHost(&instance->buffer_A,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->buffer_B,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer,
		result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	instance->index = (int*)malloc(result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(int));
	if (!instance->index)
	{
		goto release_index;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * sizeof(float));

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

release_index:
	delete[] instance->index;
release_buffer_B:

	cudaFreeHost(instance->buffer_B);
release_buffer_A:

	cudaFreeHost(instance->buffer_A);
release_cuda_stream:
	delete[] instance->stream;

release_instance:

	free(instance);
	return false;
}

extern "C"
bool initialize(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N)
{
	return initialize_full(_instance,
		matA_M, matA_N, matB_M, matB_N,
		block_M, block_N,
		strideA_M, strideA_N,
		strideB_M, strideB_N,
		paddingA_M, paddingA_N,
		paddingB_M, paddingB_N);
}