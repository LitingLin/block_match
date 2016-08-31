#include "block_match_internal.h"

#include <cuda_runtime.h>

bool initialize_local(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int block_M, int block_N, int neighbour_M, int neighbour_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N)
{
	static bool isGlobalContextInitialized = false;
	if (!isGlobalContextInitialized)
	{
		globalContext.initialize();
		isGlobalContextInitialized = true;
	}

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

	for (int i = 0; i < numberOfThreads * 2; ++i)
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
	delete[] instance->stream;

release_instance:

	free(instance);
	return false;
}

void determineGpuTaskDistribution(int *numberOfGpuThreads, int *numberOfGpuProcessor, int *numberOfQueuedData, int numberOfBlockBPerBlockA)
{
	int maxNumberOfGpuThreads = *numberOfGpuThreads;
	if (numberOfBlockBPerBlockA > maxNumberOfGpuThreads)
	{
		
	}
	else
	{
		*numberOfQueuedData = *numberOfGpuThreads / numberOfBlockBPerBlockA;

		*numberOfGpuThreads = numberOfBlockBPerBlockA / *numberOfGpuProcessor * (*numberOfGpuProcessor);
	}
}

bool initialize_full(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N,
	int retain)
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
	instance->sequenceAPadding_M = paddingA_M;
	instance->sequenceAPadding_N = paddingA_N;
	instance->sequenceBPadding_M = paddingB_M;
	instance->sequenceBPadding_N = paddingB_N;

	instance->retain = retain;

	int result_dim0 = getLength(matA_M, paddingA_M, block_M, strideA_M);
	int result_dim1 = getLength(matA_N, paddingA_N, block_N, strideA_N);
	int result_dim2;

	int numberOfThreads = globalContext.numberOfThreads;
	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	if (retain)
		result_dim2 = retain;
	else
		result_dim2 = getLength(matB_M, paddingB_M, block_M, strideB_M)* getLength(matA_N, paddingB_N, block_N, strideB_N);
	//int result_dim2 = (matB_M + 2 * paddingB_M - block_M + strideB_M - 1) / strideB_M;
	//int result_dim3 = (matA_N + 2 * paddingB_N - block_N + strideB_N - 1) / strideB_N;
	
	int group_M = getLength(matB_M, paddingB_M, block_M, strideB_M);
	int group_N = getLength(matB_N, paddingB_N, block_N, strideB_N);
	int numberOfBlockBPerBlockA = group_M * group_N;
	if (numberOfBlockBPerBlockA > numberOfGPUProcessorThread)
		numberOfGPUProcessorThread = numberOfBlockBPerBlockA;

	instance->numberOfBlockBPerBlockA = numberOfBlockBPerBlockA;

	if (retain > numberOfBlockBPerBlockA)
		return false;
	if (result_dim0 < globalContext.numberOfThreads)
		return false;

	int matBufferSize = numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
	int resultSize = result_dim0 * result_dim1 * result_dim2;

	int	bufferSize = numberOfGPUProcessorThread / numberOfBlockBPerBlockA * numberOfBlockBPerBlockA * numberOfGPUDeviceMultiProcessor;
	instance->perThreadBufferSize = bufferSize;
	bufferSize *= numberOfThreads;

	instance->result_dims[0] = result_dim0;
	instance->result_dims[1] = result_dim1;
	instance->result_dims[2] = result_dim2;


	cudaError_t cuda_error;

	instance->stream = new cudaStream_t[numberOfThreads*2];
	if (!instance->stream) {
		goto release_instance;
	}

	for (int i = 0; i < numberOfThreads * 2; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
		{
			for (int j = i - 1; j >= 0; j--)
			{
				cudaStreamDestroy(instance->stream[j]);
			}
			delete[] instance->stream;
			goto release_instance;
		}
	}

	// Remember to * sizeof(type size)

	cuda_error = cudaMallocHost(&instance->buffer_A,
		(matBufferSize * 2 + bufferSize) * sizeof(float)
	);

	if (cuda_error != cudaSuccess)
		goto release_cuda_stream;

	instance->buffer_B = instance->buffer_A + matBufferSize;
	instance->result_buffer = instance->buffer_B + matBufferSize;
	
	instance->index_x = (int *)malloc((resultSize + bufferSize) * 2 * sizeof(int) + resultSize * sizeof(float) + numberOfBlockBPerBlockA*(numberOfThreads + 1) * sizeof(int));
	if (instance->index_x == nullptr)
		goto release_page_locked_memory;
	instance->index_y = instance->index_x + resultSize;
	instance->index_x_buffer = instance->index_y + resultSize;
	instance->index_y_buffer = instance->index_x_buffer + bufferSize;

	instance->result = (float*)instance->index_y_buffer + bufferSize;
	
	instance->index_buffer = (int*)(instance->result + resultSize);
	instance->index_buffer_sort = instance->index_buffer + numberOfBlockBPerBlockA;

	generateIndexSequence(instance->index_buffer, numberOfBlockBPerBlockA);

	cuda_error = cudaMalloc(&instance->device_buffer_A,
		(matBufferSize * 2 + bufferSize) * sizeof(float));
	if (cuda_error != cudaSuccess)
		goto release_memory;

	instance->device_buffer_B = instance->device_buffer_A + matBufferSize;
	instance->device_result_buffer = instance->device_buffer_B + matBufferSize;

	*_instance = instance;

	return true;

release_device_memory:
	cudaFree(instance->device_buffer_A);

release_memory:
	delete[] instance->index_x;
release_page_locked_memory:

	cudaFreeHost(instance->buffer_A);

release_cuda_stream:
	for (int i = 0; i < numberOfThreads * 2; ++i)
	{
		cudaStreamDestroy(instance->stream[i]);
	}
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
	int paddingB_M, int paddingB_N,
	int retain)
{
	static bool isGlobalContextInitialized = false;
	if (!isGlobalContextInitialized)
	{
		if (!globalContext.initialize())
			return false;
		isGlobalContextInitialized = true;
	}

	return initialize_full(_instance,
		matA_M, matA_N, matB_M, matB_N,
		block_M, block_N,
		strideA_M, strideA_N,
		strideB_M, strideB_N,
		paddingA_M, paddingA_N,
		paddingB_M, paddingB_N,
		retain);
}