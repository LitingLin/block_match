#include "lib_match_internal.h"

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

	struct BlockMatchContext * instance = (struct BlockMatchContext *)malloc(sizeof(struct BlockMatchContext));
	if (!instance)
		return false;

	instance->matrixA_M = matA_M;
	instance->matrixA_N = matA_N;
	instance->matrixB_M = matB_M;
	instance->matrixB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->searchRegion_M = neighbour_M;
	instance->searchRegion_N = neighbour_N;

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

	cuda_error = cudaMallocHost(&instance->matrixA_buffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->matrixB_buffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->matrixC_buffer,
		result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->matrixA_deviceBuffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->matrixB_deviceBuffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->matrixC_deviceBuffer,
		numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * sizeof(float));

	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_B;
	}
	*_instance = instance;

	return true;

release_device_buffer_B:

	cudaFree(instance->matrixB_deviceBuffer);
release_device_buffer_A:

	cudaFree(instance->matrixA_deviceBuffer);
release_result_buffer:

	cudaFreeHost(instance->matrixC_buffer);
release_buffer_B:

	cudaFreeHost(instance->matrixB_buffer);
release_buffer_A:

	cudaFreeHost(instance->matrixA_buffer);
release_cuda_stream:
	delete[] instance->stream;

release_instance:

	free(instance);
	return false;
}

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations)
{
	double numberOfBlockAPerProcessor = (double)maxNumberOfGpuThreads / (double)numberOfBlockBPerBlockA;
	if (numberOfBlockAPerProcessor > 1.0)
	{
		int fixedNumberOfBlockAPerProcessor = (int)numberOfBlockAPerProcessor;
		*numberOfSubmitThreadsPerProcessor = (int)numberOfBlockAPerProcessor * numberOfBlockBPerBlockA;
		*numberOfSubmitProcessors = numberOfGpuProcessors;
		*numberOfIterations = fixedNumberOfBlockAPerProcessor * numberOfGpuProcessors;
	}
	else
	{
		double numberOfProcessorPerBlockA = 1.0 / numberOfBlockAPerProcessor;
		if (numberOfProcessorPerBlockA < numberOfGpuProcessors)
		{
			int _numberOfIterations = (int)((double)numberOfGpuProcessors / numberOfProcessorPerBlockA);
			int _numberOfSubmitProcessors = (int)std::ceil(_numberOfIterations * numberOfProcessorPerBlockA);
			*numberOfSubmitThreadsPerProcessor = maxNumberOfGpuThreads;
			*numberOfSubmitProcessors = _numberOfSubmitProcessors;
			*numberOfIterations = _numberOfIterations;
		}
		else
		{
			*numberOfSubmitThreadsPerProcessor = maxNumberOfGpuThreads;
			*numberOfSubmitProcessors = std::ceil(numberOfProcessorPerBlockA);
			*numberOfIterations = 1;
		}
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

	struct BlockMatchContext * instance = (struct BlockMatchContext *)malloc(sizeof(struct BlockMatchContext));
	if (!instance)
		return false;

	instance->matrixA_M = matA_M;
	instance->matrixA_N = matA_N;
	instance->matrixB_M = matB_M;
	instance->matrixB_N = matB_N;

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

	instance->numberOfIndexRetain = retain;

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
	//int result_dim2 = (matrixB_M + 2 * paddingB_M - block_M + strideB_M - 1) / strideB_M;
	//int result_dim3 = (matrixA_N + 2 * paddingB_N - block_N + strideB_N - 1) / strideB_N;

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

	instance->stream = new cudaStream_t[numberOfThreads * 2];
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

	cuda_error = cudaMallocHost(&instance->matrixA_buffer,
		(matBufferSize * 2 + bufferSize) * sizeof(float)
	);

	if (cuda_error != cudaSuccess)
		goto release_cuda_stream;

	instance->matrixB_buffer = instance->matrixA_buffer + matBufferSize;
	instance->matrixC_buffer = instance->matrixB_buffer + matBufferSize;

	instance->index_x = (int *)malloc(
		(resultSize + bufferSize) * 2 * sizeof(int) +
		resultSize * sizeof(float) +
		numberOfBlockBPerBlockA*(numberOfThreads + 1) * sizeof(int));

	if (instance->index_x == nullptr)
		goto release_page_locked_memory;

	instance->index_y = instance->index_x + resultSize;
	instance->index_x_buffer = instance->index_y + resultSize;
	instance->index_y_buffer = instance->index_x_buffer + bufferSize;

	instance->C = (float*)instance->index_y_buffer + bufferSize;

	instance->common_buffer = (int*)(instance->C + resultSize);
	instance->indexSorting_buffer = instance->common_buffer + numberOfBlockBPerBlockA;

	generateIndexSequence(instance->common_buffer, numberOfBlockBPerBlockA);

	cuda_error = cudaMalloc(&instance->matrixA_deviceBuffer,
		(matBufferSize * 2 + bufferSize) * sizeof(float));
	if (cuda_error != cudaSuccess)
		goto release_memory;

	instance->matrixB_deviceBuffer = instance->matrixA_deviceBuffer + matBufferSize;
	instance->matrixC_deviceBuffer = instance->matrixB_deviceBuffer + matBufferSize;

	*_instance = instance;

	return true;

release_device_memory:
	cudaFree(instance->matrixA_deviceBuffer);

release_memory:
	delete[] instance->index_x;
release_page_locked_memory:

	cudaFreeHost(instance->matrixA_buffer);

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

bool initialize_(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int neighbour_M, int neighbour_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N,
	int retain)
{
	struct BlockMatchContext * instance = (struct BlockMatchContext *)malloc(sizeof(struct BlockMatchContext));
	if (!instance)
		return false;

	instance->matrixA_M = matA_M;
	instance->matrixA_N = matA_N;
	instance->matrixB_M = matB_M;
	instance->matrixB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->searchRegion_M = neighbour_M;
	instance->searchRegion_N = neighbour_N;

	instance->strideA_M = strideA_M;
	instance->strideA_N = strideA_N;
	instance->strideB_M = strideB_M;
	instance->strideB_N = strideB_N;
	instance->sequenceAPadding_M = paddingA_M;
	instance->sequenceAPadding_N = paddingA_N;
	instance->sequenceBPadding_M = paddingB_M;
	instance->sequenceBPadding_N = paddingB_N;

	instance->numberOfIndexRetain = retain;

	int result_dim0 = getLength(matA_M, paddingA_M, block_M, strideA_M);
	int result_dim1 = getLength(matA_N, paddingA_N, block_N, strideA_N);

	int result_dim2, result_dim3;

	int numberOfThreads = globalContext.numberOfThreads;
	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	if (retain)
		result_dim2 = retain;
	else
		result_dim2 = getLength(matB_M, paddingB_M, block_M, strideB_M)* getLength(matA_N, paddingB_N, block_N, strideB_N);
	//int result_dim2 = (matrixB_M + 2 * paddingB_M - block_M + strideB_M - 1) / strideB_M;
	//int result_dim3 = (matrixA_N + 2 * paddingB_N - block_N + strideB_N - 1) / strideB_N;

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

	instance->stream = new cudaStream_t[numberOfThreads * 2];
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

	cuda_error = cudaMallocHost(&instance->matrixA_buffer,
		(matBufferSize * 2 + bufferSize) * sizeof(float)
	);

	if (cuda_error != cudaSuccess)
		goto release_cuda_stream;

	instance->matrixB_buffer = instance->matrixA_buffer + matBufferSize;
	instance->matrixC_buffer = instance->matrixB_buffer + matBufferSize;

	instance->index_x = (int *)malloc(
		(resultSize + bufferSize) * 2 * sizeof(int) +
		resultSize * sizeof(float) +
		numberOfBlockBPerBlockA*(numberOfThreads + 1) * sizeof(int));

	if (instance->index_x == nullptr)
		goto release_page_locked_memory;

	instance->index_y = instance->index_x + resultSize;
	instance->index_x_buffer = instance->index_y + resultSize;
	instance->index_y_buffer = instance->index_x_buffer + bufferSize;

	instance->C = (float*)instance->index_y_buffer + bufferSize;

	instance->common_buffer = (int*)(instance->C + resultSize);
	instance->indexSorting_buffer = instance->common_buffer + numberOfBlockBPerBlockA;

	generateIndexSequence(instance->common_buffer, numberOfBlockBPerBlockA);

	cuda_error = cudaMalloc(&instance->matrixA_deviceBuffer,
		(matBufferSize * 2 + bufferSize) * sizeof(float));
	if (cuda_error != cudaSuccess)
		goto release_memory;

	instance->matrixB_deviceBuffer = instance->matrixA_deviceBuffer + matBufferSize;
	instance->matrixC_deviceBuffer = instance->matrixB_deviceBuffer + matBufferSize;

	*_instance = instance;

	return true;

release_device_memory:
	cudaFree(instance->matrixA_deviceBuffer);

release_memory:
	delete[] instance->index_x;
release_page_locked_memory:

	cudaFreeHost(instance->matrixA_buffer);

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

bool blockMatchInitialize(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int searchRegion_M, int searchRegion_N,
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