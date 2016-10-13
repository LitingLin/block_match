#include "lib_match_internal.h"

#include <cuda_runtime.h>
/*
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

	instance->C_dimensions[0] = result_dim0;
	instance->C_dimensions[1] = result_dim1;
	instance->C_dimensions[2] = result_dim2;
	instance->C_dimensions[3] = result_dim3;

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
}*/

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
/*
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

	instance->C_dimensions[0] = result_dim0;
	instance->C_dimensions[1] = result_dim1;
	instance->C_dimensions[2] = result_dim2;


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
	instance->index_x_sorting_buffer = instance->index_y + resultSize;
	instance->index_y_sorting_buffer = instance->index_x_sorting_buffer + bufferSize;

	instance->C = (float*)instance->index_y_sorting_buffer + bufferSize;

	instance->common_buffer = (int*)(instance->C + resultSize);
	instance->index_raw_sorting_buffer = instance->common_buffer + numberOfBlockBPerBlockA;

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
}*/

void initializeBasicInstanceInformation(BlockMatchContext *instance,
	const int matrixA_M, const int matrixA_N, const int matrixB_M, const int matrixB_N,
	const int searchRegion_M, const int searchRegion_N,
	const int block_M, const int block_N,
	const int strideA_M, const int strideA_N,
	const int strideB_M, const int strideB_N,
	const int matrixAPadding_M_pre, const int matrixAPadding_M_post,
	const int matrixAPadding_N_pre, const int matrixAPadding_N_post,
	const int matrixBPadding_M_pre, const int matrixBPadding_M_post,
	const int matrixBPadding_N_pre, const int matrixBPadding_N_post,
	const int numberOfIndexRetain,
	const int matrixA_padded_M, const int matrixA_padded_N,
	const int matrixB_padded_M, const int matrixB_padded_N
)
{
	instance->matrixA_M = matrixA_M;
	instance->matrixA_N = matrixA_N;
	instance->matrixB_M = matrixB_M;
	instance->matrixB_N = matrixB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->searchRegion_M = searchRegion_M;
	instance->searchRegion_N = searchRegion_N;

	instance->strideA_M = strideA_M;
	instance->strideA_N = strideA_N;
	instance->strideB_M = strideB_M;
	instance->strideB_N = strideB_N;

	instance->matrixAPadding_M_pre = matrixAPadding_M_pre;
	instance->matrixAPadding_M_post = matrixAPadding_M_post;
	instance->matrixAPadding_N_pre = matrixAPadding_N_pre;
	instance->matrixAPadding_N_post = matrixAPadding_N_post;

	instance->matrixBPadding_M_pre = matrixBPadding_M_pre;
	instance->matrixBPadding_M_post = matrixBPadding_M_post;
	instance->matrixBPadding_N_pre = matrixBPadding_N_pre;
	instance->matrixBPadding_N_post = matrixBPadding_N_post;

	instance->numberOfIndexRetain = numberOfIndexRetain;

	instance->matrixA_padded_M = matrixA_padded_M;
	instance->matrixA_padded_N = matrixA_padded_N;
	instance->matrixB_padded_M = matrixB_padded_M;
	instance->matrixB_padded_N = matrixB_padded_N;
}

int determineNumberOfThreads(const int A_M, const int A_N,
	const int maxNumberOfThreads)
{
	if (A_M * A_N < maxNumberOfThreads)
		return A_M * A_N;
	else
		return maxNumberOfThreads;
}

int determineSizeOfMatrixC_O(int numberOfIndexRetain, int group_M, int group_N)
{
	if (numberOfIndexRetain)
		return numberOfIndexRetain;
	else
		return group_M * group_N;
}

bool initializeMemoryResources(BlockMatchContext *instance)
{
	const int block_M = instance->block_M;
	const int block_N = instance->block_N;
	const int numberOfBlockBPerBlockA = instance->numberOfBlockBPerBlockA;

	const int numberOfThreads = instance->numberOfThreads;
	const int numberOfGPUDeviceMultiProcessor = instance->numberOfSubmitProcessors;
	const int numberOfGPUProcessorThread = instance->numberOfSubmitThreadsPerProcessor;
	const int matBufferSize = numberOfThreads * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;

	const int bufferSize = numberOfGPUProcessorThread * numberOfGPUDeviceMultiProcessor * numberOfThreads;

	BlockMatchContext::WorkerContext &workerContext = instance->workerContext;
	workerContext.numberOfIteration = (int*)malloc(numberOfThreads * sizeof(int) * sizeof(BlockMatchContext::WorkerContext) / sizeof(int*));
	if (workerContext.numberOfIteration == nullptr)
		goto release_instance;

	workerContext.rawMatrixCIndex_begin = workerContext.numberOfIteration + numberOfThreads;
	workerContext.beginMatrixAIndex_M = workerContext.rawMatrixCIndex_begin + numberOfThreads;
	workerContext.beginMatrixAIndex_N = workerContext.beginMatrixAIndex_M + numberOfThreads;

	BlockMatchContext::Buffer &buffer = instance->buffer;

	cudaError_t cuda_error;

	instance->stream = new cudaStream_t[numberOfThreads * 2];
	if (!instance->stream) {
		goto release_worker_context;
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
			goto release_worker_context;
		}
	}

	// Remember to * sizeof(type)
	cuda_error = cudaMallocHost(&buffer.matrixA_buffer,
		(matBufferSize * 2 + bufferSize) * sizeof(float)
	);

	if (cuda_error != cudaSuccess)
		goto release_cuda_stream;

	buffer.matrixB_buffer = buffer.matrixA_buffer + matBufferSize;
	buffer.matrixC_buffer = buffer.matrixB_buffer + matBufferSize;

	buffer.index_x_sorting_buffer = (int *)malloc(
		bufferSize * 2 * sizeof(int) +
		numberOfBlockBPerBlockA*(numberOfThreads + 1) * sizeof(int));

	if (buffer.index_x_sorting_buffer == nullptr)
		goto release_page_locked_memory;

	buffer.index_y_sorting_buffer = buffer.index_x_sorting_buffer + bufferSize;

	buffer.common_buffer = buffer.index_y_sorting_buffer + bufferSize;
	buffer.index_raw_sorting_buffer = buffer.common_buffer + numberOfBlockBPerBlockA;

	generateIndexSequence(buffer.common_buffer, numberOfBlockBPerBlockA);

	cuda_error = cudaMalloc(&buffer.matrixA_deviceBuffer,
		(matBufferSize * 2 + bufferSize) * sizeof(float));
	if (cuda_error != cudaSuccess)
		goto release_memory;

	buffer.matrixB_deviceBuffer = buffer.matrixA_deviceBuffer + matBufferSize;
	buffer.matrixC_deviceBuffer = buffer.matrixB_deviceBuffer + matBufferSize;

	return true;

release_device_memory:
	cudaFree(buffer.matrixA_deviceBuffer);

release_memory:

	free(buffer.index_x_sorting_buffer);
release_page_locked_memory:

	cudaFreeHost(buffer.matrixA_buffer);

release_cuda_stream:
	for (int i = 0; i < numberOfThreads * 2; ++i)
	{
		cudaStreamDestroy(instance->stream[i]);
	}
	delete[] instance->stream;

release_worker_context:
	free(instance->workerContext.numberOfIteration);

release_instance:

	free(instance);
	return false;
}

void determineBeginMatrixAIndex(int *beginMatrixAIndex_M, int *beginMatrixAIndex_N, SearchType searchType,
	int indexOfWorker, int numberOfWorker,
	int matrixA_M, int matrixA_N,
	int matrixAPadding_M_pre, int matrixAPadding_M_post, int matrixAPadding_N_pre, int matrixAPadding_N_post,
	int searchRegion_M, int searchRegion_N)
{
	switch (searchType)
	{
	case SearchType::Local:
	{
		const int searchRegion_M_pre = searchRegion_M / 2,
			searchRegion_M_post = searchRegion_M - searchRegion_M_pre,
			searchRegion_N_pre = searchRegion_N / 2,
			searchRegion_N_post = searchRegion_N - searchRegion_N_pre;

		const int matrixAIndex_M_begin = 


	}
	break;
	case SearchType::Global:

		break;
	default: break;
	}
}

void initializeInstanceWorkerContext(BlockMatchContext *context)
{
	const int numberOfThreads = context->numberOfThreads;
	char *beginPointer = reinterpret_cast<char*>(context) + sizeof(BlockMatchContext);

	for (ptrdiff_t offset = 0; offset < sizeof(BlockMatchContext::PerThreadBufferPointer); offset += sizeof(void*))
	{
		void** pointer = reinterpret_cast<void**>(reinterpret_cast<char*>(&context->perThreadBufferPointer) + offset);
		*pointer = beginPointer + numberOfThreads * offset;
	}

	BlockMatchContext::PerThreadBufferPointer &perThreadBufferPointer = context->perThreadBufferPointer;
	BlockMatchContext::WorkerContext &workerContext = context->workerContext;

	const int numberOfSubmitProcessors = context->numberOfSubmitProcessors;
	const int numberOfSubmitThreadsPerProcessor = context->numberOfSubmitThreadsPerProcessor;
	const int block_M = context->block_M;
	const int block_N = context->block_N;

	const size_t sizeOfTaskQueue = numberOfSubmitProcessors * numberOfSubmitThreadsPerProcessor;
	const size_t sizeOfTaskSourceData = sizeOfTaskQueue * block_M * block_N;

	BlockMatchContext::Buffer &buffer = context->buffer;

	const int numberOfBlockBPerBlockA_M = context->numberOfBlockBPerBlockA_M;
	const int numberOfBlockBPerBlockA_N = context->numberOfBlockBPerBlockA_N;
	const int matrixC_M = context->C_dimensions[0], matrixC_N = context->C_dimensions[1], matrixC_O = context->C_dimensions[2];
	const int numberOfTasks = matrixC_M * matrixC_N;
	const int numberOfTasksPerWorker_minimum = numberOfTasks / numberOfThreads;

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		perThreadBufferPointer.matrixA_buffer[indexOfThread] = buffer.matrixA_buffer +
			indexOfThread * sizeOfTaskSourceData;

		perThreadBufferPointer.matrixB_buffer[indexOfThread] = buffer.matrixB_buffer +
			indexOfThread * sizeOfTaskSourceData;

		perThreadBufferPointer.matrixC_buffer[indexOfThread] = buffer.matrixC_buffer + indexOfThread * sizeOfTaskQueue;
		perThreadBufferPointer.matrixA_deviceBuffer[indexOfThread] = buffer.matrixA_deviceBuffer + indexOfThread * sizeOfTaskSourceData;
		perThreadBufferPointer.matrixB_deviceBuffer[indexOfThread] = buffer.matrixB_deviceBuffer + indexOfThread * sizeOfTaskSourceData;
		perThreadBufferPointer.matrixC_deviceBuffer[indexOfThread] = buffer.matrixC_deviceBuffer + indexOfThread * sizeOfTaskQueue;

		perThreadBufferPointer.index_x_sorting_buffer[indexOfThread] = buffer.index_x_sorting_buffer + indexOfThread * sizeOfTaskQueue;
		perThreadBufferPointer.index_y_sorting_buffer[indexOfThread] = buffer.index_y_sorting_buffer + indexOfThread * sizeOfTaskQueue;

		perThreadBufferPointer.index_raw_sorting_buffer[indexOfThread] = buffer.index_raw_sorting_buffer + indexOfThread * context->numberOfBlockBPerBlockA;

		// TODO
		workerContext.numberOfIteration[indexOfThread] = numberOfTasksPerWorker_minimum;
		workerContext.rawMatrixCIndex_begin[indexOfThread] = indexOfThread * numberOfTasksPerWorker_minimum;
		//workerContext.beginMatrixAIndex_M[indexOfThread] = ;
		//workerContext.beginMatrixAIndex_N[indexOfThread];
	}
}

void zeroInstanceOptionalInformation(BlockMatchContext *context)
{
	memset(&context->optionalBuffer, 0, sizeof(context->optionalBuffer));
}

bool allocateInternalBuffer(void **buffer, size_t size)
{
	float *buffer_ = static_cast<float*>(malloc(size));
	if (buffer_ == nullptr)
		return false;

	*buffer = buffer_;

	return true;
}

size_t getMatrixAPaddedSizeInBytes(const BlockMatchContext *context)
{
	return context->matrixA_padded_M * context->matrixA_padded_N * sizeof(float);
}

size_t getMatrixBPaddedSizeInBytes(const BlockMatchContext *context)
{
	return context->matrixB_padded_M * context->matrixB_padded_N * sizeof(float);
}

bool allocateMatrixAPaddedInternalBuffer(BlockMatchContext *context)
{
	size_t size = getMatrixAPaddedSizeInBytes(context);

	return allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.matrixA_padded_internal), size);
}

bool allocateMatrixBPaddedInternalBuffer(BlockMatchContext *context)
{
	size_t size = getMatrixBPaddedSizeInBytes(context);

	return allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.matrixB_padded_internal), size);
}

// TODO support 4D C
size_t getIndexSizeInBytes(BlockMatchContext *context)
{
	return context->C_dimensions[0] * context->C_dimensions[1] * context->C_dimensions[2];
}

bool allocateIndexXInternal(BlockMatchContext *context)
{
	size_t size = getIndexSizeInBytes(context);

	return allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.index_x_internal), size);
}

bool allocateIndexYInternal(BlockMatchContext *context)
{
	size_t size = getIndexSizeInBytes(context);

	return allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.index_y_internal), size);
}

bool allocateInternalBuffer(BlockMatchContext *context, enum class InternalBufferType bufferType)
{
	switch (bufferType)
	{
	case InternalBufferType::MatrixA_Padded_Buffer:
		return allocateMatrixAPaddedInternalBuffer(context);
	case InternalBufferType::MatrixB_Padded_Buffer:
		return allocateMatrixBPaddedInternalBuffer(context);
	case InternalBufferType::Index_X_Internal:
		return allocateIndexXInternal(context);
	case InternalBufferType::Index_Y_Internal:
		return allocateIndexYInternal(context);
	default:
		return false;
	}
}



void initializeWorkerInternalBuffer(BlockMatchContext *context, void *buffer, enum class InternalBufferType bufferType)
{
	size_t size = context->C_dimensions[2];

	switch (bufferType)
	{
	case InternalBufferType::Index_X_Internal:
		for (int indexOfThreads = 0; indexOfThreads < context->numberOfThreads; ++indexOfThreads)
		{
			context->optionalPerThreadBufferPointer.index_x_internal[indexOfThreads] = static_cast<int*>(buffer) +
				context->workerContext.rawMatrixCIndex_begin[indexOfThreads] * size;
		}
		break;
	case InternalBufferType::Index_Y_Internal:
		for (int indexOfThreads = 0; indexOfThreads < context->numberOfThreads; ++indexOfThreads)
		{
			context->optionalPerThreadBufferPointer.index_y_internal[indexOfThreads] = static_cast<int*>(buffer) +
				context->workerContext.rawMatrixCIndex_begin[indexOfThreads] * size;
		}
		break;
	default: break;
	}
}



BlockMatchContext * allocateContext(const int numberOfThreads)
{
	return static_cast<BlockMatchContext*>(malloc(sizeof(BlockMatchContext) +
		(sizeof(BlockMatchContext::PerThreadBufferPointer) + sizeof(BlockMatchContext::OptionalPerThreadBufferPointer)) * numberOfThreads));
}

bool blockMatchAndSortingInitialize(void **LIB_MATCH_OUT(instance),
	int matrixA_M, int matrixA_N, int matrixB_M, int matrixB_N,
	int searchRegion_M, int searchRegion_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int matrixAPadding_M_pre, int matrixAPadding_M_post,
	int matrixAPadding_N_pre, int matrixAPadding_N_post,
	int matrixBPadding_M_pre, int matrixBPadding_M_post,
	int matrixBPadding_N_pre, int matrixBPadding_N_post,
	int numberOfIndexRetain,
	int *LIB_MATCH_OUT(matrixC_M), int *LIB_MATCH_OUT(matrixC_N), int *LIB_MATCH_OUT(matrixC_O),
	int *LIB_MATCH_OUT(matrixA_padded_M) = nullptr, int *LIB_MATCH_OUT(matrixA_padded_N) = nullptr,
	int *LIB_MATCH_OUT(matrixB_padded_M) = nullptr, int *LIB_MATCH_OUT(matrixB_padded_N) = nullptr)
{
	// TODO glocal and local
	const int numberOfBlockBPerBlockA_M = getLength(matrixB_M, matrixBPadding_M_pre, matrixBPadding_N_post, block_M, strideB_M);
	const int numberOfBlockBPerBlockA_N = getLength(matrixB_N, matrixBPadding_N_pre, matrixBPadding_N_post, block_N, strideB_N);

	const int matrixC_M = getLength(matrixA_M, matrixAPadding_M_pre, matrixAPadding_M_post, block_M, strideA_M);
	const int matrixC_N = getLength(matrixA_N, matrixAPadding_N_pre, matrixAPadding_N_post, block_N, strideA_N);
	const int matrixC_O = determineSizeOfMatrixC_O(numberOfIndexRetain, numberOfBlockBPerBlockA_M, numberOfBlockBPerBlockA_N);

	// In case number of threads > size of A
	const int numberOfThreads = determineNumberOfThreads(matrixC_M, matrixC_N, globalContext.numberOfThreads);

	BlockMatchContext * instance = allocateContext(numberOfThreads);
	if (!instance) {
		setLastErrorString("Error: memory allocation failed");
		return false;
	}

	const int matrixA_padded_M = matrixA_M + matrixAPadding_M_pre + matrixAPadding_M_post;
	const int matrixA_padded_N = matrixA_N + matrixAPadding_N_pre + matrixAPadding_N_post;
	const int matrixB_padded_M = matrixB_M + matrixBPadding_M_pre + matrixBPadding_M_post;
	const int matrixB_padded_N = matrixB_N + matrixBPadding_N_pre + matrixBPadding_N_post;

	initializeBasicInstanceInformation(instance,
		matrixA_M, matrixA_N, matrixB_M, matrixB_N,
		searchRegion_M, searchRegion_N,
		block_M, block_N,
		strideA_M, strideA_N,
		strideB_M, strideB_N,
		matrixAPadding_M_pre, matrixAPadding_M_post,
		matrixAPadding_N_pre, matrixAPadding_N_post,
		matrixBPadding_M_pre, matrixBPadding_M_post,
		matrixBPadding_N_pre, matrixBPadding_N_post,
		numberOfIndexRetain,
		matrixA_padded_M, matrixA_padded_N,
		matrixB_padded_M, matrixB_padded_N
	);

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	const int numberOfBlockBPerBlockA = numberOfBlockBPerBlockA_M * numberOfBlockBPerBlockA_N;

	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;

	determineGpuTaskConfiguration(numberOfGPUProcessorThread, numberOfGPUDeviceMultiProcessor, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &sizeOfGpuTaskQueue);

	instance->C_dimensions[0] = matrixC_M;
	instance->C_dimensions[1] = matrixC_N;
	instance->C_dimensions[2] = matrixC_O;

	instance->numberOfSubmitThreadsPerProcessor = numberOfSubmitThreadsPerProcessor;
	instance->numberOfSubmitProcessors = numberOfSubmitProcessors;
	instance->sizeOfGpuTaskQueue = sizeOfGpuTaskQueue;

	instance->numberOfBlockBPerBlockA_M = numberOfBlockBPerBlockA_M;
	instance->numberOfBlockBPerBlockA_N = numberOfBlockBPerBlockA_N;
	instance->numberOfBlockBPerBlockA = numberOfBlockBPerBlockA;
	instance->numberOfThreads = numberOfThreads;

	if (numberOfIndexRetain > numberOfBlockBPerBlockA)
	{
		setLastErrorString("Check Error: Parameter 'retain' cannot larger than number of blocks of B");
		return false;
	}

	if (!initializeMemoryResources(instance)) {
		setLastErrorString("Error: memory allocation failed");
		return false;
	}

	zeroInstanceOptionalInformation(instance);

	initializeInstanceWorkerContext(instance);

	*LIB_MATCH_OUT(instance) = instance;

	if (LIB_MATCH_OUT(matrixC_M) != nullptr)
	{
		*LIB_MATCH_OUT(matrixC_M) = matrixC_M;
		*LIB_MATCH_OUT(matrixC_N) = matrixC_N;
		*LIB_MATCH_OUT(matrixC_O) = matrixC_O;
	}
	if (LIB_MATCH_OUT(matrixA_padded_M) != nullptr)
	{
		*LIB_MATCH_OUT(matrixA_padded_M) = matrixA_padded_M;
		*LIB_MATCH_OUT(matrixA_padded_N) = matrixA_padded_N;
	}
	if (LIB_MATCH_OUT(matrixB_padded_M) != nullptr)
	{
		*LIB_MATCH_OUT(matrixB_padded_M) = matrixB_padded_M;
		*LIB_MATCH_OUT(matrixB_padded_N) = matrixB_padded_N;
	}

	return true;
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
{/*
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
		retain);*/
}