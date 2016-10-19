#include "lib_match_internal.h"

#include <cuda_runtime.h>

#include "block_match_execute.hpp"

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

void initializeBasicInstanceInformation(BlockMatchContext *instance,
	const SearchType searchType,
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
	instance->searchType = searchType;

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

/*
 * Initialized parameter list:
 * WorkerContext:
 *  numberOfIteration
 *  rawMatrixCIndex_begin
 *  beginMatrixAIndex_M
 *  beginMatrixAIndex_N
 * stream
 * Buffer:
 *  matrixA_buffer
 *  matrixB_buffer
 *  matrixC_buffer
 *  index_x_sorting_buffer
 *  index_y_sorting_buffer
 *  common_buffer
 *  index_raw_sorting_buffer
 *  matrixA_deviceBuffer
 *  matrixB_deviceBuffer
 *  matrixC_deviceBuffer
 */
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
	workerContext.numberOfIteration = static_cast<int*>(malloc(numberOfThreads * sizeof(int) * sizeof(BlockMatchContext::WorkerContext) / sizeof(int*)
		+ numberOfThreads * sizeof(void*)
		+ numberOfThreads * sizeof(ExecutionContext)));
	if (workerContext.numberOfIteration == nullptr)
		goto failed;

	workerContext.rawMatrixCIndex_begin = workerContext.numberOfIteration + numberOfThreads;
	workerContext.beginMatrixAIndex_M = workerContext.rawMatrixCIndex_begin + numberOfThreads;
	workerContext.beginMatrixAIndex_N = workerContext.beginMatrixAIndex_M + numberOfThreads;
	instance->threadPoolTaskHandle = reinterpret_cast<void**>(workerContext.beginMatrixAIndex_N + numberOfThreads);
	workerContext.executionContext = reinterpret_cast<ExecutionContext*>(instance->threadPoolTaskHandle + numberOfThreads);

	BlockMatchContext::Buffer &buffer = instance->buffer;

	cudaError_t cuda_error;

	instance->stream = static_cast<cudaStream_t*>(malloc(numberOfThreads * 2 * sizeof(cudaStream_t)));
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
			free(instance->stream);
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

	buffer.index_x_sorting_buffer = static_cast<int *>(malloc(
		bufferSize * 2 * sizeof(int) +
		numberOfBlockBPerBlockA * (numberOfThreads + 1) * sizeof(int)));

	if (buffer.index_x_sorting_buffer == nullptr)
		goto release_page_locked_memory;

	buffer.index_y_sorting_buffer = buffer.index_x_sorting_buffer + bufferSize;

	buffer.common_buffer = buffer.index_y_sorting_buffer + bufferSize;
	buffer.index_raw_sorting_buffer = buffer.common_buffer + numberOfBlockBPerBlockA;

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
	free(instance->stream);

release_worker_context:
	free(instance->workerContext.numberOfIteration);

failed:
	return false;
}

void initializeInstanceWorkerContext(BlockMatchContext *context)
{
	const int numberOfThreads = context->numberOfThreads;
	context->perThreadBufferPointer =
		reinterpret_cast<BlockMatchContext::PerThreadBufferPointer*>(reinterpret_cast<char*>(context) + sizeof(BlockMatchContext));
	context->optionalPerThreadBufferPointer =
		reinterpret_cast<BlockMatchContext::OptionalPerThreadBufferPointer*>
		(reinterpret_cast<char*>(context) + sizeof(BlockMatchContext) + sizeof(BlockMatchContext::PerThreadBufferPointer) * numberOfThreads);

	BlockMatchContext::PerThreadBufferPointer* &perThreadBufferPointer = context->perThreadBufferPointer;
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
	const int strideA_M = context->strideA_M;
	const int strideA_N = context->strideA_N;

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		perThreadBufferPointer[indexOfThread].matrixA_buffer = buffer.matrixA_buffer +
			indexOfThread * sizeOfTaskSourceData;

		perThreadBufferPointer[indexOfThread].matrixB_buffer = buffer.matrixB_buffer +
			indexOfThread * sizeOfTaskSourceData;

		perThreadBufferPointer[indexOfThread].matrixC_buffer = buffer.matrixC_buffer + indexOfThread * sizeOfTaskQueue;
		perThreadBufferPointer[indexOfThread].matrixA_deviceBuffer = buffer.matrixA_deviceBuffer + indexOfThread * sizeOfTaskSourceData;
		perThreadBufferPointer[indexOfThread].matrixB_deviceBuffer = buffer.matrixB_deviceBuffer + indexOfThread * sizeOfTaskSourceData;
		perThreadBufferPointer[indexOfThread].matrixC_deviceBuffer = buffer.matrixC_deviceBuffer + indexOfThread * sizeOfTaskQueue;

		perThreadBufferPointer[indexOfThread].index_x_sorting_buffer = buffer.index_x_sorting_buffer + indexOfThread * sizeOfTaskQueue;
		perThreadBufferPointer[indexOfThread].index_y_sorting_buffer = buffer.index_y_sorting_buffer + indexOfThread * sizeOfTaskQueue;

		perThreadBufferPointer[indexOfThread].index_raw_sorting_buffer = buffer.index_raw_sorting_buffer + indexOfThread * context->numberOfBlockBPerBlockA;

		workerContext.numberOfIteration[indexOfThread] = numberOfTasksPerWorker_minimum;
		workerContext.rawMatrixCIndex_begin[indexOfThread] = indexOfThread * numberOfTasksPerWorker_minimum;
		int indexC_M = workerContext.rawMatrixCIndex_begin[indexOfThread] % matrixC_M;
		int indexC_N = workerContext.rawMatrixCIndex_begin[indexOfThread] / matrixC_M;
		workerContext.beginMatrixAIndex_M[indexOfThread] = indexC_M * strideA_M + context->indexA_M_begin;
		workerContext.beginMatrixAIndex_N[indexOfThread] = indexC_N * strideA_N + context->indexA_N_begin;
	}
	workerContext.numberOfIteration[numberOfThreads - 1] += (numberOfTasks - numberOfThreads * numberOfTasksPerWorker_minimum);
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

void initializeWorkerInternalBuffer(BlockMatchContext *context, void *buffer, enum class InternalBufferType bufferType)
{
	size_t size = context->C_dimensions[2];

	switch (bufferType)
	{
	case InternalBufferType::Index_X_Internal:
		for (int indexOfThreads = 0; indexOfThreads < context->numberOfThreads; ++indexOfThreads)
		{
			context->optionalPerThreadBufferPointer[indexOfThreads].index_x_internal = static_cast<int*>(buffer) +
				context->workerContext.rawMatrixCIndex_begin[indexOfThreads] * size;
		}
		break;
	case InternalBufferType::Index_Y_Internal:
		for (int indexOfThreads = 0; indexOfThreads < context->numberOfThreads; ++indexOfThreads)
		{
			context->optionalPerThreadBufferPointer[indexOfThreads].index_y_internal = static_cast<int*>(buffer) +
				context->workerContext.rawMatrixCIndex_begin[indexOfThreads] * size;
		}
		break;
	default: break;
	}
}

bool allocateIndexXInternal(BlockMatchContext *context)
{
	size_t size = getIndexSizeInBytes(context);

	if (allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.index_x_internal), size))
	{
		initializeWorkerInternalBuffer(context, context->optionalBuffer.index_x_internal, InternalBufferType::Index_X_Internal);
		return true;
	}
	else
		return false;
}

bool allocateIndexYInternal(BlockMatchContext *context)
{
	size_t size = getIndexSizeInBytes(context);

	if (allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.index_y_internal), size))
	{
		initializeWorkerInternalBuffer(context, context->optionalBuffer.index_y_internal, InternalBufferType::Index_Y_Internal);
		return true;
	}
	else
		return false;
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


BlockMatchContext * allocateContext(const int numberOfThreads)
{
	return static_cast<BlockMatchContext*>(malloc(sizeof(BlockMatchContext) +
		(sizeof(BlockMatchContext::PerThreadBufferPointer) + sizeof(BlockMatchContext::OptionalPerThreadBufferPointer)) * numberOfThreads));
}

int determineNumberOfBlockBPerBlockA(SearchType searchType, int searchRegion,
	int matrixB, int matrixBPadding_pre, int matrixBPadding_post, int block, int strideB)
{
	if (searchType == SearchType::local)
	{
		return searchRegion;
	}
	else
	{
		return getLength(matrixB, matrixBPadding_pre, matrixBPadding_post, block, strideB);
	}
}
bool blockMatchAndSortingInitialize(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethod,
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
	int *LIB_MATCH_OUT(matrixA_padded_M), int *LIB_MATCH_OUT(matrixA_padded_N),
	int *LIB_MATCH_OUT(matrixB_padded_M), int *LIB_MATCH_OUT(matrixB_padded_N))
{
	const int numberOfBlockBPerBlockA_M = determineNumberOfBlockBPerBlockA(searchType,
		searchRegion_M,
		matrixB_M, matrixBPadding_M_pre, matrixBPadding_M_post, block_M, strideB_M);
	const int numberOfBlockBPerBlockA_N = determineNumberOfBlockBPerBlockA(searchType,
		searchRegion_N,
		matrixB_N, matrixBPadding_N_pre, matrixBPadding_N_post, block_N, strideB_N);
	int indexA_M_begin, indexA_M_end, indexA_N_begin, indexA_N_end;

	const int matrixA_padded_M = matrixA_M + matrixAPadding_M_pre + matrixAPadding_M_post;
	const int matrixA_padded_N = matrixA_N + matrixAPadding_N_pre + matrixAPadding_N_post;
	const int matrixB_padded_M = matrixB_M + matrixBPadding_M_pre + matrixBPadding_M_post;
	const int matrixB_padded_N = matrixB_N + matrixBPadding_N_pre + matrixBPadding_N_post;

	if (searchType == SearchType::local)
	{
		indexA_M_begin = searchRegion_M / 2;
		if (matrixA_padded_M > matrixB_padded_M)
			indexA_M_end = determineEndOfIndex(matrixB_padded_M, block_M) - (searchRegion_M - searchRegion_M / 2) + 1;
		else
			indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M) - (searchRegion_M - searchRegion_M / 2) + 1;
		indexA_N_begin = searchRegion_N / 2;
		if (matrixA_padded_N > matrixB_padded_N)
			indexA_N_end = determineEndOfIndex(matrixB_padded_N, block_N) - (searchRegion_N - searchRegion_N / 2) + 1;
		else
			indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N) - (searchRegion_N - searchRegion_N / 2) + 1;
	}
	else
	{
		indexA_M_begin = 0;
		indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M);
		indexA_N_begin = 0;
		indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N);
	}

	const int matrixC_M = (indexA_M_end - indexA_M_begin + strideA_M - 1) / strideA_M;
	const int matrixC_N = (indexA_N_end - indexA_N_begin + strideA_N - 1) / strideA_N;
	const int matrixC_O = determineSizeOfMatrixC_O(numberOfIndexRetain, numberOfBlockBPerBlockA_M, numberOfBlockBPerBlockA_N);

	// In case number of threads > size of A
	const int numberOfThreads = determineNumberOfThreads(matrixC_M, matrixC_N, globalContext.numberOfThreads);

	BlockMatchContext * instance = allocateContext(numberOfThreads);
	if (!instance) {
		setLastErrorString("Error: memory allocation failed");
		return false;
	}

	instance->indexA_M_begin = indexA_M_begin;
	instance->indexA_M_end = indexA_M_end;
	instance->indexA_N_begin = indexA_N_begin;
	instance->indexA_N_end = indexA_N_end;

	if (searchType == SearchType::local)
	{
		if (measureMethod == LibMatchMeasureMethod::mse)
			if (numberOfIndexRetain)
				instance->executionMethod = processWorker<determineBlockB_index_local, recordIndex, block_match_mse_check_border, sortWithIndex_partial>;
			else
				instance->executionMethod = processWorker<determineBlockB_index_local, recordIndex, block_match_mse_check_border, sortWithIndex>;
		else if (measureMethod == LibMatchMeasureMethod::cc)
			if (numberOfIndexRetain)
				instance->executionMethod = processWorker<determineBlockB_index_local, recordIndex, block_match_cc_check_border, sortWithIndex_partial>;
			else
				instance->executionMethod = processWorker<determineBlockB_index_local, recordIndex, block_match_cc_check_border, sortWithIndex>;
	}
	else if (searchType == SearchType::global)
	{
		if (measureMethod == LibMatchMeasureMethod::mse)
			if (numberOfIndexRetain)
				instance->executionMethod = processWorker<determineBlockB_index_full, recordIndex, block_match_mse_check_border, sortWithIndex_partial>;
			else
				instance->executionMethod = processWorker<determineBlockB_index_full, recordIndex, block_match_mse_check_border, sortWithIndex>;
		else if (measureMethod == LibMatchMeasureMethod::cc)
			if (numberOfIndexRetain)
				instance->executionMethod = processWorker<determineBlockB_index_full, recordIndex, block_match_cc_check_border, sortWithIndex_partial>;
			else
				instance->executionMethod = processWorker<determineBlockB_index_full, recordIndex, block_match_cc_check_border, sortWithIndex>;
	}

	switch (padMethod)
	{
	case PadMethod::zero:
		instance->padMethod = zeroPadding<float>;
		break;
	case PadMethod::circular:
		instance->padMethod = circularPadding<float>;
		break;
	case PadMethod::replicate:
		instance->padMethod = replicatePadding<float>;
		break;
	case PadMethod::symmetric:
		instance->padMethod = symmetricPadding<float>;
		break;
	default: break;
	}

	initializeBasicInstanceInformation(instance,
		searchType,
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
		goto Failed;
	}

	if (!initializeMemoryResources(instance)) {
		setLastErrorString("Error: memory allocation failed");
		goto Failed;
	}

	zeroInstanceOptionalInformation(instance);

	initializeInstanceWorkerContext(instance);

	generateIndexSequence(instance->buffer.common_buffer, numberOfBlockBPerBlockA);

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

Failed:
	free(instance);
	return false;
}