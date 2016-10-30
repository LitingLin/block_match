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

template <typename Type>
void initializeBasicInstanceInformation(BlockMatchContext<Type> *instance,
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

int determineNumberOfThreads(bool sort,
	const int A_M, const int A_N,
	const int maxNumberOfThreads)
{
	if (sort) {
		if (A_M * A_N < maxNumberOfThreads)
			return A_M * A_N;
		else
			return maxNumberOfThreads;
	}
	else
		if (2 <= maxNumberOfThreads)
			return 2;
		else
			return 1;
}

int determineSizeOfMatrixC_X(int numberOfIndexRetain, int group_M, int group_N)
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
template <typename Type>
bool initializeMemoryResources(BlockMatchContext<Type> *instance)
{
	const int block_M = instance->block_M;
	const int block_N = instance->block_N;
	const int numberOfBlockBPerBlockA = instance->numberOfBlockBPerBlockA;

	const int numberOfThreads = instance->numberOfThreads;
	const int numberOfGPUDeviceMultiProcessor = instance->numberOfSubmitProcessors;
	const int numberOfGPUProcessorThread = instance->numberOfSubmitThreadsPerProcessor;

	const int bufferSize = instance->sizeOfGpuTaskQueue * instance->numberOfBlockBPerBlockA * numberOfThreads;
	const int matBufferSize = bufferSize * block_M * block_N;

	typename BlockMatchContext<Type>::WorkerContext &workerContext = instance->workerContext;
	workerContext.numberOfIteration = static_cast<int*>(malloc(numberOfThreads * sizeof(int) * sizeof(BlockMatchContext<Type>::WorkerContext) / sizeof(int*)
		+ numberOfThreads * sizeof(void*)
		+ numberOfThreads * sizeof(ExecutionContext<Type>)));
	if (workerContext.numberOfIteration == nullptr)
		goto failed;

	workerContext.rawMatrixCIndex_begin = workerContext.numberOfIteration + numberOfThreads;
	workerContext.beginMatrixAIndex_M = workerContext.rawMatrixCIndex_begin + numberOfThreads;
	workerContext.beginMatrixAIndex_N = workerContext.beginMatrixAIndex_M + numberOfThreads;
	instance->threadPoolTaskHandle = reinterpret_cast<void**>(workerContext.beginMatrixAIndex_N + numberOfThreads);
	workerContext.executionContext = reinterpret_cast<ExecutionContext<Type>*>(instance->threadPoolTaskHandle + numberOfThreads);

	typename BlockMatchContext<Type>::Buffer &buffer = instance->buffer;

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
		(matBufferSize * 2 + bufferSize) * sizeof(Type)
	);

	if (cuda_error != cudaSuccess)
		goto release_cuda_stream;

	buffer.matrixB_buffer = buffer.matrixA_buffer + matBufferSize;
	buffer.matrixC_buffer = buffer.matrixB_buffer + matBufferSize;

	buffer.index_x_sorting_buffer = static_cast<int *>(malloc(
		bufferSize * 2 * sizeof(int) +
		numberOfBlockBPerBlockA * (numberOfThreads + 1) * sizeof(int)));

	if (buffer.index_x_sorting_buffer == nullptr) {
		goto release_page_locked_memory;
	}

	buffer.index_y_sorting_buffer = buffer.index_x_sorting_buffer + bufferSize;
	buffer.common_buffer = buffer.index_y_sorting_buffer + bufferSize;
	buffer.index_raw_sorting_buffer = buffer.common_buffer + numberOfBlockBPerBlockA;

	cuda_error = cudaMalloc(&buffer.matrixA_deviceBuffer,
		(matBufferSize * 2 + bufferSize) * sizeof(Type));
	if (cuda_error != cudaSuccess)
		goto release_memory;

	buffer.matrixB_deviceBuffer = buffer.matrixA_deviceBuffer + matBufferSize;
	buffer.matrixC_deviceBuffer = buffer.matrixB_deviceBuffer + matBufferSize;

	return true;

release_device_memory:
	cudaFree(buffer.matrixA_deviceBuffer);

release_memory:
	free(buffer.common_buffer);

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

template <typename Type>
void initializeInstanceWorkerContext(BlockMatchContext<Type> *context,
	BorderType sequenceABorderType)
{
	const int numberOfThreads = context->numberOfThreads;
	context->perThreadBufferPointer =
		reinterpret_cast<typename BlockMatchContext<Type>::PerThreadBufferPointer*>(reinterpret_cast<char*>(context) + sizeof(BlockMatchContext<Type>));
	context->optionalPerThreadBufferPointer =
		reinterpret_cast<typename BlockMatchContext<Type>::OptionalPerThreadBufferPointer*>
		(reinterpret_cast<char*>(context) + sizeof(BlockMatchContext<Type>) + sizeof(typename BlockMatchContext<Type>::PerThreadBufferPointer) * numberOfThreads);

	typename BlockMatchContext<Type>::PerThreadBufferPointer* &perThreadBufferPointer = context->perThreadBufferPointer;
	typename BlockMatchContext<Type>::WorkerContext &workerContext = context->workerContext;

	const int numberOfSubmitProcessors = context->numberOfSubmitProcessors;
	const int numberOfSubmitThreadsPerProcessor = context->numberOfSubmitThreadsPerProcessor;
	const int block_M = context->block_M;
	const int block_N = context->block_N;

	const size_t sizeOfTaskQueue = context->sizeOfGpuTaskQueue * context->numberOfBlockBPerBlockA;
	const size_t sizeOfTaskSourceData = sizeOfTaskQueue * block_M * block_N;

	typename BlockMatchContext<Type>::Buffer &buffer = context->buffer;

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
		int indexC_M = workerContext.rawMatrixCIndex_begin[indexOfThread] / matrixC_N;
		int indexC_N = workerContext.rawMatrixCIndex_begin[indexOfThread] % matrixC_N;
		workerContext.beginMatrixAIndex_M[indexOfThread] = indexC_M * strideA_M + context->indexA_M_begin;
		workerContext.beginMatrixAIndex_N[indexOfThread] = indexC_N * strideA_N + context->indexA_N_begin;
		if (sequenceABorderType == BorderType::includeLastBlock) {
			if (workerContext.beginMatrixAIndex_M[indexOfThread] >= context->indexA_M_end)
				workerContext.beginMatrixAIndex_M[indexOfThread] = context->indexA_M_end - 1;
			if (workerContext.beginMatrixAIndex_N[indexOfThread] >= context->indexA_N_end)
				workerContext.beginMatrixAIndex_N[indexOfThread] = context->indexA_N_end - 1;
		}
	}
	workerContext.numberOfIteration[numberOfThreads - 1] += (numberOfTasks - numberOfThreads * numberOfTasksPerWorker_minimum);
}

template <typename Type>
void zeroInstanceOptionalInformation(BlockMatchContext<Type> *context)
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

template <typename Type>
size_t getMatrixAPaddedSizeInBytes(const BlockMatchContext<Type> *context)
{
	return context->matrixA_padded_M * context->matrixA_padded_N * sizeof(float);
}

template <typename Type>
size_t getMatrixBPaddedSizeInBytes(const BlockMatchContext<Type> *context)
{
	return context->matrixB_padded_M * context->matrixB_padded_N * sizeof(float);
}

template <typename Type>
bool allocateMatrixAPaddedInternalBuffer(BlockMatchContext<Type> *context)
{
	size_t size = getMatrixAPaddedSizeInBytes(context);

	return allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.matrixA_padded_internal), size);
}

template <typename Type>
bool allocateMatrixBPaddedInternalBuffer(BlockMatchContext<Type> *context)
{
	size_t size = getMatrixBPaddedSizeInBytes(context);

	return allocateInternalBuffer(reinterpret_cast<void**>(&context->optionalBuffer.matrixB_padded_internal), size);
}

// TODO support 4D C
template <typename Type>
size_t getIndexSizeInBytes(BlockMatchContext<Type> *context)
{
	return context->C_dimensions[0] * context->C_dimensions[1] * context->C_dimensions[2];
}

template <typename Type>
void initializeWorkerInternalBuffer(BlockMatchContext<Type> *context, void *buffer, enum class InternalBufferType bufferType)
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

template <typename Type>
bool allocateIndexXInternal(BlockMatchContext<Type> *context)
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

template <typename Type>
bool allocateIndexYInternal(BlockMatchContext<Type> *context)
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

template <typename Type>
bool allocateInternalBuffer(BlockMatchContext<Type> *context, enum class InternalBufferType bufferType)
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

template <typename Type>
BlockMatchContext<Type> * allocateContext(const int numberOfThreads)
{
	return static_cast<BlockMatchContext<Type>*>(malloc(sizeof(BlockMatchContext<Type>) +
		(sizeof(typename BlockMatchContext<Type>::PerThreadBufferPointer) +
			sizeof(typename BlockMatchContext<Type>::OptionalPerThreadBufferPointer)) * numberOfThreads));
}

int determineNumberOfBlockBPerBlockA(SearchType searchType, int searchRegion,
	int matrixB, int matrixBPadding_pre, int matrixBPadding_post, int block, int strideB)
{
	if (searchType == SearchType::local)
	{
		return (searchRegion + strideB - 1) / strideB;
	}
	else
	{
		return getLength(matrixB, matrixBPadding_pre, matrixBPadding_post, block, strideB);
	}
}


template <typename Type>
bool blockMatchInitialize(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	bool sort,
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
	int *LIB_MATCH_OUT(matrixC_M), int *LIB_MATCH_OUT(matrixC_N), int *LIB_MATCH_OUT(matrixC_X),
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
		indexA_M_begin = 0;
		if (matrixA_padded_M > (matrixB_padded_M - searchRegion_M + 1))
			indexA_M_end = determineEndOfIndex(matrixB_padded_M, block_M) - searchRegion_M + 1;
		else
			indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M);
		indexA_N_begin = 0;
		if (matrixA_padded_N > (matrixB_padded_N - searchRegion_N + 1))
			indexA_N_end = determineEndOfIndex(matrixB_padded_N, block_N) - searchRegion_N + 1;
		else
			indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N);

		/*
		indexA_M_begin = searchRegion_M / 2;
		if (matrixA_padded_M > matrixB_padded_M)
			indexA_M_end = determineEndOfIndex(matrixB_padded_M, block_M) - (searchRegion_M - searchRegion_M / 2) + 1;
		else
			indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M) - (searchRegion_M - searchRegion_M / 2) + 1;
		indexA_N_begin = searchRegion_N / 2;
		if (matrixA_padded_N > matrixB_padded_N)
			indexA_N_end = determineEndOfIndex(matrixB_padded_N, block_N) - (searchRegion_N - searchRegion_N / 2) + 1;
		else
			indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N) - (searchRegion_N - searchRegion_N / 2) + 1;*/
	}
	else
	{
		indexA_M_begin = 0;
		indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M);
		indexA_N_begin = 0;
		indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N);
	}

	int matrixC_M = (indexA_M_end - indexA_M_begin + strideA_M - 1) / strideA_M;
	int matrixC_N = (indexA_N_end - indexA_N_begin + strideA_N - 1) / strideA_N;
	const int matrixC_X = determineSizeOfMatrixC_X(numberOfIndexRetain, numberOfBlockBPerBlockA_M, numberOfBlockBPerBlockA_N);
	if (sequenceABorderType == BorderType::includeLastBlock) {
		if ((indexA_M_end - indexA_M_begin - 1) % strideA_M)
			++matrixC_M;

		if ((indexA_N_end - indexA_N_begin - 1) % strideA_N)
			++matrixC_N;
	}
	// In case number of threads > size of A
	const int numberOfThreads = determineNumberOfThreads(sort, matrixC_M, matrixC_N, globalContext.numberOfThreads);

	BlockMatchContext<Type> * instance = allocateContext<Type>(numberOfThreads);
	if (!instance) {
		setLastErrorString("Error: memory allocation failed");
		return false;
	}

	instance->indexA_M_begin = indexA_M_begin;
	instance->indexA_M_end = indexA_M_end;
	instance->indexA_N_begin = indexA_N_begin;
	instance->indexA_N_end = indexA_N_end;

	if (sort) {
		if (searchType == SearchType::local)
		{
			if (measureMethod == LibMatchMeasureMethod::mse)
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
			else if (measureMethod == LibMatchMeasureMethod::cc)
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_local,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
		}
		else if (searchType == SearchType::global)
		{
			if (measureMethod == LibMatchMeasureMethod::mse) {
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
			}
			else if (measureMethod == LibMatchMeasureMethod::cc) {
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
					else
						instance->executionMethod = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
			}
			else
				abort();
		}
	}
	else
	{
		if (searchType == SearchType::local)
		{
			if (measureMethod == LibMatchMeasureMethod::mse)
				if (sequenceABorderType == BorderType::normal)
					instance->executionMethod = processWorker<Type, determineBlockB_index_local,
					block_match_mse_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					instance->executionMethod = processWorker<Type, determineBlockB_index_local,
					block_match_mse_check_border, dummySort, tryToIncludeLastBlock>;
			else if (measureMethod == LibMatchMeasureMethod::cc)
				if (sequenceABorderType == BorderType::normal)
				instance->executionMethod = processWorker<Type, determineBlockB_index_local,
				block_match_cc_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					instance->executionMethod = processWorker<Type, determineBlockB_index_local,
					block_match_cc_check_border, dummySort, tryToIncludeLastBlock>;
			else
				abort();
		}
		else if (searchType == SearchType::global)
		{
			if (measureMethod == LibMatchMeasureMethod::mse)
				if (sequenceABorderType == BorderType::normal)
				instance->executionMethod = processWorker<Type, determineBlockB_index_full,
				block_match_mse_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					instance->executionMethod = processWorker<Type, determineBlockB_index_full,
					block_match_mse_check_border, dummySort, tryToIncludeLastBlock>;
			else if (measureMethod == LibMatchMeasureMethod::cc)
				if (sequenceABorderType == BorderType::normal)
				instance->executionMethod = processWorker<Type, determineBlockB_index_full,
				block_match_cc_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					instance->executionMethod = processWorker<Type, determineBlockB_index_full,
					block_match_cc_check_border, dummySort, tryToIncludeLastBlock>;
			else
				abort();
		}
	}

	switch (padMethodA)
	{
	case PadMethod::zero:
		instance->padMethodA = zeroPadding<Type>;
		break;
	case PadMethod::circular:
		instance->padMethodA = circularPadding<Type>;
		break;
	case PadMethod::replicate:
		instance->padMethodA = replicatePadding<Type>;
		break;
	case PadMethod::symmetric:
		instance->padMethodA = symmetricPadding<Type>;
		break;
	default: break;
	}

	switch (padMethodB)
	{
	case PadMethod::zero:
		instance->padMethodB = zeroPadding<Type>;
		break;
	case PadMethod::circular:
		instance->padMethodB = circularPadding<Type>;
		break;
	case PadMethod::replicate:
		instance->padMethodB = replicatePadding<Type>;
		break;
	case PadMethod::symmetric:
		instance->padMethodB = symmetricPadding<Type>;
		break;
	default: break;
	}

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
	instance->C_dimensions[2] = matrixC_X;

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

	initializeInstanceWorkerContext(instance, sequenceABorderType);

	generateIndexSequence(instance->buffer.common_buffer, numberOfBlockBPerBlockA);

	*LIB_MATCH_OUT(instance) = instance;

	if (LIB_MATCH_OUT(matrixC_M) != nullptr)
	{
		*LIB_MATCH_OUT(matrixC_M) = matrixC_M;
		*LIB_MATCH_OUT(matrixC_N) = matrixC_N;
		*LIB_MATCH_OUT(matrixC_X) = matrixC_X;
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

LIB_MATCH_EXPORT
template
bool blockMatchInitialize<float>(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	bool sort,
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
	int *LIB_MATCH_OUT(matrixB_padded_M), int *LIB_MATCH_OUT(matrixB_padded_N));

LIB_MATCH_EXPORT
template
bool blockMatchInitialize<double>(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	bool sort,
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
	int *LIB_MATCH_OUT(matrixB_padded_M), int *LIB_MATCH_OUT(matrixB_padded_N));

template
BlockMatchContext<float> * allocateContext(const int numberOfThreads);
template
BlockMatchContext<double> * allocateContext(const int numberOfThreads);
template
bool allocateInternalBuffer(BlockMatchContext<float> *context, enum class InternalBufferType bufferType);
template
bool allocateInternalBuffer(BlockMatchContext<double> *context, enum class InternalBufferType bufferType);
template
void initializeWorkerInternalBuffer(BlockMatchContext<float> *context, void *buffer, enum class InternalBufferType bufferType);
template
void initializeWorkerInternalBuffer(BlockMatchContext<double> *context, void *buffer, enum class InternalBufferType bufferType);
