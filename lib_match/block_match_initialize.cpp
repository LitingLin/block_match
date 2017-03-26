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
			*numberOfSubmitProcessors = (int)std::ceil(numberOfProcessorPerBlockA);
			*numberOfIterations = 1;
		}
	}
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

template <typename Type>
void initializeInstanceWorkerContext(BlockMatchContext<Type> *context,
	BorderType sequenceABorderType)
{
	const int numberOfThreads = context->numberOfThreads;

	std::vector<typename BlockMatchContext<Type>::WorkerContext> &workerContext = context->workerContext;
	
	const int numberOfSubmitProcessors = context->numberOfSubmitProcessors;
	const int numberOfSubmitThreadsPerProcessor = context->numberOfSubmitThreadsPerProcessor;
	const int block_M = context->block_M;
	const int block_N = context->block_N;

	const size_t sizeOfTaskQueue = context->sizeOfGpuTaskQueue * context->numberOfBlockBPerBlockA;
	const size_t sizeOfTaskSourceData = sizeOfTaskQueue * block_M * block_N;
	
	const int numberOfBlockBPerBlockA_M = context->numberOfBlockBPerBlockA_M;
	const int numberOfBlockBPerBlockA_N = context->numberOfBlockBPerBlockA_N;
	const int matrixC_M = context->C_dimensions[0], matrixC_N = context->C_dimensions[1], matrixC_O = context->C_dimensions[2];
	const int numberOfTasks = matrixC_M * matrixC_N;
	const int numberOfTasksPerWorker_minimum = numberOfTasks / numberOfThreads;
	const int strideA_M = context->strideA_M;
	const int strideA_N = context->strideA_N;

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		int rawMatrixCIndex_begin = indexOfThread * numberOfTasksPerWorker_minimum;
		int indexC_M = rawMatrixCIndex_begin / matrixC_N;
		int indexC_N = rawMatrixCIndex_begin % matrixC_N;
		int beginMatrixAIndex_M = indexC_M * strideA_M + context->indexA_M_begin;
		int beginMatrixAIndex_N = indexC_N * strideA_N + context->indexA_N_begin;
		if (sequenceABorderType == BorderType::includeLastBlock) {
			if (beginMatrixAIndex_M >= context->indexA_M_end)
				beginMatrixAIndex_M = context->indexA_M_end - 1;
			if (beginMatrixAIndex_N >= context->indexA_N_end)
				beginMatrixAIndex_N = context->indexA_N_end - 1;
		}

		workerContext.emplace_back(BlockMatchContext<Type>::WorkerContext{
			numberOfTasksPerWorker_minimum, // numberOfIteration
			rawMatrixCIndex_begin,
			beginMatrixAIndex_M,
			beginMatrixAIndex_N,
			std::make_unique<ExecutionContext<Type>>()
		});
	}
	workerContext[numberOfThreads - 1].numberOfIteration += (numberOfTasks - numberOfThreads * numberOfTasksPerWorker_minimum);
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
void blockMatchInitialize(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	SearchFrom searchFrom,
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
	int indexA_M_begin = 0, indexA_M_end = 0, indexA_N_begin = 0, indexA_N_end = 0;

	const int matrixA_padded_M = matrixA_M + matrixAPadding_M_pre + matrixAPadding_M_post;
	const int matrixA_padded_N = matrixA_N + matrixAPadding_N_pre + matrixAPadding_N_post;
	const int matrixB_padded_M = matrixB_M + matrixBPadding_M_pre + matrixBPadding_M_post;
	const int matrixB_padded_N = matrixB_N + matrixBPadding_N_pre + matrixBPadding_N_post;

	if (searchType == SearchType::local)
	{
		if (searchFrom == SearchFrom::topLeft)
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
		}
		else if (searchFrom == SearchFrom::center)
		{
			indexA_M_begin = searchRegion_M / 2;
			// TODO: Border check
			if (matrixA_padded_M > matrixB_padded_M - (searchRegion_M - searchRegion_M / 2) + 1)
				indexA_M_end = determineEndOfIndex(matrixB_padded_M, block_M) - (searchRegion_M - searchRegion_M / 2) + 1;
			else
				indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M);

			indexA_N_begin = searchRegion_N / 2;
			if (matrixA_padded_N > matrixB_padded_N - (searchRegion_N - searchRegion_N / 2) + 1)
				indexA_N_end = determineEndOfIndex(matrixB_padded_N, block_N) - (searchRegion_N - searchRegion_N / 2) + 1;
			else
				indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N);
		}
		else
		{
			NOT_IMPLEMENTED_ERROR;
		}
	}
	else
	{
		indexA_M_begin = 0;
		indexA_M_end = determineEndOfIndex(matrixA_padded_M, block_M);
		indexA_N_begin = 0;
		indexA_N_end = determineEndOfIndex(matrixA_padded_N, block_N);
	}

	if (indexA_M_end <= indexA_M_begin || indexA_N_end <= indexA_N_begin)
		throw std::exception("Parameter 'blockSize' is too large.");

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

	PadFunction<Type> *padFunctionA = nullptr;
	PadFunction<Type> *padFunctionB = nullptr;
	ExecutionFunction<Type> *executionFunction = nullptr;

	if (sort) {
		if (searchType == SearchType::local)
		{
			if (measureMethod == LibMatchMeasureMethod::mse)
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
			else if (measureMethod == LibMatchMeasureMethod::cc)
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
		}
		else if (searchType == SearchType::global)
		{
			if (measureMethod == LibMatchMeasureMethod::mse) {
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
			}
			else if (measureMethod == LibMatchMeasureMethod::cc) {
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						block_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
			}
			else
				NOT_IMPLEMENTED_ERROR;
		}
	}
	else
	{
		if (searchType == SearchType::local)
		{
			if (measureMethod == LibMatchMeasureMethod::mse)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_local,
					block_match_mse_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_local,
					block_match_mse_check_border, dummySort, tryToIncludeLastBlock>;
			else if (measureMethod == LibMatchMeasureMethod::cc)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_local,
					block_match_cc_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_local,
					block_match_cc_check_border, dummySort, tryToIncludeLastBlock>;
			else
				NOT_IMPLEMENTED_ERROR;
		}
		else if (searchType == SearchType::global)
		{
			if (measureMethod == LibMatchMeasureMethod::mse)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_full,
					block_match_mse_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_full,
					block_match_mse_check_border, dummySort, tryToIncludeLastBlock>;
			else if (measureMethod == LibMatchMeasureMethod::cc)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_full,
					block_match_cc_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_full,
					block_match_cc_check_border, dummySort, tryToIncludeLastBlock>;
			else
				NOT_IMPLEMENTED_ERROR;
		}
	}

	switch (padMethodA)
	{
	case PadMethod::zero:
		padFunctionA = zeroPadding<Type>;
		break;
	case PadMethod::circular:
		padFunctionA = circularPadding<Type>;
		break;
	case PadMethod::replicate:
		padFunctionA = replicatePadding<Type>;
		break;
	case PadMethod::symmetric:
		padFunctionA = symmetricPadding<Type>;
		break;
	default: break;
	}

	switch (padMethodB)
	{
	case PadMethod::zero:
		padFunctionB = zeroPadding<Type>;
		break;
	case PadMethod::circular:
		padFunctionB = circularPadding<Type>;
		break;
	case PadMethod::replicate:
		padFunctionB = replicatePadding<Type>;
		break;
	case PadMethod::symmetric:
		padFunctionB = symmetricPadding<Type>;
		break;
	default: break;
	}
	
	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	const int numberOfBlockBPerBlockA = numberOfBlockBPerBlockA_M * numberOfBlockBPerBlockA_N;

	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;

	determineGpuTaskConfiguration(numberOfGPUProcessorThread, numberOfGPUDeviceMultiProcessor, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &sizeOfGpuTaskQueue);
	
	if (numberOfIndexRetain > numberOfBlockBPerBlockA)
		throw std::runtime_error("Check Error: Parameter 'retain' cannot larger than number of blocks of B");

	const int perThreadMatrixCBufferSize = sizeOfGpuTaskQueue * numberOfBlockBPerBlockA;
	const int perThreadMatrixABufferSize = perThreadMatrixCBufferSize * block_M * block_N;

	BlockMatchContext<Type> *instance = new BlockMatchContext<Type>{
		matrixA_M, matrixA_N, matrixB_M, matrixB_N,
	matrixA_padded_M, matrixA_padded_N, matrixB_padded_M, matrixB_padded_N,
	block_M, block_N,
	searchRegion_M, searchRegion_N,
	strideA_M, strideA_N, strideB_M, strideB_N,
	matrixAPadding_M_pre, matrixAPadding_M_post,matrixAPadding_N_pre, matrixAPadding_N_post,
	matrixBPadding_M_pre, matrixBPadding_M_post, matrixBPadding_N_pre, matrixBPadding_N_post,
	indexA_M_begin, indexA_M_end, indexA_N_begin, indexA_N_end,
	numberOfIndexRetain,
	numberOfThreads,
	padFunctionA, padFunctionB , executionFunction,
		numberOfBlockBPerBlockA_M,numberOfBlockBPerBlockA_N,numberOfBlockBPerBlockA,
		{matrixC_M, matrixC_N, matrixC_X},
		std::vector<cudaStreamWarper>(numberOfThreads), // streams
		numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue,
		std::vector<void *>(), // threadPoolTaskHandle
		system_memory_allocator<int>(numberOfBlockBPerBlockA), // common_buffer
		std::vector<typename BlockMatchContext<Type>::WorkerContext>(), // workerContext
		std::vector<typename BlockMatchContext<Type>::OptionalPerThreadBuffer>(), // optionalPerThreadBuffer
		std::vector<typename BlockMatchContext<Type>::OptionalBuffer>(), // optionalBuffer
		std::vector<typename BlockMatchContext<Type>::PerThreadBuffer>() // perThreadBuffer
	};

	instance->workerContext.reserve(numberOfThreads);
	instance->threadPoolTaskHandle.reserve(numberOfThreads);
	instance->streams.resize(numberOfThreads);
	instance->perThreadBuffer.reserve(numberOfThreads);

	for (int i = 0; i < numberOfThreads; ++i)
	{
		instance->perThreadBuffer.emplace_back(BlockMatchContext<Type>::PerThreadBuffer{
			page_locked_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixA_buffer
			page_locked_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixB_buffer
			page_locked_memory_allocator<Type>(perThreadMatrixCBufferSize), // matrixC_buffer
			gpu_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixA_deviceBuffer
			gpu_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixB_deviceBuffer
			gpu_memory_allocator<Type>(perThreadMatrixCBufferSize), // matrixC_deviceBuffer
			system_memory_allocator<int>(perThreadMatrixCBufferSize), // index_x_sorting_buffer
			system_memory_allocator<int>(perThreadMatrixCBufferSize), // index_y_sorting_buffer
			system_memory_allocator<int>(numberOfBlockBPerBlockA) // index_raw_sorting_buffer
		});
	}

	initializeInstanceWorkerContext(instance, sequenceABorderType);

	instance->common_buffer.alloc();
	generateIndexSequence(instance->common_buffer.get(), numberOfBlockBPerBlockA);

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
}

LIB_MATCH_EXPORT
template
void blockMatchInitialize<float>(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	SearchFrom searchFrom,
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
void blockMatchInitialize<double>(void **LIB_MATCH_OUT(instance),
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	SearchFrom searchFrom,
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
