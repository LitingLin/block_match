#include "lib_match_internal.h"

#include <cuda_runtime.h>

#include "lib_match_initialize.h"

#include "block_match_execute.h"
#include "block_match_execute.hpp"

int determineSizeOfMatrixC_X(int numberOfIndexRetain, int group_M, int group_N)
{
	if (numberOfIndexRetain)
		return numberOfIndexRetain;
	else
		return group_M * group_N;
}

template <typename Type>
void initializeInstanceWorkerContext(BlockMatchContext<Type> *context,
	BorderType sequenceABorderType, std::vector<int> deviceLists)
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
			deviceLists[0],
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
BlockMatch<Type>::BlockMatch(std::type_index inputADataType, std::type_index inputBDataType, 
	std::type_index outputDataType,
	std::type_index indexDataType,
	SearchType searchType,
	MeasureMethod measureMethod,
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
	int numberOfResultRetain,
	bool doThresholding,
	Type thresholdValue, Type replacementValue,
	bool indexStartFromOne,
	int indexOfDevice,
	unsigned numberOfThreads_byUser)
	: m_instance(nullptr), 
		inputADataType(inputADataType), inputBDataType(inputBDataType),
		outputDataType(outputDataType), indexDataType(indexDataType)
{
	CHECK_POINT(globalContext.hasGPU);

	CHECK_POINT_LT(indexOfDevice, globalContext.numberOfGPUDeviceMultiProcessor.size());

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
	const int matrixC_X = determineSizeOfMatrixC_X(numberOfResultRetain, numberOfBlockBPerBlockA_M, numberOfBlockBPerBlockA_N);
	if (sequenceABorderType == BorderType::includeLastBlock) {
		if ((indexA_M_end - indexA_M_begin - 1) % strideA_M)
			++matrixC_M;

		if ((indexA_N_end - indexA_N_begin - 1) % strideA_N)
			++matrixC_N;
	}
	if (!numberOfThreads_byUser)
		numberOfThreads_byUser = globalContext.numberOfThreads;

	// In case number of threads > size of A
	const int numberOfThreads = determineNumberOfThreads(sort, matrixC_M * matrixC_N, 
		globalContext.numberOfThreads < numberOfThreads_byUser ? globalContext.numberOfThreads : numberOfThreads_byUser);

	PadFunction *padFunctionA = nullptr;
	PadFunction *padFunctionB = nullptr;
	ExecutionFunction<Type> *executionFunction = nullptr;

	DataPostProcessingMethod<Type> *dataPostProcessingFunction = nullptr;
	BlockCopyMethod *blockCopyingAFunction = nullptr;
	BlockCopyMethod *blockCopyingBFunction = nullptr;
	DetermineBlockBRangeMethod *determineBlockBRangeFunction = nullptr;
	IterationIndexPostProcessMethod *iterationIndexPostProcessFunction = nullptr;
	IndexRecordMethod *indexRecordFunction = nullptr;

	if (sort)
	{
		if (indexDataType == typeid(nullptr)) 
		{
			if (measureMethod == MeasureMethod::mse && numberOfResultRetain)
			{
				if (doThresholding) {
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortPartialAscend<Type>, threshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
				else
				{
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortPartialAscend<Type>, noThreshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
			}
			else if (measureMethod == MeasureMethod::mse && !numberOfResultRetain)
			{
				if (doThresholding) {
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortAscend<Type>, threshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
				else {
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortAscend<Type>, noThreshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
			}
			else if (measureMethod == MeasureMethod::cc && numberOfResultRetain)
			{
				if (doThresholding) {
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortDescend<Type>, threshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
				else
				{
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortDescend<Type>, noThreshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp					
				}
			}
			else if (measureMethod == MeasureMethod::cc && !numberOfResultRetain)
			{
				if (doThresholding) {
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortPartialDescend<Type>, threshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
				else
				{
#define exp(type) \
	dataPostProcessingFunction = sort_noRecordIndex<Type, type, sortPartialDescend<Type>, noThreshold<Type>>
					RuntimeTypeInference(outputDataType, exp);
#undef exp
				}
			}
			else
				NOT_IMPLEMENTED_ERROR;
		}
		else
		{
			if (measureMethod == MeasureMethod::mse && numberOfResultRetain)
			{
				if (doThresholding) {
					if (indexStartFromOne) {
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialAscend<Type>, threshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialAscend<Type>, threshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
				}
				else
				{
					if (indexStartFromOne)
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialAscend<Type>, noThreshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialAscend<Type>, noThreshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
				}
			}
			else if (measureMethod == MeasureMethod::mse && !numberOfResultRetain)
			{
				if (doThresholding) {
					if (indexStartFromOne) {
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortAscend<Type>, threshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortAscend<Type>, threshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp						
					}
				}
				else
				{
					if (indexStartFromOne)
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortAscend<Type>, noThreshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp						
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortAscend<Type>, noThreshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp						
					}					
				}
			}
			else if (measureMethod == MeasureMethod::cc && numberOfResultRetain)
			{
				if (doThresholding) {
					if (indexStartFromOne) {
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialDescend<Type>, threshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialDescend<Type>, threshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
				}
				else
				{
					if (indexStartFromOne)
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialDescend<Type>, threshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortPartialDescend<Type>, noThreshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
				}
			}
			else if (measureMethod == MeasureMethod::cc && !numberOfResultRetain)
			{
				if (doThresholding)
				{
					if (indexStartFromOne)
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortDescend<Type>, threshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortDescend<Type>, threshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
				}
				else
				{
					if (indexStartFromOne)
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortDescend<Type>, noThreshold<Type>, indexValuePlusOne<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
					else
					{
#define exp(type1, type2) \
	dataPostProcessingFunction = sort_recordIndex<Type, type1, type2, sortDescend<Type>, noThreshold<Type>, noChangeIndexValue<type2>>
						RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
					}
				}
			}
		}
	}
	else
	{
		if (indexDataType == typeid(nullptr))
		{
			if (doThresholding) {
#define exp(type) \
	dataPostProcessingFunction = noSort_noRecordIndex<Type, type, threshold<Type>>
				RuntimeTypeInference(outputDataType, exp);
#undef exp
			}
			else
			{
#define exp(type) \
	dataPostProcessingFunction = noSort_noRecordIndex<Type, type, noThreshold<Type>>
				RuntimeTypeInference(outputDataType, exp);
#undef exp
			}
		}
		else
		{
			if (doThresholding) {
				if (indexStartFromOne)
				{
#define exp(type1, type2) \
	dataPostProcessingFunction = noSort_recordIndex<Type, type1, type2, threshold<Type>, indexValuePlusOne<type2>>
					RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
				}
				else
				{
#define exp(type1, type2) \
	dataPostProcessingFunction = noSort_recordIndex<Type, type1, type2, threshold<Type>, noChangeIndexValue<type2>>
					RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
				}
			}
			else
			{
				if (indexStartFromOne)
				{
#define exp(type1, type2) \
	dataPostProcessingFunction = noSort_recordIndex<Type, type1, type2, noThreshold<Type>, indexValuePlusOne<type2>>
					RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp					
				}
				else
				{
#define exp(type1, type2) \
	dataPostProcessingFunction = noSort_recordIndex<Type, type1, type2, noThreshold<Type>, noChangeIndexValue<type2>>
					RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
				}
			}
		}
	}

#define EXP(type) \
	blockCopyingAFunction = (BlockCopyMethod*)(copyBlock<Type, type>)
	RuntimeTypeInference(inputADataType, EXP);
#undef EXP

#define EXP(type) \
	blockCopyingBFunction = (BlockCopyMethod*)(copyBlock<Type, type>)
	RuntimeTypeInference(inputBDataType, EXP);
#undef EXP

	if (searchType == SearchType::global)
	{
		determineBlockBRangeFunction = determineBlockB_index_full;
	}
	else if (searchType == SearchType::local && searchFrom == SearchFrom::center)
	{
		determineBlockBRangeFunction = determineBlockB_index_local;
	}
	else if (searchType == SearchType::local && searchFrom == SearchFrom::topLeft)
	{
		determineBlockBRangeFunction = determineBlockB_index_local_topLeft;
	}
	else
		NOT_IMPLEMENTED_ERROR;

	if (sequenceABorderType == BorderType::normal)
	{
		iterationIndexPostProcessFunction = noIndexPostProcess;
	}
	else if (sequenceABorderType == BorderType::includeLastBlock)
	{
		iterationIndexPostProcessFunction = tryToIncludeLastBlock;
	}

	if (indexDataType == typeid(nullptr))
	{
		indexRecordFunction = noRecordIndex;
	}
	else
	{
		indexRecordFunction = recordIndexPlusOne;
	}

	if (measureMethod == MeasureMethod::mse)
	{
		executionFunction = processWorker<Type, lib_match_mse_check_border<Type>>;
	}
	else if (measureMethod == MeasureMethod::cc)
	{
		executionFunction = processWorker<Type, lib_match_cc_check_border<Type>>;
	}
	
	switch (padMethodA)
	{
	case PadMethod::zero:
#define exp(type) \
	padFunctionA = (PadFunction*)zeroPadding<type>
		RuntimeTypeInference(inputADataType, exp);
#undef exp
		break;
	case PadMethod::circular:
#define exp(type) \
	padFunctionA = (PadFunction*)circularPadding<type>
		RuntimeTypeInference(inputADataType, exp);
#undef exp
		break;
	case PadMethod::replicate:
#define exp(type) \
	padFunctionA = (PadFunction*)replicatePadding<type>
		RuntimeTypeInference(inputADataType, exp);
#undef exp
		break;
	case PadMethod::symmetric:
#define exp(type) \
	padFunctionA = (PadFunction*)symmetricPadding<type>
		RuntimeTypeInference(inputADataType, exp);
#undef exp
		break;
	default: break;
	}

	switch (padMethodB)
	{
	case PadMethod::zero:
#define exp(type) \
	padFunctionB = (PadFunction*)zeroPadding<type>
		RuntimeTypeInference(inputBDataType, exp);
#undef exp
		break;
	case PadMethod::circular:
#define exp(type) \
	padFunctionB = (PadFunction*)circularPadding<type>
		RuntimeTypeInference(inputBDataType, exp);
#undef exp
		break;
	case PadMethod::replicate:
#define exp(type) \
	padFunctionB = (PadFunction*)replicatePadding<type>
		RuntimeTypeInference(inputBDataType, exp);
#undef exp
		break;
	case PadMethod::symmetric:
#define exp(type) \
	padFunctionB = (PadFunction*)symmetricPadding<type>
		RuntimeTypeInference(inputBDataType, exp);
#undef exp
		break;
	default: break;
	}
	const int numberOfBlockBPerBlockA = numberOfBlockBPerBlockA_M * numberOfBlockBPerBlockA_N;

	if (numberOfResultRetain > numberOfBlockBPerBlockA)
		throw std::runtime_error("Check Error: Parameter 'retain' cannot larger than number of blocks of B");

	BlockMatchContext<Type> *instance = new BlockMatchContext<Type>{
		matrixA_M, matrixA_N, matrixB_M, matrixB_N,
		matrixA_padded_M, matrixA_padded_N, matrixB_padded_M, matrixB_padded_N,
		block_M, block_N,
		searchRegion_M, searchRegion_N,
		strideA_M, strideA_N, strideB_M, strideB_N,
		matrixAPadding_M_pre, matrixAPadding_M_post,matrixAPadding_N_pre, matrixAPadding_N_post,
		matrixBPadding_M_pre, matrixBPadding_M_post, matrixBPadding_N_pre, matrixBPadding_N_post,
		indexA_M_begin, indexA_M_end, indexA_N_begin, indexA_N_end,
		numberOfResultRetain,
		numberOfThreads,
		padFunctionA, padFunctionB , executionFunction,
		dataPostProcessingFunction, blockCopyingAFunction, blockCopyingBFunction,
		determineBlockBRangeFunction, iterationIndexPostProcessFunction, indexRecordFunction,
		numberOfBlockBPerBlockA_M,numberOfBlockBPerBlockA_N,numberOfBlockBPerBlockA,
		{ matrixC_M, matrixC_N, matrixC_X },
		thresholdValue, replacementValue,
		std::vector<cudaStream_guard>(), // streams
		0, 0, 0, // numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue
		std::vector<void *>(), // threadPoolTaskHandle
		memory_allocator<int, memory_type::system>(0), // common_buffer
		std::vector<typename BlockMatchContext<Type>::WorkerContext>(), // workerContext
		/*std::vector<typename BlockMatchContext<Type>::OptionalPerThreadBuffer>(), // optionalPerThreadBuffer
		std::vector<typename BlockMatchContext<Type>::OptionalBuffer>(), // optionalBuffer */
		{ memory_allocator<Type, memory_type::system>(matrixA_padded_M*matrixA_padded_N), 
			memory_allocator<Type, memory_type::system>(matrixB_padded_M * matrixB_padded_N)},
		std::vector<typename BlockMatchContext<Type>::PerThreadBuffer>() // perThreadBuffer
	};

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor[indexOfDevice];
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;

	determineGpuTaskConfiguration(numberOfGPUProcessorThread, numberOfGPUDeviceMultiProcessor, numberOfBlockBPerBlockA,
		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &sizeOfGpuTaskQueue);

	instance->numberOfSubmitThreadsPerProcessor = numberOfSubmitThreadsPerProcessor;
	instance->numberOfSubmitProcessors = numberOfSubmitProcessors;
	instance->sizeOfGpuTaskQueue = sizeOfGpuTaskQueue;

	const int perThreadMatrixCBufferSize = sizeOfGpuTaskQueue * numberOfBlockBPerBlockA;
	const int perThreadMatrixABufferSize = sizeOfGpuTaskQueue * block_M * block_N;
	const int perThreadMatrixBBufferSize = perThreadMatrixCBufferSize * block_M * block_N;

	instance->workerContext.reserve(numberOfThreads);
	instance->threadPoolTaskHandle.resize(numberOfThreads);
	instance->streams.resize(numberOfThreads);
	instance->perThreadBuffer.reserve(numberOfThreads);

	CUDA_CHECK_POINT(cudaSetDevice(indexOfDevice));

	if (sort && indexDataType != typeid(nullptr))
	{
		instance->common_buffer.resize(numberOfBlockBPerBlockA);

		for (int i = 0; i < numberOfThreads; ++i)
		{
			instance->perThreadBuffer.emplace_back(BlockMatchContext<Type>::PerThreadBuffer{
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixABufferSize), // matrixA_buffer
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixBBufferSize), // matrixB_buffer
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixCBufferSize), // matrixC_buffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixABufferSize), // matrixA_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixBBufferSize), // matrixB_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixCBufferSize), // matrixC_deviceBuffer
				memory_allocator<int, memory_type::system>(perThreadMatrixCBufferSize), // index_x_sorting_buffer
				memory_allocator<int, memory_type::system>(perThreadMatrixCBufferSize), // index_y_sorting_buffer
				memory_allocator<int, memory_type::system>(numberOfBlockBPerBlockA) // index_raw_sorting_buffer
			});
		}
	}
	else if (!sort && indexDataType != typeid(nullptr))
	{
		for (int i = 0; i < numberOfThreads; ++i)
		{
			instance->perThreadBuffer.emplace_back(BlockMatchContext<Type>::PerThreadBuffer{
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixABufferSize), // matrixA_buffer
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixBBufferSize), // matrixB_buffer
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixCBufferSize), // matrixC_buffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixABufferSize), // matrixA_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixBBufferSize), // matrixB_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixCBufferSize), // matrixC_deviceBuffer
				memory_allocator<int, memory_type::system>(perThreadMatrixCBufferSize), // index_x_sorting_buffer
				memory_allocator<int, memory_type::system>(perThreadMatrixCBufferSize), // index_y_sorting_buffer
				memory_allocator<int, memory_type::system>(0) // index_raw_sorting_buffer
			});
		}
	}
	else if (indexDataType == typeid(nullptr))
	{
		for (int i = 0; i < numberOfThreads; ++i)
		{
			instance->perThreadBuffer.emplace_back(BlockMatchContext<Type>::PerThreadBuffer{
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixABufferSize), // matrixA_buffer
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixBBufferSize), // matrixB_buffer
				memory_allocator<Type, memory_type::page_locked>(perThreadMatrixCBufferSize), // matrixC_buffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixABufferSize), // matrixA_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixBBufferSize), // matrixB_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(perThreadMatrixCBufferSize), // matrixC_deviceBuffer
				memory_allocator<int, memory_type::system>(0), // index_x_sorting_buffer
				memory_allocator<int, memory_type::system>(0), // index_y_sorting_buffer
				memory_allocator<int, memory_type::system>(0) // index_raw_sorting_buffer
			});
		}
	}
	else
		NOT_IMPLEMENTED_ERROR;

	std::vector<int> deviceLists(1, indexOfDevice);

	initializeInstanceWorkerContext(instance, sequenceABorderType, deviceLists);
	this->m_instance = instance;
}

template <typename Type>
void BlockMatch<Type>::initialize()
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(m_instance);
	instance->common_buffer.alloc();
	if (instance->common_buffer.allocated())
		generateIndexSequence(instance->common_buffer.get(), instance->numberOfBlockBPerBlockA);
	
	for (int indexOfThread = 0; indexOfThread != instance->numberOfThreads; ++indexOfThread)
	{
		typename BlockMatchContext<Type>::PerThreadBuffer &perThreadBuffer = instance->perThreadBuffer[indexOfThread];
		perThreadBuffer.matrixA_buffer.alloc();
		perThreadBuffer.matrixB_buffer.alloc();
		perThreadBuffer.matrixC_buffer.alloc();
		perThreadBuffer.matrixA_deviceBuffer.alloc();
		perThreadBuffer.matrixB_deviceBuffer.alloc();
		perThreadBuffer.matrixC_deviceBuffer.alloc();
		perThreadBuffer.index_x_sorting_buffer.alloc();
		perThreadBuffer.index_y_sorting_buffer.alloc();
		perThreadBuffer.index_raw_sorting_buffer.alloc();
	}
}

template
LIB_MATCH_EXPORT
BlockMatch<float>::BlockMatch(
	std::type_index inputADataType, std::type_index inputBDataType,
	std::type_index outputDataType,
	std::type_index indexDataType,
	SearchType searchType,
	MeasureMethod measureMethod,
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
	bool doThresholding,
	float thresholdValue, float replacementValue,
	bool indexStartFromOne,
	int indexOfDevice,
	unsigned numberOfThreads);
template
LIB_MATCH_EXPORT
BlockMatch<double>::BlockMatch(
	std::type_index inputADataType, std::type_index inputBDataType,
	std::type_index outputDataType,
	std::type_index indexDataType,
	SearchType searchType,
	MeasureMethod measureMethod,
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
	bool doThresholding,
	double thresholdValue, double replacementValue,
	bool indexStartFromOne,
	int indexOfDevice,
	unsigned numberOfThreads);

template
LIB_MATCH_EXPORT
void BlockMatch<float>::initialize();
template
LIB_MATCH_EXPORT
void BlockMatch<double>::initialize();
