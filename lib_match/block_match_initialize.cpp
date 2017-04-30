#include "lib_match_internal.h"

#include <cuda_runtime.h>

#include "block_match_execute.hpp"

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations)
{
	double numberOfBlockAPerProcessor = static_cast<double>(maxNumberOfGpuThreads) / static_cast<double>(numberOfBlockBPerBlockA);
	if (numberOfBlockAPerProcessor > 1.0)
	{
		int fixedNumberOfBlockAPerProcessor = static_cast<int>(numberOfBlockAPerProcessor);
		*numberOfSubmitThreadsPerProcessor = static_cast<int>(numberOfBlockAPerProcessor) * numberOfBlockBPerBlockA;
		*numberOfSubmitProcessors = numberOfGpuProcessors;
		*numberOfIterations = fixedNumberOfBlockAPerProcessor * numberOfGpuProcessors;
	}
	else
	{
		double numberOfProcessorPerBlockA = 1.0 / numberOfBlockAPerProcessor;
		if (numberOfProcessorPerBlockA < numberOfGpuProcessors)
		{
			int _numberOfIterations = static_cast<int>(static_cast<double>(numberOfGpuProcessors) / numberOfProcessorPerBlockA);
			int _numberOfSubmitProcessors = static_cast<int>(std::ceil(_numberOfIterations * numberOfProcessorPerBlockA));
			*numberOfSubmitThreadsPerProcessor = maxNumberOfGpuThreads;
			*numberOfSubmitProcessors = _numberOfSubmitProcessors;
			*numberOfIterations = _numberOfIterations;
		}
		else
		{
			*numberOfSubmitThreadsPerProcessor = maxNumberOfGpuThreads;
			*numberOfSubmitProcessors = static_cast<int>(std::ceil(numberOfProcessorPerBlockA));
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
//
//template <typename Type>
//void resourceInitialize_GPU_sorting(BlockMatchContext<Type> *instance, const int numberOfBlockBPerBlockA, )
//{
//	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
//	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;
//
//	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;
//
//	determineGpuTaskConfiguration(numberOfGPUProcessorThread, numberOfGPUDeviceMultiProcessor, numberOfBlockBPerBlockA,
//		&numberOfSubmitThreadsPerProcessor, &numberOfSubmitProcessors, &sizeOfGpuTaskQueue);
//
//
//	const int perThreadMatrixCBufferSize = sizeOfGpuTaskQueue * numberOfBlockBPerBlockA;
//	const int perThreadMatrixABufferSize = perThreadMatrixCBufferSize * block_M * block_N;
//
//
//	instance->workerContext.reserve(numberOfThreads);
//	instance->threadPoolTaskHandle.reserve(numberOfThreads);
//	instance->streams.resize(numberOfThreads);
//	instance->perThreadBuffer.reserve(numberOfThreads);
//
//	for (int i = 0; i < numberOfThreads; ++i)
//	{
//		instance->perThreadBuffer.emplace_back(BlockMatchContext<Type>::PerThreadBuffer{
//			page_locked_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixA_buffer
//			page_locked_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixB_buffer
//			page_locked_memory_allocator<Type>(perThreadMatrixCBufferSize), // matrixC_buffer
//			gpu_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixA_deviceBuffer
//			gpu_memory_allocator<Type>(perThreadMatrixABufferSize), // matrixB_deviceBuffer
//			gpu_memory_allocator<Type>(perThreadMatrixCBufferSize), // matrixC_deviceBuffer
//			system_memory_allocator<int>(perThreadMatrixCBufferSize), // index_x_sorting_buffer
//			system_memory_allocator<int>(perThreadMatrixCBufferSize), // index_y_sorting_buffer
//			system_memory_allocator<int>(numberOfBlockBPerBlockA) // index_raw_sorting_buffer
//		});
//	}
//
//	initializeInstanceWorkerContext(instance, sequenceABorderType);
//
//	instance->common_buffer.alloc();
//	generateIndexSequence(instance->common_buffer.get(), numberOfBlockBPerBlockA);
//}

template <typename Type>
BlockMatch<Type>::BlockMatch(std::type_index inputDataType, std::type_index outputDataType,
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
	int numberOfIndexRetain)
	: m_instance(nullptr)
{
	CHECK_POINT(globalContext.hasGPU);

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

	PadFunction *padFunctionA = nullptr;
	PadFunction *padFunctionB = nullptr;
	ExecutionFunction<Type> *executionFunction = nullptr;

	/*
	if (sort && searchType == SearchType::local && measureMethod == MeasureMethod::mse
	&& numberOfIndexRetain && sequenceABorderType == BorderType::normal && searchFrom == SearchFrom::topLeft)
	{

	}*/

	if (sort) {
		if (searchType == SearchType::local)
		{
			if (measureMethod == MeasureMethod::mse)
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
			else if (measureMethod == MeasureMethod::cc)
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
					else
						if (searchFrom == SearchFrom::topLeft)
							executionFunction = processWorker<Type, determineBlockB_index_local_topLeft,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
						else
							executionFunction = processWorker<Type, determineBlockB_index_local,
							lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
		}
		else if (searchType == SearchType::global)
		{
			if (measureMethod == MeasureMethod::mse) {
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialAscend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_mse_check_border, sortWithIndex<Type, SortMethodProxy::sortAscend<Type>>, tryToIncludeLastBlock>;
			}
			else if (measureMethod == MeasureMethod::cc) {
				if (numberOfIndexRetain)
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortPartialDescend<Type>>, tryToIncludeLastBlock>;
				else
					if (sequenceABorderType == BorderType::normal)
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, dummyCheckIsLastBlock>;
					else
						executionFunction = processWorker<Type, determineBlockB_index_full,
						lib_match_cc_check_border, sortWithIndex<Type, SortMethodProxy::sortDescend<Type>>, tryToIncludeLastBlock>;
			}
			else
				NOT_IMPLEMENTED_ERROR;
		}
	}
	else
	{
		if (searchType == SearchType::local)
		{
			if (measureMethod == MeasureMethod::mse)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_local,
					lib_match_mse_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_local,
					lib_match_mse_check_border, dummySort, tryToIncludeLastBlock>;
			else if (measureMethod == MeasureMethod::cc)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_local,
					lib_match_cc_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_local,
					lib_match_cc_check_border, dummySort, tryToIncludeLastBlock>;
			else
				NOT_IMPLEMENTED_ERROR;
		}
		else if (searchType == SearchType::global)
		{
			if (measureMethod == MeasureMethod::mse)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_full,
					lib_match_mse_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_full,
					lib_match_mse_check_border, dummySort, tryToIncludeLastBlock>;
			else if (measureMethod == MeasureMethod::cc)
				if (sequenceABorderType == BorderType::normal)
					executionFunction = processWorker<Type, determineBlockB_index_full,
					lib_match_cc_check_border, dummySort, dummyCheckIsLastBlock>;
				else
					executionFunction = processWorker<Type, determineBlockB_index_full,
					lib_match_cc_check_border, dummySort, tryToIncludeLastBlock>;
			else
				NOT_IMPLEMENTED_ERROR;
		}
	}

#define RuntimeTypeInference(type, exp) \
	if (type == typeid(uint8_t)) \
		exp<uint8_t>; \
	else if (type == typeid(int8_t)) \
		exp<int8_t>; \
	else if (type == typeid(uint16_t)) \
		exp<uint16_t>; \
	else if (type == typeid(int16_t)) \
		exp<int16_t>; \
	else if (type == typeid(uint32_t)) \
		exp<uint32_t>; \
	else if (type == typeid(int32_t)) \
		exp<int32_t>; \
	else if (type == typeid(uint64_t)) \
		exp<uint64_t>; \
	else if (type == typeid(int64_t)) \
		exp<int64_t>; \
	else if (type == typeid(float)) \
		exp<float>; \
	else if (type == typeid(double)) \
		exp<double>

	switch (padMethodA)
	{
	case PadMethod::zero:
		RuntimeTypeInference(inputDataType, padFunctionA = (PadFunction*)zeroPadding);
		break;
	case PadMethod::circular:
		RuntimeTypeInference(inputDataType, padFunctionA = (PadFunction*)circularPadding);
		break;
	case PadMethod::replicate:
		RuntimeTypeInference(inputDataType, padFunctionA = (PadFunction*)replicatePadding);
		break;
	case PadMethod::symmetric:
		RuntimeTypeInference(inputDataType, padFunctionA = (PadFunction*)symmetricPadding);
		break;
	default: break;
	}

	switch (padMethodB)
	{
	case PadMethod::zero:
		RuntimeTypeInference(inputDataType, padFunctionB = (PadFunction*)zeroPadding);
		break;
	case PadMethod::circular:
		RuntimeTypeInference(inputDataType, padFunctionB = (PadFunction*)circularPadding);
		break;
	case PadMethod::replicate:
		RuntimeTypeInference(inputDataType, padFunctionB = (PadFunction*)replicatePadding);
		break;
	case PadMethod::symmetric:
		RuntimeTypeInference(inputDataType, padFunctionB = (PadFunction*)symmetricPadding);
		break;
	default: break;
	}

	const int numberOfBlockBPerBlockA = numberOfBlockBPerBlockA_M * numberOfBlockBPerBlockA_N;

	if (numberOfIndexRetain > numberOfBlockBPerBlockA)
		throw std::runtime_error("Check Error: Parameter 'retain' cannot larger than number of blocks of B");

	BlockMatchContext<Type> *instance = new BlockMatchContext<Type>{
		inputDataType, outputDataType,
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
		{ matrixC_M, matrixC_N, matrixC_X },
		std::vector<cudaStream_guard>(), // streams
		0, 0, 0, // numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue
		std::vector<void *>(), // threadPoolTaskHandle
		memory_allocator<int, memory_type::system>(0), // common_buffer
		std::vector<typename BlockMatchContext<Type>::WorkerContext>(), // workerContext
		/*std::vector<typename BlockMatchContext<Type>::OptionalPerThreadBuffer>(), // optionalPerThreadBuffer
		std::vector<typename BlockMatchContext<Type>::OptionalBuffer>(), // optionalBuffer */
		std::vector<typename BlockMatchContext<Type>::PerThreadBuffer>() // perThreadBuffer
	};

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
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

	if (sort)
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
	else
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
				memory_allocator<int, memory_type::system>(numberOfBlockBPerBlockA) // index_raw_sorting_buffer
			});
		}
	}

	initializeInstanceWorkerContext(instance, sequenceABorderType);
	this->m_instance = instance;
}

template <typename Type>
void BlockMatch<Type>::initialize()
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(m_instance);
	instance->common_buffer.alloc();
	if (instance->common_buffer.get())
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
	std::type_index, std::type_index,
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
	int numberOfIndexRetain);
template
LIB_MATCH_EXPORT
BlockMatch<double>::BlockMatch(
	std::type_index, std::type_index,
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
	int numberOfIndexRetain);

template
LIB_MATCH_EXPORT
void BlockMatch<float>::initialize();
template
LIB_MATCH_EXPORT
void BlockMatch<double>::initialize();
