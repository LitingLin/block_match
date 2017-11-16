#pragma once

#include "lib_match_internal.h"

#include "lib_match_execute.hpp"

#ifdef _MSC_VER
#pragma warning( disable : 4800 )  
#endif

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType,
	RawSortMethod_WithIndex<ComputingDataType> sortType, ThresholdMethod<ComputingDataType> thresholdFunction,
	IndexValueOffsetMethod<IndexDataType> indexValueOffset>
	inline void
	sort_recordIndex(Iterator *index_x, Iterator *index_y, Iterator *result,
		int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
		int numberOfBlockA, int *numberOfBlockBPerBlockA, int retain,
		ComputingDataType threshold, ComputingDataType replacementValue,
		const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		int c_retain = retain, c_numberOfBlockBPerBlockA = numberOfBlockBPerBlockA[i];

		if (retain > c_numberOfBlockBPerBlockA)
			c_retain = c_numberOfBlockBPerBlockA;
		memcpy(index_buffer_sort, index_buffer, c_numberOfBlockBPerBlockA * sizeof(*index_buffer_sort));

		sortType(index_buffer_sort, result_buffer, c_numberOfBlockBPerBlockA, c_retain);

		ResultDataType *result_ptr = static_cast<ResultDataType *>(result->get());
		IndexDataType *index_x_ptr = static_cast<IndexDataType *>(index_x->get());
		IndexDataType *index_y_ptr = static_cast<IndexDataType *>(index_y->get());
		for (int j = 0; j < c_retain; ++j)
		{
			ComputingDataType value = result_buffer[index_buffer_sort[j]];
			thresholdFunction(&value, threshold, replacementValue);
			*result_ptr++ = static_cast<ResultDataType>(value);
			*index_x_ptr = static_cast<IndexDataType>(index_x_buffer[index_buffer_sort[j]]);
			indexValueOffset(index_x_ptr++);
			*index_y_ptr = static_cast<IndexDataType>(index_y_buffer[index_buffer_sort[j]]);
			indexValueOffset(index_y_ptr++);
		}
		for (int j = c_retain; j < retain; ++j)
		{
			*result_ptr++ = NAN;
			*index_x_ptr++ = 0;
			*index_y_ptr++ = 0;
		}
		result->next();
		index_x->next();
		index_y->next();

		index_x_buffer += c_numberOfBlockBPerBlockA;
		index_y_buffer += c_numberOfBlockBPerBlockA;
		result_buffer += c_numberOfBlockBPerBlockA;
	}
}

template <typename ComputingDataType, typename ResultDataType,
	RawSortMethod<ComputingDataType> sortType, ThresholdMethod<ComputingDataType> thresholdFunction>
	inline void
	sort_noRecordIndex(Iterator *index_x, Iterator *index_y, Iterator *result,
		int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
		int numberOfBlockA, int *numberOfBlockBPerBlockA, int retain,
		ComputingDataType threshold, ComputingDataType replacementValue,
		const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		int c_retain = retain, c_numberOfBlockBPerBlockA = numberOfBlockBPerBlockA[i];
		if (retain > c_numberOfBlockBPerBlockA)
			c_retain = c_numberOfBlockBPerBlockA;
		sortType(result_buffer, c_numberOfBlockBPerBlockA, c_retain);

		ResultDataType *result_ptr = static_cast<ResultDataType *>(result->get());
		for (int j = 0; j < c_retain; ++j)
		{
			ComputingDataType value = *result_buffer++;
			thresholdFunction(&value, threshold, replacementValue);
			*result_ptr++ = static_cast<ResultDataType>(value);
		}
		for (int j = c_retain; j < retain; ++j)
		{
			*result_ptr++ = NAN;
		}
		result_buffer += c_numberOfBlockBPerBlockA - c_retain;
		result->next();
	}
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType,
	ThresholdMethod<ComputingDataType> thresholdFunction,
	IndexValueOffsetMethod<IndexDataType> indexValueOffset>
	inline void
	noSort_recordIndex(Iterator *index_x, Iterator *index_y, Iterator *result,
		int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
		int numberOfBlockA, int *numberOfBlockBPerBlockA, int retain,
		ComputingDataType threshold, ComputingDataType replacementValue,
		const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		int c_retain = retain, c_numberOfBlockBPerBlockA = numberOfBlockBPerBlockA[i];
		if (retain > c_numberOfBlockBPerBlockA)
			c_retain = c_numberOfBlockBPerBlockA;
		ResultDataType *result_ptr = static_cast<ResultDataType *>(result->get());
		IndexDataType *index_x_ptr = static_cast<IndexDataType *>(index_x->get());
		IndexDataType *index_y_ptr = static_cast<IndexDataType *>(index_y->get());
		for (int j = 0; j < c_retain; ++j)
		{
			ComputingDataType value = *result_buffer++;
			thresholdFunction(&value, threshold, replacementValue);
			*result_ptr++ = static_cast<ResultDataType>(value);
			*index_x_ptr = static_cast<IndexDataType>(*index_x_buffer++);
			indexValueOffset(index_x_ptr++);
			*index_y_ptr = static_cast<IndexDataType>(*index_y_buffer++);
			indexValueOffset(index_y_ptr++);
		}
		for (int j = c_retain; j < retain; ++j)
		{
			*result_ptr++ = NAN;
			*index_x_ptr++ = 0;
			*index_y_ptr++ = 0;
		}
		result_buffer += c_numberOfBlockBPerBlockA - c_retain;
		index_x_buffer += c_numberOfBlockBPerBlockA - c_retain;
		index_y_buffer += c_numberOfBlockBPerBlockA - c_retain;
		result->next();
		index_x->next();
		index_y->next();
	}
}


template <typename ComputingDataType, typename ResultDataType,
	ThresholdMethod<ComputingDataType> thresholdFunction>
	inline void
	noSort_noRecordIndex(Iterator *index_x, Iterator *index_y, Iterator *result,
		int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
		int numberOfBlockA, int *numberOfBlockBPerBlockA, int retain,
		ComputingDataType threshold, ComputingDataType replacementValue,
		const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		int c_retain = retain, c_numberOfBlockBPerBlockA = numberOfBlockBPerBlockA[i];
		if (retain > c_numberOfBlockBPerBlockA)
			c_retain = c_numberOfBlockBPerBlockA;
		ResultDataType *result_ptr = static_cast<ResultDataType*>(result->get());
		for (int j = 0; j < c_retain; ++j)
		{
			ComputingDataType value = *result_buffer++;
			thresholdFunction(&value, threshold, replacementValue);
			*result_ptr++ = static_cast<ResultDataType>(value);
		}
		for (int j = c_retain; j < retain; ++j)
		{
			*result_ptr++ = NAN;
		}
		result_buffer += c_numberOfBlockBPerBlockA - c_retain;
		result->next();
	}
}

#ifdef _MSC_VER
#pragma warning( default : 4800 )  
#endif

typedef void RecordIndex(int*, int*, int, int);

// TODO: Fix busy waiting gpu tasks
template <typename Type,
	ProcessFunction<Type> processFunction>
	unsigned processWorker(ExecutionContext<Type> *executionContext)
{
	CUDA_CHECK_POINT(cudaSetDevice(executionContext->indexOfDevice));

	void *matrixA = executionContext->matrixA, *matrixB = executionContext->matrixB;
	int numberOfChannels = executionContext->numberOfChannels;
	Iterator *matrixC = executionContext->matrixC.get();
	Type *matrixA_buffer = executionContext->matrixA_buffer, *matrixB_buffer = executionContext->matrixB_buffer,
		*matrixC_buffer = executionContext->matrixC_buffer,
		*matrixA_deviceBuffer = executionContext->matrixA_deviceBuffer, *matrixB_deviceBuffer = executionContext->matrixB_deviceBuffer,
		*matrixC_deviceBuffer = executionContext->matrixC_deviceBuffer;
	int matrixA_M = executionContext->matrixA_M, matrixA_N = executionContext->matrixA_N,
		matrixB_M = executionContext->matrixB_M, matrixB_N = executionContext->matrixB_N;
	Iterator *index_x = executionContext->index_x.get(), *index_y = executionContext->index_y.get();
	int *index_x_buffer = executionContext->index_x_buffer, *index_y_buffer = executionContext->index_y_buffer,
		*rawIndexTemplate = executionContext->rawIndexTemplate, *rawIndexBuffer = executionContext->rawIndexBuffer,
		*sizeBuffer = executionContext->sizeBuffer,
		*offsetA_buffer = executionContext->offsetA_buffer,
		*offsetA_deviceBuffer = executionContext->offsetA_deviceBuffer,
		block_M = executionContext->block_M, block_N = executionContext->block_N,
		strideA_M = executionContext->strideA_M, strideA_N = executionContext->strideA_N,
		strideB_M = executionContext->strideB_M, strideB_N = executionContext->strideB_N,
		searchRegion_M_pre = executionContext->searchRegion_M_pre, searchRegion_M_post = executionContext->searchRegion_M_post,
		searchRegion_N_pre = executionContext->searchRegion_N_pre, searchRegion_N_post = executionContext->searchRegion_N_post,
		numberOfBlockBPerBlockA = executionContext->numberOfBlockBPerBlockA,
		numberOfIndexRetain = executionContext->numberOfIndexRetain,
		indexA_M_begin = executionContext->indexA_M_begin, indexA_N_begin = executionContext->indexA_N_begin,
		indexA_M_end = executionContext->indexA_M_end, indexA_N_end = executionContext->indexA_N_end,
		startIndexOfMatrixA_M = executionContext->startIndexOfMatrixA_M, startIndexOfMatrixA_N = executionContext->startIndexOfMatrixA_N,
		numberOfIteration = executionContext->numberOfIteration;

	cudaStream_t stream = executionContext->stream; // TODO: Double buffering
	int maxNumberOfThreadsPerProcessor = executionContext->maxNumberOfThreadsPerProcessor,
		numberOfSubmitThreadsPerProcessor = executionContext->numberOfSubmitThreadsPerProcessor,
		numberOfSubmitProcessors = executionContext->numberOfSubmitProcessors,
		lengthOfGpuTaskQueue = executionContext->lengthOfGpuTaskQueue;

	Type thresholdValue = executionContext->thresholdValue, replacementValue = executionContext->replacementValue;

	int blockSize = executionContext->block_M * executionContext->block_N * numberOfChannels;
	Type *c_bufferA = executionContext->matrixA_buffer;
	Type *c_bufferB = executionContext->matrixB_buffer;
	int *c_index_x_buffer = executionContext->index_x_buffer, *c_index_y_buffer = executionContext->index_y_buffer;

	int numberOfBlockA = 0, numberOfBlockB = 0;
	if (!numberOfIndexRetain)
		numberOfIndexRetain = numberOfBlockBPerBlockA;

	int indexOfIteration = 0;
	int indexA_M = startIndexOfMatrixA_M, indexA_N = startIndexOfMatrixA_N;

	int blockStrideA_M = executionContext->blockStrideA_M;
	int blockStrideA_N = executionContext->blockStrideA_N;
	int blockStrideB_M = executionContext->blockStrideB_M;
	int blockStrideB_N = executionContext->blockStrideB_N;
	int realBlockA_M = block_M * blockStrideA_M;
	int realBlockA_N = block_N * blockStrideA_N;
	int realBlockB_M = block_M * blockStrideB_M;
	int realBlockB_N = block_N * blockStrideB_N;

	DataPostProcessingMethod<Type> *dataPostProcessing = executionContext->dataPostProcessingFunction;
	BlockCopyMethod *blockCopyA = executionContext->blockCopyingAFunction;
	BlockCopyMethod *blockCopyB = executionContext->blockCopyingBFunction;
	DetermineBlockBRangeMethod *determineBlockBRange = executionContext->determineBlockBRangeFunction;
	IterationIndexPostProcessMethod *iterationIndexPostProcess = executionContext->iterationIndexPostProcessFunction;
	IndexRecordMethod *indexRecord = executionContext->indexRecordFunction;

	goto JumpIn;

	for (/*indexA_M = indexA_M_begin*/; indexA_M < indexA_M_end || outOfIndexError(); indexA_M += strideA_M)
	{
		for (indexA_N = indexA_N_begin; indexA_N < indexA_N_end; indexA_N += strideA_N)
		{
		JumpIn:
			blockCopyA(numberOfChannels, c_bufferA, matrixA,
				matrixA_M, matrixA_N,
				indexA_M, indexA_N, block_M, block_N, blockStrideA_M, blockStrideA_N);

			int sequenceBCount = 0;
			int indexB_M_begin, indexB_M_end;
			determineBlockBRange(&indexB_M_begin, &indexB_M_end,
				matrixB_M, realBlockB_M,
				searchRegion_M_pre, searchRegion_M_post, indexA_M);
			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
			{
				int indexB_N_begin, indexB_N_end;
				determineBlockBRange(&indexB_N_begin, &indexB_N_end,
					matrixB_N, realBlockB_N,
					searchRegion_N_pre, searchRegion_N_post, indexA_N);
				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
				{
					blockCopyB(numberOfChannels, c_bufferB, matrixB,
						matrixB_M, matrixB_N,
						indexB_M, indexB_N, realBlockB_M, block_N, blockStrideB_M, blockStrideB_N);
					indexRecord(c_index_x_buffer++, c_index_y_buffer++, indexB_M, indexB_N);
					c_bufferB += blockSize;
					offsetA_buffer[numberOfBlockB++] = numberOfBlockA;
					sequenceBCount++;
				}
			}

			sizeBuffer[numberOfBlockA] = sequenceBCount;

			++numberOfBlockA;

			c_bufferA += blockSize;

			if (numberOfBlockA == lengthOfGpuTaskQueue)
			{/*
				if (checkIsInterruptPending())
					return 2;*/

				submitGpuTask_offset<Type, processFunction>(matrixA_buffer, matrixB_buffer, matrixC_buffer,
					matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
					blockSize, numberOfBlockA, numberOfBlockB,
					offsetA_buffer, offsetA_deviceBuffer,
					numberOfSubmitProcessors, numberOfSubmitThreadsPerProcessor, stream);

				CUDA_CHECK_POINT(cudaStreamSynchronize(stream));

				//std::swap(streamA, streamB);

				c_index_x_buffer = index_x_buffer;
				c_index_y_buffer = index_y_buffer;

				dataPostProcessing(index_x, index_y, matrixC, index_x_buffer, index_y_buffer,
					matrixC_buffer,
					numberOfBlockA, sizeBuffer, numberOfIndexRetain,
					thresholdValue, replacementValue,
					rawIndexTemplate, rawIndexBuffer);

				//c_result += numTasks;
				c_bufferA = matrixA_buffer;
				c_bufferB = matrixB_buffer;

				numberOfBlockA = 0;
				numberOfBlockB = 0;
			}
			iterationIndexPostProcess(&indexA_N, strideA_N, indexA_N_end);

			++indexOfIteration;

			if (indexOfIteration == numberOfIteration)
				goto JumpOut;
		}

		iterationIndexPostProcess(&indexA_M, strideA_M, indexA_M_end);
	}
JumpOut:
	if (numberOfBlockA)
	{/*
		if (checkIsInterruptPending())
			return 2;*/

		submitGpuTask_offset<Type, processFunction>(matrixA_buffer, matrixB_buffer, matrixC_buffer,
			matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
			blockSize, numberOfBlockA, numberOfBlockB,
			offsetA_buffer, offsetA_deviceBuffer,
			(numberOfBlockB + maxNumberOfThreadsPerProcessor - 1) / maxNumberOfThreadsPerProcessor,
			maxNumberOfThreadsPerProcessor, stream);

		CUDA_CHECK_POINT(cudaStreamSynchronize(stream));

		dataPostProcessing(index_x, index_y, matrixC, index_x_buffer, index_y_buffer, matrixC_buffer,
			numberOfBlockA, sizeBuffer, numberOfIndexRetain,
			thresholdValue, replacementValue,
			rawIndexTemplate, rawIndexBuffer);
	}

	return 0;
}
//
//template <typename Type,
//	ProcessFunctionCPU<Type> processFunction>
//	unsigned processWorker_cpu(ExecutionContext<Type> *executionContext)
//{
//	void *matrixA = executionContext->matrixA, *matrixB = executionContext->matrixB;
//	Iterator *matrixC = executionContext->matrixC.get();
//	Type *matrixA_buffer = executionContext->matrixA_buffer, *matrixB_buffer = executionContext->matrixB_buffer,
//		*matrixC_buffer = executionContext->matrixC_buffer;
//
//	int matrixA_M = executionContext->matrixA_M, matrixA_N = executionContext->matrixA_N,
//		matrixB_M = executionContext->matrixB_M, matrixB_N = executionContext->matrixB_N;
//	Iterator *index_x = executionContext->index_x.get(), *index_y = executionContext->index_y.get();
//	int	*index_x_buffer = executionContext->index_x_buffer, *index_y_buffer = executionContext->index_y_buffer,
//		*rawIndexTemplate = executionContext->rawIndexTemplate, *rawIndexBuffer = executionContext->rawIndexBuffer,
//		block_M = executionContext->block_M, block_N = executionContext->block_N,
//		strideA_M = executionContext->strideA_M, strideA_N = executionContext->strideA_N,
//		strideB_M = executionContext->strideB_M, strideB_N = executionContext->strideB_N,
//		neighbour_M = executionContext->neighbour_M, neighbour_N = executionContext->neighbour_N,
//		numberOfBlockBPerBlockA = executionContext->numberOfBlockBPerBlockA,
//		numberOfIndexRetain = executionContext->numberOfIndexRetain,
//		indexA_M_begin = executionContext->indexA_M_begin, indexA_N_begin = executionContext->indexA_N_begin,
//		indexA_M_end = executionContext->indexA_M_end, indexA_N_end = executionContext->indexA_N_end,
//		startIndexOfMatrixA_M = executionContext->startIndexOfMatrixA_M, startIndexOfMatrixA_N = executionContext->startIndexOfMatrixA_N;
//
//	int blockSize = executionContext->block_M * executionContext->block_N;
//	int *c_index_x_buffer = executionContext->index_x_buffer, *c_index_y_buffer = executionContext->index_y_buffer;
//
//	Type *c_matrixC_buffer = matrixC_buffer;
//
//	if (!numberOfIndexRetain)
//		numberOfIndexRetain = numberOfBlockBPerBlockA;
//
//	int indexA_M = startIndexOfMatrixA_M, indexA_N = startIndexOfMatrixA_N;
//
//	DataPostProcessingMethod<Type> *dataPostProcessing = executionContext->dataPostProcessingFunction;
//	BlockCopyMethod *blockCopyA = executionContext->blockCopyingAFunction;
//	BlockCopyMethod *blockCopyB = executionContext->blockCopyingBFunction;
//	DetermineBlockBRangeMethod *determineBlockBRange = executionContext->determineBlockBRangeFunction;
//	IterationIndexPostProcessMethod *iterationIndexPostProcess = executionContext->iterationIndexPostProcessFunction;
//	IndexRecordMethod *indexRecord = executionContext->indexRecordFunction;
//	
//	goto JumpIn;
//
//	for (/*indexA_M = indexA_M_begin*/; indexA_M < indexA_M_end || outOfIndexError(); indexA_M += strideA_M)
//	{
//		for (indexA_N = indexA_N_begin; indexA_N < indexA_N_end; indexA_N += strideA_N)
//		{
//		JumpIn:
//			blockCopyA(matrixA_buffer, matrixA,
//				matrixA_M, matrixA_N,
//				indexA_M, indexA_N, block_M, block_N);
//
//#ifndef NDEBUG
//			int sequenceBCount = 0;
//#endif
//			int indexB_M_begin, indexB_M_end;
//			determineBlockBRange(&indexB_M_begin, &indexB_M_end,
//				matrixB_M, block_M,
//				neighbour_M, indexA_M);
//			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
//			{
//				int indexB_N_begin, indexB_N_end;
//				determineBlockBRange(&indexB_N_begin, &indexB_N_end,
//					matrixB_N, block_N, neighbour_N, indexA_N);
//				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
//				{
//					blockCopyB(matrixB_buffer, matrixB,
//						matrixB_M, matrixB_N,
//						indexB_M, indexB_N, block_M, block_N);
//					indexRecord(c_index_x_buffer++, c_index_y_buffer++, indexB_M, indexB_N);
//
//					processFunction(matrixA_buffer, matrixB_buffer, blockSize, c_matrixC_buffer++);
//
//#ifndef NDEBUG
//					sequenceBCount++;
//#endif
//				}
//			}
//
//#ifndef NDEBUG
//			if (sequenceBCount != numberOfBlockBPerBlockA)
//				logger.critical("Internal logical error: sequenceBCount != numberOfBlockBPerBlockA");
//#endif
//
//			c_index_x_buffer = index_x_buffer;
//			c_index_y_buffer = index_y_buffer;
//
//			dataPostProcessing(index_x, index_y, matrixC, index_x_buffer, index_y_buffer,
//				matrixC_buffer,
//				1, numberOfBlockBPerBlockA, numberOfIndexRetain,
//				rawIndexTemplate, rawIndexBuffer);
//			
//			iterationIndexPostProcess(&indexA_N, strideA_N, indexA_N_end);
//		}
//		iterationIndexPostProcess(&indexA_M, strideA_M, indexA_M_end);
//	}
//
//	return 0;
//}