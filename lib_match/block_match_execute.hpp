#pragma once

#include "lib_match_internal.h"

template <typename Type>
using ProcessFunction = cudaError_t(*)(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
using ProcessFunctionCPU = void(*)(Type *blocks_A, Type *blocks_B, int blockSize, Type *result);
//typedef void(CopyBlockMethod)(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);

template <typename Type, ProcessFunction<Type> processFunction>
void submitGpuTask(Type *bufferA, Type *bufferB, Type *resultBuffer, Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
	int blockSize,
	int numberOfBlockA, int numberOfBlockBPerBlockA,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	int numberOfBlockB = numberOfBlockA * numberOfBlockBPerBlockA;

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferA, bufferA, numberOfBlockA * blockSize * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferB, bufferB, numberOfBlockB * blockSize * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(processFunction(deviceBufferA, deviceBufferB, numberOfBlockA, numberOfBlockBPerBlockA, blockSize, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfBlockB * sizeof(Type), cudaMemcpyDeviceToHost, stream));
}

template <typename Type>
using RawSortMethod_WithIndex = void(*)(int *index, Type *value, int size, int retain);
template <typename Type>
using RawSortMethod = void(*)(Type *value, int size, int retain);

template <typename Type>
void sortAscend(int *index, Type *value, int size, int retain)
{
	lib_match_sort(index, value, size);
}

template <typename Type>
void sortPartialAscend(int *index, Type *value, int size, int retain)
{
	lib_match_sort_partial(index, value, size, retain);
}

template <typename Type>
void sortDescend(int *index, Type *value, int size, int retain)
{
	lib_match_sort_descend(index, value, size);
}

template <typename Type>
void sortPartialDescend(int *index, Type *value, int size, int retain)
{
	lib_match_sort_partial_descend(index, value, size, retain);
}

template <typename Type>
void sortAscend(Type *value, int size, int retain)
{
	lib_match_sort(value, size);
}

template <typename Type>
void sortPartialAscend(Type *value, int size, int retain)
{
	lib_match_sort_partial(value, size, retain);
}

template <typename Type>
void sortDescend(Type *value, int size, int retain)
{
	lib_match_sort_descend(value, size);
}

template <typename Type>
void sortPartialDescend(Type *value, int size, int retain)
{
	lib_match_sort_partial_descend(value, size, retain);
}
/*
template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
using DataPostProcessingMethod =
void(IndexDataType *index_x, IndexDataType *index_y, ResultDataType *result,
	int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort);
	*/
template <typename ComputingDataType, typename ResultDataType, typename IndexDataType, RawSortMethod_WithIndex<ComputingDataType> sortType>
inline void
sort_recordIndex(IndexDataType **index_x, IndexDataType **index_y, ResultDataType **result,
	int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{
	IndexDataType *c_index_x = *index_x;
	IndexDataType *c_index_y = *index_y;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		memcpy(index_buffer_sort, index_buffer, numberOfBlockBPerBlockA * sizeof(*index_buffer_sort));

		sortType(index_buffer_sort, result_buffer, numberOfBlockBPerBlockA, retain);

		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = result_buffer[index_buffer_sort[j]];

			*c_index_x++ = index_x_buffer[index_buffer_sort[j]];
			*c_index_y++ = index_y_buffer[index_buffer_sort[j]];
		}

		index_x_buffer += numberOfBlockBPerBlockA;
		index_y_buffer += numberOfBlockBPerBlockA;
		result_buffer += numberOfBlockBPerBlockA;
	}

	*index_x = c_index_x;
	*index_y = c_index_y;
	*result = *c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType, RawSortMethod<ComputingDataType> sortType>
inline void
sort_noRecordIndex(IndexDataType **index_x, IndexDataType **index_y, ResultDataType **result,
	int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		sortType(result_buffer, numberOfBlockBPerBlockA, retain);

		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = *result_buffer++;
		}

		result_buffer += numberOfBlockBPerBlockA;
	}
	*result = *c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
inline void
noSort_recordIndex(IndexDataType **index_x, IndexDataType **index_y, ResultDataType **result,
	int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{
	IndexDataType *c_index_x = *index_x;
	IndexDataType *c_index_y = *index_y;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = *result_buffer++;

			*c_index_x++ = *index_x_buffer++;
			*c_index_y++ = *index_y_buffer++;
		}
	}

	*index_x = c_index_x;
	*index_y = c_index_y;
	*result = *c_result;
}


template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
inline void
noSort_noRecordIndex(IndexDataType **index_x, IndexDataType **index_y, ResultDataType **result,
	int *index_x_buffer, int *index_y_buffer, ComputingDataType *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*result++ = *result_buffer++;
		}
	}
}

typedef void RecordIndex(int*, int*, int, int);

inline
void recordIndex(int *index_x_buffer, int *index_y_buffer, int index_x, int index_y)
{
	*index_x_buffer = index_x;
	*index_y_buffer = index_y;
}

inline
void dummyRecordIndex(int *index_x_buffer, int *index_y_buffer, int index_x, int index_y)
{
}

inline
void determineBlockB_index_local(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A)
{
	*indexB_begin = index_A - neighbour / 2;
	*indexB_end = index_A - neighbour / 2 + neighbour;
}

inline
void determineBlockB_index_local_topLeft(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A)
{
	*indexB_begin = index_A;
	*indexB_end = index_A + neighbour;
}

inline
void determineBlockB_index_full(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A)
{
	*indexB_begin = 0;
	*indexB_end = determineEndOfIndex(matB, block);
}

inline
bool indexA_M_outOfIndexError()
{
#ifndef NDEBUG
	logger.critical("Internal logical error: indexA_M out of index");
#endif
	return false;
}

inline void tryToIncludeLastBlock(int *indexA, int strideA, int indexA_end)
{
	if (*indexA + 1 == indexA_end)
		return;
	if (*indexA + strideA >= indexA_end)
		*indexA = indexA_end - strideA - 1; // + strideA immediately later
}

inline void noIndexPostProcess(int *indexA, int strideA, int indexA_end)
{
}

// TODO: Fix busy waiting gpu tasks
template <typename Type,
	ProcessFunction<Type> processFunction>
	unsigned processWorker(ExecutionContext<Type> *executionContext)
{
	void *matrixA = executionContext->matrixA, *matrixB = executionContext->matrixB,
		*matrixC = executionContext->matrixC;
	Type *matrixA_buffer = executionContext->matrixA_buffer, *matrixB_buffer = executionContext->matrixB_buffer,
		*matrixC_buffer = executionContext->matrixC_buffer,
		*matrixA_deviceBuffer = executionContext->matrixA_deviceBuffer, *matrixB_deviceBuffer = executionContext->matrixB_deviceBuffer,
		*matrixC_deviceBuffer = executionContext->matrixC_deviceBuffer;
	int matrixA_M = executionContext->matrixA_M, matrixA_N = executionContext->matrixA_N,
		matrixB_M = executionContext->matrixB_M, matrixB_N = executionContext->matrixB_N;
	int *index_x = executionContext->index_x, *index_y = executionContext->index_y,
		*index_x_buffer = executionContext->index_x_buffer, *index_y_buffer = executionContext->index_y_buffer,
		*rawIndexTemplate = executionContext->rawIndexTemplate, *rawIndexBuffer = executionContext->rawIndexBuffer,
		block_M = executionContext->block_M, block_N = executionContext->block_N,
		strideA_M = executionContext->strideA_M, strideA_N = executionContext->strideA_N,
		strideB_M = executionContext->strideB_M, strideB_N = executionContext->strideB_N,
		neighbour_M = executionContext->neighbour_M, neighbour_N = executionContext->neighbour_N,
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

	int blockSize = executionContext->block_M * executionContext->block_N;
	Type *c_bufferA = executionContext->matrixA_buffer;
	Type *c_bufferB = executionContext->matrixB_buffer;
	void *c_result = executionContext->matrixC;
	void *c_index_x = executionContext->index_x, *c_index_y = executionContext->index_y;
	int *c_index_x_buffer = executionContext->index_x_buffer, *c_index_y_buffer = executionContext->index_y_buffer;

	int numberOfBlockA = 0;
	if (!numberOfIndexRetain)
		numberOfIndexRetain = numberOfBlockBPerBlockA;

	int indexOfIteration = 0;
	int indexA_M = startIndexOfMatrixA_M, indexA_N = startIndexOfMatrixA_N;

	DataPostProcessingMethod *dataPostProcessing = executionContext->dataPostProcessing;
	BlockCopyMethod *blockCopy = executionContext->blockCopy;
	DetermineBlockBRangeMethod *determineBlockBRange = executionContext->determineBlockBRange;
	IterationIndexPostProcessMethod *iterationIndexPostProcess = executionContext->iterationIndexPostProcess;
	IndexRecordMethod *indexRecord = executionContext->indexRecord;

	goto JumpIn;
	
	for (/*indexA_M = indexA_M_begin*/; indexA_M < indexA_M_end || indexA_M_outOfIndexError(); indexA_M += strideA_M)
	{
		for (indexA_N = indexA_N_begin; indexA_N < indexA_N_end; indexA_N += strideA_N)
		{
		JumpIn:
			blockCopy(c_bufferA, matrixA,
				matrixA_M, matrixA_N,
				indexA_M, indexA_N, block_M, block_N);

#ifndef NDEBUG
			int sequenceBCount = 0;
#endif
			int indexB_M_begin, indexB_M_end;
			determineBlockBRange(&indexB_M_begin, &indexB_M_end,
				matrixB_M, block_M,
				neighbour_M, indexA_M);
			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
			{
				int indexB_N_begin, indexB_N_end;
				determineBlockBRange(&indexB_N_begin, &indexB_N_end,
					matrixB_N, block_N, neighbour_N, indexA_N);
				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
				{
					blockCopy(c_bufferB, matrixB,
						matrixB_M, matrixB_N,
						indexB_M, indexB_N, block_M, block_N);
					indexRecord(c_index_x_buffer++, c_index_y_buffer++, indexB_M, indexB_N);
					c_bufferB += blockSize;

#ifndef NDEBUG
					sequenceBCount++;
#endif
				}
			}

#ifndef NDEBUG
			if (sequenceBCount != numberOfBlockBPerBlockA)
				logger.critical("Internal logical error: sequenceBCount != numberOfBlockBPerBlockA");
#endif

			++numberOfBlockA;

			c_bufferA += blockSize;

			if (numberOfBlockA == lengthOfGpuTaskQueue)
			{/*
				if (checkIsInterruptPending())
					return 2;*/

				submitGpuTask<Type, processFunction>(matrixA_buffer, matrixB_buffer, matrixC_buffer,
					matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
					blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
					numberOfSubmitProcessors, numberOfSubmitThreadsPerProcessor, stream);

				CUDA_CHECK_POINT(cudaStreamSynchronize(stream));

				//std::swap(streamA, streamB);

				c_index_x_buffer = index_x_buffer;
				c_index_y_buffer = index_y_buffer;

				dataPostProcessing(&c_index_x, &c_index_y, &c_result, index_x_buffer, index_y_buffer,
					matrixC_buffer,
					numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain,
					rawIndexTemplate, rawIndexBuffer);

				//c_result += numTasks;
				c_bufferA = matrixA_buffer;
				c_bufferB = matrixB_buffer;

				numberOfBlockA = 0;
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

		int remainBlocks = numberOfBlockA * numberOfBlockBPerBlockA;

		submitGpuTask<Type, processFunction>(matrixA_buffer, matrixB_buffer,
			matrixC_buffer,
			matrixA_deviceBuffer, matrixB_deviceBuffer,
			matrixC_deviceBuffer,
			blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
			(remainBlocks + maxNumberOfThreadsPerProcessor - 1) / maxNumberOfThreadsPerProcessor,
			maxNumberOfThreadsPerProcessor, stream);

		CUDA_CHECK_POINT(cudaStreamSynchronize(stream));

		dataPostProcessing(&c_index_x, &c_index_y, &c_result, index_x_buffer, index_y_buffer, matrixC_buffer,
			numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain,
			rawIndexTemplate, rawIndexBuffer);
	}

	return 0;
}


template <typename Type,
	ProcessFunctionCPU<Type> processFunction>
	unsigned processWorker_cpu(ExecutionContext<Type> *executionContext)
{
	Type *matrixA = executionContext->matrixA, *matrixB = executionContext->matrixB,
		*matrixC = executionContext->matrixC,
		*matrixA_buffer = executionContext->matrixA_buffer, *matrixB_buffer = executionContext->matrixB_buffer,
		*matrixC_buffer = executionContext->matrixC_buffer;

	int matrixA_M = executionContext->matrixA_M, matrixA_N = executionContext->matrixA_N,
		matrixB_M = executionContext->matrixB_M, matrixB_N = executionContext->matrixB_N;
	int *index_x = executionContext->index_x, *index_y = executionContext->index_y,
		*index_x_buffer = executionContext->index_x_buffer, *index_y_buffer = executionContext->index_y_buffer,
		*rawIndexTemplate = executionContext->rawIndexTemplate, *rawIndexBuffer = executionContext->rawIndexBuffer,
		block_M = executionContext->block_M, block_N = executionContext->block_N,
		strideA_M = executionContext->strideA_M, strideA_N = executionContext->strideA_N,
		strideB_M = executionContext->strideB_M, strideB_N = executionContext->strideB_N,
		neighbour_M = executionContext->neighbour_M, neighbour_N = executionContext->neighbour_N,
		numberOfBlockBPerBlockA = executionContext->numberOfBlockBPerBlockA,
		numberOfIndexRetain = executionContext->numberOfIndexRetain,
		indexA_M_begin = executionContext->indexA_M_begin, indexA_N_begin = executionContext->indexA_N_begin,
		indexA_M_end = executionContext->indexA_M_end, indexA_N_end = executionContext->indexA_N_end,
		startIndexOfMatrixA_M = executionContext->startIndexOfMatrixA_M, startIndexOfMatrixA_N = executionContext->startIndexOfMatrixA_N;

	int blockSize = executionContext->block_M * executionContext->block_N;
	int *c_index_x = executionContext->index_x, *c_index_y = executionContext->index_y,
		*c_index_x_buffer = executionContext->index_x_buffer, *c_index_y_buffer = executionContext->index_y_buffer;

	Type *c_matrixC = matrixC, c_matrixC_buffer = matrixC_buffer;

	if (!numberOfIndexRetain)
		numberOfIndexRetain = numberOfBlockBPerBlockA;

	int indexA_M = startIndexOfMatrixA_M, indexA_N = startIndexOfMatrixA_N;

	DataPostProcessingMethod *dataPostProcessing = executionContext->dataPostProcessing;
	BlockCopyMethod *blockCopy = executionContext->blockCopy;
	DetermineBlockBRangeMethod *determineBlockBRange = executionContext->determineBlockBRange;
	IterationIndexPostProcessMethod *iterationIndexPostProcess = executionContext->iterationIndexPostProcess;

	goto JumpIn;

	for (indexA_M = indexA_M_begin; indexA_M < indexA_M_end || indexA_M_outOfIndexError(); indexA_M += strideA_M)
	{
		for (indexA_N = indexA_N_begin; indexA_N < indexA_N_end; indexA_N += strideA_N)
		{
		JumpIn:
			blockCopy(matrixA_buffer, matrixA,
				matrixA_M, matrixA_N,
				indexA_M, indexA_N, block_M, block_N);

#ifndef NDEBUG
			int sequenceBCount = 0;
#endif
			int indexB_M_begin, indexB_M_end;
			determineBlockBRange(&indexB_M_begin, &indexB_M_end,
				matrixB_M, block_M,
				neighbour_M, indexA_M);
			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
			{
				int indexB_N_begin, indexB_N_end;
				determineBlockBRange(&indexB_N_begin, &indexB_N_end,
					matrixB_N, block_N, neighbour_N, indexA_N);
				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
				{
					blockCopy(matrixB_buffer, matrixB,
						matrixB_M, matrixB_N,
						indexB_M, indexB_N, block_M, block_N);
					recordIndex(c_index_x_buffer++, c_index_y_buffer++, indexB_M, indexB_N);

					processFunction(matrixA_buffer, matrixB_buffer, blockSize, c_matrixC_buffer++);

#ifndef NDEBUG
					sequenceBCount++;
#endif
				}
			}

#ifndef NDEBUG
			if (sequenceBCount != numberOfBlockBPerBlockA)
				logger.critical("Internal logical error: sequenceBCount != numberOfBlockBPerBlockA");
#endif

			c_index_x_buffer = index_x_buffer;
			c_index_y_buffer = index_y_buffer;

			dataPostProcessing(c_index_x, c_index_y, c_matrixC, index_x_buffer, index_y_buffer,
				matrixC_buffer,
				1, numberOfBlockBPerBlockA, numberOfIndexRetain,
				rawIndexTemplate, rawIndexBuffer);

			c_index_x += numberOfIndexRetain;
			c_index_y += numberOfIndexRetain;
			c_matrixC += numberOfIndexRetain;

			iterationIndexPostProcess(&indexA_N, strideA_N, indexA_N_end);
		}
		iterationIndexPostProcess(&indexA_M, strideA_M, indexA_M_end);
	}

	return 0;
}