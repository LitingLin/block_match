#pragma once

#include "lib_match_internal.h"

typedef cudaError_t(ProcessFunction)(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
typedef void(CopyBlockMethod)(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);
typedef void(SequenceBIndexMethod)(float *buf, float *src, int);

template <ProcessFunction processFunction>
bool submitGpuTask(float *bufferA, float *bufferB, float *resultBuffer, float *deviceBufferA, float *deviceBufferB, float *deviceResultBuffer,
	int blockSize,
	int numberOfBlockA, int numberOfBlockBPerBlockA,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	int numberOfBlockB = numberOfBlockA * numberOfBlockBPerBlockA;

	cudaError_t cuda_error = cudaMemcpyAsync(deviceBufferA, bufferA, numberOfBlockA * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (cuda_error != cudaSuccess)
		return false;

	cuda_error = cudaMemcpyAsync(deviceBufferB, bufferB, numberOfBlockB * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (cuda_error != cudaSuccess)
		return false;

	cuda_error = processFunction(deviceBufferA, deviceBufferB, numberOfBlockA, numberOfBlockB, numberOfBlockBPerBlockA, blockSize, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream);
	if (cuda_error != cudaSuccess)
		return false;

	cuda_error = cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfBlockB * sizeof(float), cudaMemcpyDeviceToHost, stream);
	if (cuda_error != cudaSuccess)
		return false;

	return true;
}

typedef void(SortMethod)(int *&index_x, int *&index_y, float *&result,
	int *index_x_buffer, int *index_y_buffer, float *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort);

inline
void sortWithIndex_partial(int *&index_x, int *&index_y, float *&result,
	int *index_x_buffer, int *index_y_buffer, float *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		memcpy(index_buffer_sort, index_buffer, numberOfBlockBPerBlockA * sizeof(*index_buffer_sort));

		block_sort_partial(index_buffer_sort, result_buffer, numberOfBlockBPerBlockA, retain);

		for (int j = 0; j < retain; ++j)
		{
			*result++ = result_buffer[index_buffer_sort[j]];

			*index_x++ = index_x_buffer[index_buffer_sort[j]];
			*index_y++ = index_y_buffer[index_buffer_sort[j]];
		}

		index_x_buffer += numberOfBlockBPerBlockA;
		index_y_buffer += numberOfBlockBPerBlockA;
		result_buffer += numberOfBlockBPerBlockA;
	}
}

inline
void sortWithIndex(int *&index_x, int *&index_y, float *&result,
	int *index_x_buffer, int *index_y_buffer, float *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{
	for (int i = 0; i < numberOfBlockA; ++i)
	{
		memcpy(index_buffer_sort, index_buffer, numberOfBlockBPerBlockA * sizeof(*index_buffer_sort));

		block_sort(index_buffer_sort, result_buffer, numberOfBlockBPerBlockA);

		for (int j = 0; j < retain; ++j)
		{
			*result++ = result_buffer[index_buffer_sort[j]];

			*index_x++ = index_x_buffer[index_buffer_sort[j]];
			*index_y++ = index_y_buffer[index_buffer_sort[j]];
		}

		index_x_buffer += numberOfBlockBPerBlockA;
		index_y_buffer += numberOfBlockBPerBlockA;
		result_buffer += numberOfBlockBPerBlockA;
	}
}
inline
void dummySort(int *&index_x, int *&index_y, float *&result,
	int *index_x_buffer, int *index_y_buffer, float *result_buffer,
	int numberOfBlockA, int numberOfBlockBPerBlockA, int retain,
	const int *index_buffer, int *index_buffer_sort)
{

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
/*
typedef void DetermineBlockBIndex(int &, int &, int, int, int, int, int);

inline
void determineBlockB_index_local(int &indexB_begin, int &indexB_end, int matB, int padB, int block, int neighbour, int index_A)
{
	indexB_begin = index_A - neighbour / 2;
	indexB_end = index_A - neighbour / 2 + neighbour;
}

inline
void determineBlockB_index_full(int &indexB_begin, int &indexB_end, int matB, int padB, int block, int neighbour, int index_A)
{
	indexB_begin = -padB;
	indexB_end = determineEndOfIndex(matB, padB, block);
}
*/
typedef void DetermineBlockBIndex(int *, int *, int, int, int, int, int);

inline
void determineBlockB_index_local(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A, int matrixBPadding_pre)
{
	*indexB_begin = index_A - neighbour / 2 + matrixBPadding_pre;
	*indexB_end = index_A - neighbour / 2 + neighbour + matrixBPadding_pre;
}

inline
void determineBlockB_index_full(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A, int matrixBPadding_pre)
{
	*indexB_begin = 0;
	*indexB_end = determineEndOfIndex(matB, block);
}
/*
// TODO: Fix busy waiting gpu tasks
template <DetermineBlockBIndex determineBlockB_M_index, DetermineBlockBIndex determineBlockB_N_index,
	RecordIndex recordIndexMethod,
	ProcessFunction processFunctionA, ProcessFunction processFunctionB,
	CopyBlockMethod copyBlockAMethod, CopyBlockMethod copyBlockBMethod,
	SortMethod sortMethod>
	bool processWorker(float *matrixA, float *matrixB, float *matrixC,
		float *matrixA_buffer, int matrixA_M, int matrixA_N,
		float *matrixB_buffer, int matrixB_M, int matrixB_N,
		float *matrixC_buffer,
		float *matrixA_deviceBuffer, float *matrixB_deviceBuffer, float *matrixC_deviceBuffer,
		int *index_x, int *index_y, int *index_x_buffer, int *index_y_buffer,
		int *rawIndexTemplate, int *rawIndexBuffer,
		int block_M, int block_N,
		int padB_M, int padB_N,
		int strideA_M, int strideA_N,
		int strideB_M, int strideB_N,
		int neighbour_M, int neighbour_N,
		int numberOfBlockBPerBlockA,
		int numberOfIndexRetain,
		int rawBeginIndex_matrixA, int rawEndIndex_matrixA,

		cudaStream_t streamA, cudaStream_t streamB, // TODO: Double buffering
		int maxNumberOfThreadsPerProcessor,
		int numberOfSubmitThreadsPerProcessor, int numberOfSubmitProcessors, int lengthOfGpuTaskQueue)
{
	int blockSize = block_M * block_N;
	float *c_bufferA = matrixA_buffer;
	float *c_bufferB = matrixB_buffer;
	float *c_result = matrixC;
	int *c_index_x = index_x, *c_index_y = index_y,
		*c_index_x_buffer = index_x_buffer, *c_index_y_buffer = index_y_buffer;

	int numberOfBlockA = 0;

	if (!numberOfIndexRetain)
		numberOfIndexRetain = numberOfBlockBPerBlockA;

	int indexOfIteration = 0;



	for (int indexA_M = index_A_M_begin; indexA_M < index_A_M_end; indexA_M += strideA_M)
	{
		for (int indexA_N = index_A_N_begin; indexA_N < index_A_N_end; indexA_N += strideA_N)
		{
			copyBlockAMethod(c_bufferA, matrixA, matrixA_M, matrixA_N, indexA_M, indexA_N, block_M, block_N);

#ifndef NDEBUG
			int sequenceBCount = 0;
#endif
			int indexB_M_begin, indexB_M_end;
			determineBlockB_M_index(indexB_M_begin, indexB_M_end, matrixB_M, padB_M, block_M, neighbour_M, indexA_M);
			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
			{
				int indexB_N_begin, indexB_N_end;
				determineBlockB_N_index(indexB_N_begin, indexB_N_end, matrixB_N, padB_N, block_N, neighbour_N, indexA_N);
				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
				{
					copyBlockBMethod(c_bufferB, matrixB, matrixB_M, matrixB_N, indexB_M, indexB_N, block_M, block_N);
					recordIndexMethod(c_index_x_buffer, c_index_y_buffer, indexB_M, indexB_N);
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

			++indexOfIteration;

			numberOfBlockA += 1;

			c_bufferA += blockSize;

			if (indexOfIteration == lengthOfGpuTaskQueue)
			{
				if (!submitGpuTask<processFunctionA>(matrixA_buffer, matrixB_buffer, matrixC_buffer, matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
					blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
					numberOfSubmitProcessors, numberOfSubmitThreadsPerProcessor, streamA))
					return false;

				cudaError_t cuda_error = cudaStreamSynchronize(streamA);
				if (cuda_error != cudaSuccess)
					return false;

				//std::swap(streamA, streamB);

				c_index_x_buffer = index_x_buffer;
				c_index_y_buffer = index_y_buffer;

				sortMethod(c_index_x, c_index_y, c_result, index_x_buffer, index_y_buffer, matrixC_buffer,
					numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain, rawIndexTemplate, rawIndexBuffer);

				//c_result += numTasks;
				c_bufferA = matrixA_buffer;
				c_bufferB = matrixB_buffer;

				numberOfBlockA = 0;
				indexOfIteration = 0;
			}
		}
	}

	if (numberOfBlockA)
	{
		int remainBlocks = numberOfBlockA * numberOfBlockBPerBlockA;

		if (!submitGpuTask<processFunctionB>(matrixA_buffer, matrixB_buffer, matrixC_buffer, matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
			blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
			(remainBlocks + maxNumberOfThreadsPerProcessor - 1) / maxNumberOfThreadsPerProcessor, maxNumberOfThreadsPerProcessor, streamA))
			return false;

		cudaError_t cuda_error = cudaStreamSynchronize(streamA);
		if (cuda_error != cudaSuccess)
			return false;

		sortMethod(c_index_x, c_index_y, c_result, index_x_buffer, index_y_buffer, matrixC_buffer,
			numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain, rawIndexTemplate, rawIndexBuffer);
	}

	return true;
}*/

inline
bool indexA_M_outOfIndexError()
{
	logger.critical("Internal logical error: indexA_M out of index");
	return false;
}

// TODO: Fix busy waiting gpu tasks
template <DetermineBlockBIndex determineBlockB_index,
	RecordIndex recordIndexMethod,
	ProcessFunction processFunction,
	SortMethod sortMethod>
	unsigned processWorker(ExecutionContext *executionContext)
{
	float *matrixA = executionContext->matrixA, *matrixB = executionContext->matrixB,
		*matrixC = executionContext->matrixC,
		*matrixA_buffer = executionContext->matrixA_buffer, *matrixB_buffer = executionContext->matrixB_buffer,
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
		startIndexOfMatrixA_M = executionContext->startIndexOfMatrixA_M, startIndexOfMatrixA_N = executionContext->startIndexOfMatrixA_N,
		numberOfIteration = executionContext->numberOfIteration;
	int matrixBPadding_M_pre = executionContext->matrixBPadding_M_pre, matrixBPadding_N_pre = executionContext->matrixBPadding_N_pre;

	cudaStream_t streamA = executionContext->streamA, streamB = executionContext->streamB; // TODO: Double buffering
	int maxNumberOfThreadsPerProcessor = executionContext->maxNumberOfThreadsPerProcessor,
		numberOfSubmitThreadsPerProcessor = executionContext->numberOfSubmitThreadsPerProcessor,
		numberOfSubmitProcessors = executionContext->numberOfSubmitProcessors,
		lengthOfGpuTaskQueue = executionContext->lengthOfGpuTaskQueue;

	int blockSize = executionContext->block_M * executionContext->block_N;
	float *c_bufferA = executionContext->matrixA_buffer;
	float *c_bufferB = executionContext->matrixB_buffer;
	float *c_result = executionContext->matrixC;
	int *c_index_x = executionContext->index_x, *c_index_y = executionContext->index_y,
		*c_index_x_buffer = executionContext->index_x_buffer, *c_index_y_buffer = executionContext->index_y_buffer;

	int numberOfBlockA = 0;
	if (!numberOfIndexRetain)
		numberOfIndexRetain = numberOfBlockBPerBlockA;

	int numberOfQueuedTasks = 0, indexOfIteration = 0;
	int indexA_M = startIndexOfMatrixA_M, indexA_N = startIndexOfMatrixA_N;

	int indexA_M_end = determineEndOfIndex(matrixA_M, block_M);
	int indexA_N_end = determineEndOfIndex(matrixA_N, block_N);

	goto JumpIn;

	for (indexA_M = 0; indexA_M < indexA_M_end || indexA_M_outOfIndexError(); indexA_M += strideA_M)
	{
		for (indexA_N = 0; indexA_N < indexA_N_end; indexA_N += strideA_N)
		{
		JumpIn:
			copyBlock(c_bufferA, matrixA,
				matrixA_M, matrixA_N,
				indexA_M, indexA_N, block_M, block_N);

#ifndef NDEBUG
			int sequenceBCount = 0;
#endif
			int indexB_M_begin, indexB_M_end;
			determineBlockB_index(&indexB_M_begin, &indexB_M_end,
				matrixB_M, block_M,
				neighbour_M, indexA_M,
				matrixBPadding_M_pre);
			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
			{
				int indexB_N_begin, indexB_N_end;
				determineBlockB_index(&indexB_N_begin, &indexB_N_end,
					matrixB_N, block_N, neighbour_N, indexA_N,
					matrixBPadding_N_pre);
				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
				{
					copyBlock(c_bufferB, matrixB,
						matrixB_M, matrixB_N,
						indexB_M, indexB_N, block_M, block_N);
					recordIndexMethod(c_index_x_buffer++, c_index_y_buffer++, indexB_M, indexB_N);
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

			++numberOfQueuedTasks;

			numberOfBlockA += 1;

			c_bufferA += blockSize;

			if (numberOfQueuedTasks == lengthOfGpuTaskQueue)
			{
				if (!submitGpuTask<processFunction>(matrixA_buffer, matrixB_buffer, matrixC_buffer,
					matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
					blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
					numberOfSubmitProcessors, numberOfSubmitThreadsPerProcessor, streamA))
					return false;

				cudaError_t cuda_error = cudaStreamSynchronize(streamA);
				if (cuda_error != cudaSuccess)
					return false;

				//std::swap(streamA, streamB);

				c_index_x_buffer = index_x_buffer;
				c_index_y_buffer = index_y_buffer;

				sortMethod(c_index_x, c_index_y, c_result, index_x_buffer, index_y_buffer,
					matrixC_buffer,
					numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain,
					rawIndexTemplate, rawIndexBuffer);

				//c_result += numTasks;
				c_bufferA = matrixA_buffer;
				c_bufferB = matrixB_buffer;

				numberOfBlockA = 0;
				numberOfQueuedTasks = 0;
			}

			++indexOfIteration;

			if (indexOfIteration == numberOfIteration)
				goto JumpOut;
			}
		}
JumpOut:
	if (numberOfBlockA)
	{
		int remainBlocks = numberOfBlockA * numberOfBlockBPerBlockA;

		if (!submitGpuTask<processFunction>(matrixA_buffer, matrixB_buffer,
			matrixC_buffer,
			matrixA_deviceBuffer, matrixB_deviceBuffer,
			matrixC_deviceBuffer,
			blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
			(remainBlocks + maxNumberOfThreadsPerProcessor - 1) / maxNumberOfThreadsPerProcessor,
			maxNumberOfThreadsPerProcessor, streamA))
			return false;

		cudaError_t cuda_error = cudaStreamSynchronize(streamA);
		if (cuda_error != cudaSuccess)
			return false;

		sortMethod(c_index_x, c_index_y, c_result, index_x_buffer, index_y_buffer, matrixC_buffer,
			numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain,
			rawIndexTemplate, rawIndexBuffer);
	}

	return true;
	}