#pragma once

#include "lib_match_internal.h"

typedef cudaError_t(ProcessFunction)(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
typedef void(CopyBlockMethod)(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);
typedef void(SequenceBIndexMethod)(float *buf, float *src, int);

/*
 * All false are cuda error
 */
template <ProcessFunction processFunction>
cudaError_t submitGpuTask(float *bufferA, float *bufferB, float *resultBuffer, float *deviceBufferA, float *deviceBufferB, float *deviceResultBuffer,
	int blockSize,
	int numberOfBlockA, int numberOfBlockBPerBlockA,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	int numberOfBlockB = numberOfBlockA * numberOfBlockBPerBlockA;

	cudaError_t cuda_error = cudaMemcpyAsync(deviceBufferA, bufferA, numberOfBlockA * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (cuda_error != cudaSuccess) {
		CUDA_ERROR_CHECK_POINT(cuda_error);
		return cuda_error;
	}

	cuda_error = cudaMemcpyAsync(deviceBufferB, bufferB, numberOfBlockB * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (cuda_error != cudaSuccess) {
		CUDA_ERROR_CHECK_POINT(cuda_error);
		return cuda_error;
	}

	cuda_error = processFunction(deviceBufferA, deviceBufferB, numberOfBlockA, numberOfBlockB, numberOfBlockBPerBlockA, blockSize, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream);
	if (cuda_error != cudaSuccess) {
		CUDA_ERROR_CHECK_POINT(cuda_error);
		return cuda_error;
	}

	cuda_error = cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfBlockB * sizeof(float), cudaMemcpyDeviceToHost, stream);
	if (cuda_error != cudaSuccess)
		CUDA_ERROR_CHECK_POINT(cuda_error);

	return cuda_error;

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

typedef void DetermineBlockBIndex(int *, int *, int, int, int, int);

inline
void determineBlockB_index_local(int *indexB_begin, int *indexB_end, int matB, int block,
	int neighbour, int index_A)
{
	*indexB_begin = index_A - neighbour / 2;
	*indexB_end = index_A - neighbour / 2 + neighbour;
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
		indexA_M_begin = executionContext->indexA_M_begin, indexA_N_begin = executionContext->indexA_N_begin,
		indexA_M_end = executionContext->indexA_M_end, indexA_N_end = executionContext->indexA_N_end,
		startIndexOfMatrixA_M = executionContext->startIndexOfMatrixA_M, startIndexOfMatrixA_N = executionContext->startIndexOfMatrixA_N,
		numberOfIteration = executionContext->numberOfIteration;

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

	cudaError_t cuda_error;

	goto JumpIn;

	for (indexA_M = indexA_M_begin; indexA_M < indexA_M_end || indexA_M_outOfIndexError(); indexA_M += strideA_M)
	{
		for (indexA_N = indexA_N_begin; indexA_N < indexA_N_end; indexA_N += strideA_N)
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
				neighbour_M, indexA_M);
			for (int indexB_M = indexB_M_begin; indexB_M < indexB_M_end; indexB_M += strideB_M)
			{
				int indexB_N_begin, indexB_N_end;
				determineBlockB_index(&indexB_N_begin, &indexB_N_end,
					matrixB_N, block_N, neighbour_N, indexA_N);
				for (int indexB_N = indexB_N_begin; indexB_N < indexB_N_end; indexB_N += strideB_N)
				{
					copyBlock(c_bufferB, matrixB,
						matrixB_M, matrixB_N,
						indexB_M, indexB_N, block_M, block_N);
					recordIndexMethod(c_index_x_buffer++, c_index_y_buffer++, indexB_M, indexB_N);
					c_bufferB += blockSize;
					++numberOfQueuedTasks;

#ifndef NDEBUG
					sequenceBCount++;
#endif
				}
			}

#ifndef NDEBUG
			if (sequenceBCount != numberOfBlockBPerBlockA)
				logger.critical("Internal logical error: sequenceBCount != numberOfBlockBPerBlockA");
#endif


			numberOfBlockA += 1;

			c_bufferA += blockSize;

			if (numberOfQueuedTasks == lengthOfGpuTaskQueue)
			{
				cuda_error = submitGpuTask<processFunction>(matrixA_buffer, matrixB_buffer, matrixC_buffer,
					matrixA_deviceBuffer, matrixB_deviceBuffer, matrixC_deviceBuffer,
					blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
					numberOfSubmitProcessors, numberOfSubmitThreadsPerProcessor, streamA);

				if (cuda_error != cudaSuccess)
					goto CudaError;

				cuda_error = cudaStreamSynchronize(streamA);
				if (cuda_error != cudaSuccess)
					goto CudaError;

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

		cuda_error = submitGpuTask<processFunction>(matrixA_buffer, matrixB_buffer,
			matrixC_buffer,
			matrixA_deviceBuffer, matrixB_deviceBuffer,
			matrixC_deviceBuffer,
			blockSize, numberOfBlockA, numberOfBlockBPerBlockA,
			(remainBlocks + maxNumberOfThreadsPerProcessor - 1) / maxNumberOfThreadsPerProcessor,
			maxNumberOfThreadsPerProcessor, streamA);

		if (cuda_error != cudaSuccess)
			goto CudaError;

		cuda_error = cudaStreamSynchronize(streamA);
		if (cuda_error != cudaSuccess)
			goto CudaError;

		sortMethod(c_index_x, c_index_y, c_result, index_x_buffer, index_y_buffer, matrixC_buffer,
			numberOfBlockA, numberOfBlockBPerBlockA, numberOfIndexRetain,
			rawIndexTemplate, rawIndexBuffer);
	}

	return true;

CudaError:
	setLastErrorString("Error occured in worker execution, cudaError code: %d, error string: %s", cuda_error, cudaGetErrorString(cuda_error));
	return false;
}