#include "block_match.h"

#include "block_match_execute.hpp"

#include <tuple>
#include "stack_vector.hpp"


//
//template <ProcessFunction processFunction, ProcessFunction_BorderCheck processFunction_borderCheck,
//	CopyBlockMethod copyBlockAMethod, CopyBlockMethod copyBlockBMethod>
//	bool processWorker(float *matA, float *matB, float *result,
//		float *bufferA, int matA_M, int matA_N, int index_A_M_begin, int index_A_M_end, int index_A_N_begin, int index_A_N_end,
//		float *bufferB, int matB_M, int matB_N,
//		int block_M, int block_N,
//		int neighbour_M, int neighbour_N,
//		int strideA_M, int strideA_N,
//		int strideB_M, int strideB_N,
//		float *device_bufferA, float *device_bufferB, float *device_bufferC,
//		cudaStream_t stream, int numberOfGPUDeviceMultiProcessor, const int numberOfGPUProcessorThread)
//{
//	int blockSize = block_M * block_N;
//	int neighbourSize = (neighbour_M + strideB_M - 1) / strideB_M * (neighbour_N + strideB_N - 1) / strideB_N;
//
//	float *c_bufferA = bufferA;
//	float *c_bufferB = bufferB;
//	float *c_result = result;
//
//	int neighbour_M_middle = neighbour_M / 2;
//	int neighbour_N_middle = neighbour_N / 2;
//
//	int blocksPerProcessor = 0;
//	int filledProcessor = 0;
//	int numTasks = 0;
//	int numBlocks_A = 0, numBlocks_B = 0;
//
//	for (int ind_A_M = index_A_M_begin; ind_A_M < index_A_M_end; ind_A_M += strideA_M)
//	{
//		for (int ind_A_N = index_A_N_begin; ind_A_N < index_A_N_end; ind_A_N += strideA_N)
//		{
//			copyBlockAMethod(c_bufferA, matA, matA_M, matA_N, ind_A_M, ind_A_N, block_M, block_N);
//
//#ifndef NDEBUG
//			int sequenceBCount = 0;
//#endif
//
//			for (int ind_neighbour_M = 0; ind_neighbour_M < neighbour_M; ind_neighbour_M += strideB_M)
//			{
//				int index_x = ind_A_M - neighbour_M_middle + ind_neighbour_M;
//
//				for (int ind_neighbour_N = 0; ind_neighbour_N < neighbour_N; ind_neighbour_N += strideB_N)
//				{
//					int index_y = ind_A_N - neighbour_N_middle + ind_neighbour_N;
//
//					copyBlockBMethod(c_bufferB, matB, matB_M, matB_N, index_x, index_y, block_M, block_N);
//
//					c_bufferB += blockSize;
//
//#ifndef NDEBUG
//					sequenceBCount++;
//#endif
//				}
//			}
//
//#ifndef NDEBUG
//			if (sequenceBCount != neighbourSize) abort();
//#endif
//
//			numBlocks_B += neighbourSize;
//			numBlocks_A += 1;
//
//			numTasks += neighbourSize;
//
//			blocksPerProcessor += neighbourSize;
//			c_bufferA += blockSize;
//
//			if (blocksPerProcessor + neighbourSize > numberOfGPUProcessorThread)
//			{
//				filledProcessor++;
//
//				if (filledProcessor == numberOfGPUDeviceMultiProcessor)
//				{
//					cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
//					if (cuda_error != cudaSuccess)
//						return false;
//
//					cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
//					if (cuda_error != cudaSuccess)
//						return false;
//
//					cuda_error = processFunction(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC,
//						filledProcessor, blocksPerProcessor, stream);
//					if (cuda_error != cudaSuccess)
//						return false;
//
//					cuda_error = cudaStreamSynchronize(stream);
//					if (cuda_error != cudaSuccess)
//						return false;
//
//					cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
//					if (cuda_error != cudaSuccess)
//						return false;
//
//					c_result += numTasks;
//					c_bufferA = bufferA;
//					c_bufferB = bufferB;
//
//					numBlocks_A = 0;
//					numBlocks_B = 0;
//					numTasks = 0;
//					filledProcessor = 0;
//				}
//				blocksPerProcessor = 0;
//			}
//		}
//	}
//
//	if (numTasks)
//	{
//		cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
//		if (cuda_error != cudaSuccess)
//			return false;
//
//		cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
//		if (cuda_error != cudaSuccess)
//			return false;
//
//		cuda_error = processFunction_borderCheck(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC,
//			(numTasks + numberOfGPUProcessorThread - 1) / numberOfGPUProcessorThread, numberOfGPUProcessorThread, numTasks, stream);
//		if (cuda_error != cudaSuccess)
//			return false;
//
//		cuda_error = cudaStreamSynchronize(stream);
//		if (cuda_error != cudaSuccess)
//			return false;
//
//		cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
//		if (cuda_error != cudaSuccess)
//			return false;
//	}
//
//	return true;
//}



extern "C"
bool process_local(void *_instance, float *matA, float *matB, enum Method method, int **_index_x, int **_index_y, float **_result, int *dimensionOfResult)
{/*
	struct Context *instance = (struct Context *)_instance;
	ThreadPool &pool = globalContext.pool;

	unsigned numberOfThreads = globalContext.numberOfThreads;

	StackVector<void *, 4>task_handle(numberOfThreads);
	if (task_handle.bad_alloc()) return false;

	float *result = instance->result,
		*bufferA = instance->buffer_A,
		*bufferB = instance->buffer_B,
		*result_buffer = instance->result_buffer,
		*device_bufferA = instance->device_buffer_A,
		*device_bufferB = instance->device_buffer_B,
		*device_bufferC = instance->device_result_buffer;

	int matA_M = instance->matA_M,
		matA_N = instance->matA_N,
		matB_M = instance->matB_M,
		matB_N = instance->matB_N,
		block_M = instance->block_M,
		block_N = instance->block_N,
		strideA_M = instance->strideA_M,
		strideA_N = instance->strideA_N,
		strideB_M = instance->strideB_M,
		strideB_N = instance->strideB_N,
		sequenceAPadding_M = instance->sequenceAPadding_M,
		sequenceAPadding_N = instance->sequenceAPadding_N,
		sequenceBPadding_M = instance->sequenceBPadding_M,
		sequenceBPadding_N = instance->sequenceBPadding_N,
		numberOfBlockBPerBlockA = instance->numberOfBlockBPerBlockA,
		result_dim0 = instance->result_dims[0],
		result_dim1 = instance->result_dims[1],
		result_dim2 = instance->result_dims[2],
		result_dim3 = instance->result_dims[3],
		retain = instance->retain,
		perThreadBufferSize = instance->perThreadBufferSize,
		*index_x = instance->index_x, *index_y = instance->index_y,
		*index_x_buffer = instance->index_x_buffer, *index_y_buffer = instance->index_y_buffer,
		*index_buffer = instance->index_buffer,
		*index_buffer_sort = instance->index_buffer_sort;

	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;


	StackVector<
		std::tuple<float *, float *, float *,
		float *, int, int, int, int, int, int,
		float *, int, int,
		float *,
		int, int,
		int, int,
		int, int,
		int, int,
		int,
		float *, float *, float *,
		int *, int *, int *, int *,
		int *, int *,
		int,
		cudaStream_t, cudaStream_t,
		int, int >, 4>
		para_tuple(numberOfThreads);

	if (para_tuple.bad_alloc())
		return false;

	int ind_A_N_begin = -sequenceAPadding_N;
	int ind_A_N_end = determineEndOfIndex(matA_N, sequenceAPadding_N, block_N);

	for (unsigned i = 0; i < numberOfThreads; ++i)
	{
		int buffer_index = result_dim0 / numberOfThreads * i;

		int c_index_A_M_begin = buffer_index * strideA_M;
		c_index_A_M_begin -= sequenceAPadding_M;
		int c_index_A_M_end;
		if (i + 1 != numberOfThreads)
		{
			c_index_A_M_end = result_dim0 / numberOfThreads * (i + 1);
		}
		else
		{
			c_index_A_M_end = result_dim0;
		}
		c_index_A_M_end *= strideA_M;
		c_index_A_M_end -= sequenceAPadding_M;

		float *c_buffer_A = bufferA + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_buffer_B = bufferB + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_result = result + buffer_index * result_dim1 * result_dim2;
		float *c_result_buffer = result_buffer + i * perThreadBufferSize;
		int *c_index_x = index_x + buffer_index * result_dim1 * result_dim2;
		int *c_index_y = index_y + buffer_index * result_dim1 * result_dim2;
		int *c_index_x_buffer = index_x_buffer + i*perThreadBufferSize;
		int *c_index_y_buffer = index_y_buffer + i*perThreadBufferSize;
		int *c_index_buffer_sort = index_buffer_sort + i*numberOfBlockBPerBlockA;

		float *c_device_buffer_A = device_bufferA + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_device_buffer_B = device_bufferB + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_device_buffer_C = device_bufferC + i * perThreadBufferSize;

		cudaStream_t streamA = instance->stream[2 * i],
			streamB = instance->stream[2 * i + 1];


		para_tuple[i] =
			std::make_tuple(matA, matB, c_result,
				c_buffer_A, matA_M, matA_N, c_index_A_M_begin, c_index_A_M_end, ind_A_N_begin, ind_A_N_end,
				c_buffer_B, matB_M, matB_N,
				c_result_buffer,
				block_M, block_N,
				sequenceBPadding_M, sequenceBPadding_N,
				strideA_M, strideA_N,
				strideB_M, strideB_N,
				numberOfBlockBPerBlockA,
				c_device_buffer_A, c_device_buffer_B, c_device_buffer_C,
				c_index_x, c_index_y, c_index_x_buffer, c_index_y_buffer,
				index_buffer, c_index_buffer_sort,
				retain,
				streamA, streamB,
				numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

		if (method == MSE)
			if (retain)
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker<block_match_mse_check_border, block_match_mse_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex_partial>),
					para_tuple[i]);
			else
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker<block_match_mse_check_border, block_match_mse_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex>),
					para_tuple[i]);
		else if (method == CC)
			if (retain)
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker<block_match_cc_check_border, block_match_cc_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex_partial>), para_tuple[i]);
			else
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker<block_match_cc_check_border, block_match_cc_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex>), para_tuple[i]);
	}

	for (unsigned i = 0; i < numberOfThreads; ++i)
	{
		pool.join(task_handle[i]);
	}
	bool isFailed = false;
	for (unsigned i = 0; i < numberOfThreads; ++i)
	{
		if (pool.get_rc(task_handle[i]) == false)
			isFailed = true;
		pool.release(task_handle[i]);
	}

	if (!isFailed)
	{
		*_index_x = index_x;
		*_index_y = index_y;
		*_result = result;
		memcpy(dimensionOfResult, instance->result_dims, sizeof(*dimensionOfResult) * 4);
	}

	return !isFailed;*/
return false;
}


extern "C"
bool execute(void *_instance, float *matA, float *matB, enum Method method, int **_index_x, int **_index_y, float **_result, int *dimensionOfResult)
{
	struct Context *instance = (struct Context *)_instance;
	ThreadPool &pool = globalContext.pool;

	unsigned numberOfThreads = globalContext.numberOfThreads;

	StackVector<void *, 4>task_handle(numberOfThreads);
	if (task_handle.bad_alloc()) return false;

	float *result = instance->result,
		*bufferA = instance->buffer_A,
		*bufferB = instance->buffer_B,
		*result_buffer = instance->result_buffer,
		*device_bufferA = instance->device_buffer_A,
		*device_bufferB = instance->device_buffer_B,
		*device_bufferC = instance->device_result_buffer;

	int matA_M = instance->matA_M,
		matA_N = instance->matA_N,
		matB_M = instance->matB_M,
		matB_N = instance->matB_N,
		block_M = instance->block_M,
		block_N = instance->block_N,
		strideA_M = instance->strideA_M,
		strideA_N = instance->strideA_N,
		strideB_M = instance->strideB_M,
		strideB_N = instance->strideB_N,
		sequenceAPadding_M = instance->sequenceAPadding_M,
		sequenceAPadding_N = instance->sequenceAPadding_N,
		sequenceBPadding_M = instance->sequenceBPadding_M,
		sequenceBPadding_N = instance->sequenceBPadding_N,
		numberOfBlockBPerBlockA = instance->numberOfBlockBPerBlockA,
		result_dim0 = instance->result_dims[0],
		result_dim1 = instance->result_dims[1],
		result_dim2 = instance->result_dims[2],
		result_dim3 = instance->result_dims[3],
		retain = instance->retain,
		perThreadBufferSize = instance->perThreadBufferSize,
		*index_x = instance->index_x, *index_y = instance->index_y,
		*index_x_buffer = instance->index_x_buffer, *index_y_buffer = instance->index_y_buffer,
		*index_buffer = instance->index_buffer,
		*index_buffer_sort = instance->index_buffer_sort;

	auto *parameterBuffer = instance->parameterBuffer;

	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;
	
	int ind_A_N_begin = -sequenceAPadding_N;
	int ind_A_N_end = determineEndOfIndex(matA_N, sequenceAPadding_N, block_N);

	for (unsigned i = 0; i < numberOfThreads; ++i)
	{
		int buffer_index = result_dim0 / numberOfThreads * i;

		int c_index_A_M_begin = buffer_index * strideA_M;
		c_index_A_M_begin -= sequenceAPadding_M;
		int c_index_A_M_end;
		if (i + 1 != numberOfThreads)
		{
			c_index_A_M_end = result_dim0 / numberOfThreads * (i + 1);
		}
		else
		{
			c_index_A_M_end = result_dim0;
		}
		c_index_A_M_end *= strideA_M;
		c_index_A_M_end -= sequenceAPadding_M;

		float *c_buffer_A = bufferA + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_buffer_B = bufferB + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_result = result + buffer_index * result_dim1 * result_dim2;
		float *c_result_buffer = result_buffer + i * perThreadBufferSize;
		int *c_index_x = index_x + buffer_index * result_dim1 * result_dim2;
		int *c_index_y = index_y + buffer_index * result_dim1 * result_dim2;
		int *c_index_x_buffer = index_x_buffer + i*perThreadBufferSize;
		int *c_index_y_buffer = index_y_buffer + i*perThreadBufferSize;
		int *c_index_buffer_sort = index_buffer_sort + i*numberOfBlockBPerBlockA;

		float *c_device_buffer_A = device_bufferA + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_device_buffer_B = device_bufferB + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_device_buffer_C = device_bufferC + i * perThreadBufferSize;

		cudaStream_t streamA = instance->stream[2 * i],
			streamB = instance->stream[2 * i + 1];


		parameterBuffer[i] =
			std::make_tuple(matA, matB, c_result,
				c_buffer_A, matA_M, matA_N, c_index_A_M_begin, c_index_A_M_end, ind_A_N_begin, ind_A_N_end,
				c_buffer_B, matB_M, matB_N,
				c_result_buffer,
				block_M, block_N,
				sequenceBPadding_M, sequenceBPadding_N,
				strideA_M, strideA_N,
				strideB_M, strideB_N,
				numberOfBlockBPerBlockA,
				c_device_buffer_A, c_device_buffer_B, c_device_buffer_C,
				c_index_x, c_index_y, c_index_x_buffer, c_index_y_buffer,
				index_buffer, c_index_buffer_sort,
				retain,
				streamA, streamB,
				numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);
		/*
		if (method == MSE)
			if (retain)
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker_full<block_match_mse_check_border, block_match_mse_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex_partial>),
					parameterBuffer[i]);
			else
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker_full<block_match_mse_check_border, block_match_mse_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex>),
					parameterBuffer[i]);
		else if (method == CC)
			if (retain)
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker_full<block_match_cc_check_border, block_match_cc_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex_partial>), parameterBuffer[i]);
			else
				task_handle[i] =
				thread_pool_launcher(pool,
				(processWorker_full<block_match_cc_check_border, block_match_cc_check_border,
					copyBlockWithSymmetricPadding, copyBlockWithSymmetricPadding,
					sortWithIndex>), parameterBuffer[i]);*/
	}

	for (unsigned i = 0; i < numberOfThreads; ++i)
	{
		pool.join(task_handle[i]);
	}
	bool isFailed = false;
	for (unsigned i = 0; i < numberOfThreads; ++i)
	{
		if (pool.get_rc(task_handle[i]) == false)
			isFailed = true;
		pool.release(task_handle[i]);
	}

	if (!isFailed)
	{
		*_index_x = index_x;
		*_index_y = index_y;
		*_result = result;
		memcpy(dimensionOfResult, instance->result_dims, sizeof(*dimensionOfResult) * 4);
	}

	return !isFailed;
}
