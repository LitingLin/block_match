#include "lib_match.h"

#include "block_match_execute.hpp"

#include <tuple>
#include "stack_vector.hpp"


//
//template <ProcessFunction processFunction, ProcessFunction_BorderCheck processFunction_borderCheck,
//	CopyBlockMethod copyBlockAMethod, CopyBlockMethod copyBlockBMethod>
//	bool processWorker(float *matA, float *matB, float *C,
//		float *bufferA, int matrixA_M, int matrixA_N, int index_A_M_begin, int index_A_M_end, int index_A_N_begin, int index_A_N_end,
//		float *bufferB, int matrixB_M, int matrixB_N,
//		int block_M, int block_N,
//		int searchRegion_M, int searchRegion_N,
//		int strideA_M, int strideA_N,
//		int strideB_M, int strideB_N,
//		float *device_bufferA, float *device_bufferB, float *device_bufferC,
//		cudaStream_t stream, int numberOfGPUDeviceMultiProcessor, const int numberOfGPUProcessorThread)
//{
//	int blockSize = block_M * block_N;
//	int neighbourSize = (searchRegion_M + strideB_M - 1) / strideB_M * (searchRegion_N + strideB_N - 1) / strideB_N;
//
//	float *c_bufferA = bufferA;
//	float *c_bufferB = bufferB;
//	float *c_result = C;
//
//	int neighbour_M_middle = searchRegion_M / 2;
//	int neighbour_N_middle = searchRegion_N / 2;
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
//			copyBlockAMethod(c_bufferA, matA, matrixA_M, matrixA_N, ind_A_M, ind_A_N, block_M, block_N);
//
//#ifndef NDEBUG
//			int sequenceBCount = 0;
//#endif
//
//			for (int ind_neighbour_M = 0; ind_neighbour_M < searchRegion_M; ind_neighbour_M += strideB_M)
//			{
//				int index_x = ind_A_M - neighbour_M_middle + ind_neighbour_M;
//
//				for (int ind_neighbour_N = 0; ind_neighbour_N < searchRegion_N; ind_neighbour_N += strideB_N)
//				{
//					int index_y = ind_A_N - neighbour_N_middle + ind_neighbour_N;
//
//					copyBlockBMethod(c_bufferB, matB, matrixB_M, matrixB_N, index_x, index_y, block_M, block_N);
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
bool process_local(void *_instance, float *matA, float *matB, LibMatchMeasureMethod method, int **_index_x, int **_index_y, float **_result, int *dimensionOfResult)
{/*
	struct BlockMatchContext *instance = (struct BlockMatchContext *)_instance;
	ThreadPool &pool = globalContext.pool;

	unsigned numberOfThreads = globalContext.numberOfThreads;

	StackVector<void *, 4>task_handle(numberOfThreads);
	if (task_handle.bad_alloc()) return false;

	float *C = instance->C,
		*bufferA = instance->matrixA_buffer,
		*bufferB = instance->matrixB_buffer,
		*matrixC_buffer = instance->matrixC_buffer,
		*device_bufferA = instance->matrixA_deviceBuffer,
		*device_bufferB = instance->matrixB_deviceBuffer,
		*device_bufferC = instance->matrixC_deviceBuffer;

	int matrixA_M = instance->matrixA_M,
		matrixA_N = instance->matrixA_N,
		matrixB_M = instance->matrixB_M,
		matrixB_N = instance->matrixB_N,
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
		result_dim0 = instance->C_dimensions[0],
		result_dim1 = instance->C_dimensions[1],
		result_dim2 = instance->C_dimensions[2],
		result_dim3 = instance->C_dimensions[3],
		numberOfIndexRetain = instance->numberOfIndexRetain,
		perThreadBufferSize = instance->perThreadBufferSize,
		*index_x = instance->index_x, *index_y = instance->index_y,
		*index_x_sorting_buffer = instance->index_x_sorting_buffer, *index_y_sorting_buffer = instance->index_y_sorting_buffer,
		*common_buffer = instance->common_buffer,
		*index_raw_sorting_buffer = instance->index_raw_sorting_buffer;

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
	int ind_A_N_end = determineEndOfIndex(matrixA_N, sequenceAPadding_N, block_N);

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
		float *c_result = C + buffer_index * result_dim1 * result_dim2;
		float *c_result_buffer = matrixC_buffer + i * perThreadBufferSize;
		int *c_index_x = index_x + buffer_index * result_dim1 * result_dim2;
		int *c_index_y = index_y + buffer_index * result_dim1 * result_dim2;
		int *c_index_x_buffer = index_x_sorting_buffer + i*perThreadBufferSize;
		int *c_index_y_buffer = index_y_sorting_buffer + i*perThreadBufferSize;
		int *c_index_buffer_sort = index_raw_sorting_buffer + i*numberOfBlockBPerBlockA;

		float *c_device_buffer_A = device_bufferA + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_device_buffer_B = device_bufferB + i * numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread * block_M * block_N;
		float *c_device_buffer_C = device_bufferC + i * perThreadBufferSize;

		cudaStream_t streamA = instance->stream[2 * i],
			streamB = instance->stream[2 * i + 1];


		para_tuple[i] =
			std::make_tuple(matA, matB, c_result,
				c_buffer_A, matrixA_M, matrixA_N, c_index_A_M_begin, c_index_A_M_end, ind_A_N_begin, ind_A_N_end,
				c_buffer_B, matrixB_M, matrixB_N,
				c_result_buffer,
				block_M, block_N,
				sequenceBPadding_M, sequenceBPadding_N,
				strideA_M, strideA_N,
				strideB_M, strideB_N,
				numberOfBlockBPerBlockA,
				c_device_buffer_A, c_device_buffer_B, c_device_buffer_C,
				c_index_x, c_index_y, c_index_x_buffer, c_index_y_buffer,
				common_buffer, c_index_buffer_sort,
				numberOfIndexRetain,
				streamA, streamB,
				numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

		if (method == LIB_MATCH_MSE)
			if (numberOfIndexRetain)
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
		else if (method == LIB_MATCH_CC)
			if (numberOfIndexRetain)
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
		*_result = C;
		memcpy(dimensionOfResult, instance->C_dimensions, sizeof(*dimensionOfResult) * 4);
	}

	return !isFailed;*/
	return false;
}

bool blockMatchExecute_(void *_instance, float *A, float *B,
	float *C,
	float *padded_A, float *padded_B,
	int *index_x = nullptr, int *index_y = nullptr)
{
	BlockMatchContext *instance = static_cast<BlockMatchContext*>(_instance);

	int A_M = instance->matrixA_M,
		A_N = instance->matrixA_N,
		B_M = instance->matrixB_M,
		B_N = instance->matrixB_N,
		A_M_padPre = instance->matrixAPadding_M_pre,
		A_M_padPost = instance->matrixAPadding_M_post,
		A_N_padPre = instance->matrixAPadding_N_pre,
		A_N_padPost = instance->matrixAPadding_N_post,
		B_M_padPre = instance->matrixBPadding_M_pre,
		B_M_padPost = instance->matrixBPadding_M_post,
		B_N_padPre = instance->matrixBPadding_N_pre,
		B_N_padPost = instance->matrixBPadding_N_post;

	instance->padMethod(A, padded_A, A_M, A_N, A_M_padPre, A_M_padPost, A_N_padPre, A_N_padPost);
	instance->padMethod(B, padded_B, B_M, B_N, B_M_padPre, B_M_padPost, B_N_padPre, B_N_padPost);

	unsigned numberOfThreads = globalContext.numberOfThreads;
	int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;
	

	for (unsigned i=0;i<numberOfThreads;++i)
	{
		float *c_matrixA_padded = instance->perThreadBufferPointer.matrixA_padded[i];
		float *c_matrixB_padded = instance->perThreadBufferPointer.matrixB_padded[i];
		float *c_matrixA_buffer = instance->perThreadBufferPointer.matrixA_buffer[i];
		float *c_matrixB_buffer = instance->perThreadBufferPointer.matrixB_buffer[i];
		float *c_matrixC_buffer = instance->perThreadBufferPointer.matrixC_buffer[i];
		float *c_matrixA_deviceBuffer = instance->perThreadBufferPointer.matrixA_deviceBuffer[i];
		float *c_matrixB_deviceBuffer = instance->perThreadBufferPointer.matrixB_deviceBuffer[i];
		float *c_matrixC_deviceBuffer = instance->perThreadBufferPointer.matrixC_deviceBuffer[i];

		int *c_index_x_sorting_buffer = instance->perThreadBufferPointer.index_x_sorting_buffer[i];
		int *c_index_y_sorting_buffer = instance->perThreadBufferPointer.index_y_sorting_buffer[i];
		int *c_index_x = instance->perThreadBufferPointer.index_x[i];
		int *c_index_y = instance->perThreadBufferPointer.index_y[i];

		int *c_index_raw_sorting_buffer = instance->perThreadBufferPointer.index_raw_sorting_buffer[i];
		instance->parameterBuffer[i] = {};
		thread_pool_launcher(globalContext.pool, instance->executionMethod, instance->parameterBuffer[i]);
	}

	return true;
}

bool blockMatchExecute(void *_instance, float *matA, float *matB, LibMatchMeasureMethod method, int **_index_x, int **_index_y, float **_result, int *dimensionOfResult)
{
	struct BlockMatchContext *instance = static_cast<struct BlockMatchContext *>(_instance);
	ThreadPool &pool = globalContext.pool;

	unsigned numberOfThreads = globalContext.numberOfThreads;

	StackVector<void *, 4>task_handle(numberOfThreads);
	if (task_handle.bad_alloc()) return false;

	float *result = instance->C,
		*bufferA = instance->matrixA_buffer,
		*bufferB = instance->matrixB_buffer,
		*result_buffer = instance->matrixC_buffer,
		*device_bufferA = instance->matrixA_deviceBuffer,
		*device_bufferB = instance->matrixB_deviceBuffer,
		*device_bufferC = instance->matrixC_deviceBuffer;

	int matA_M = instance->matrixA_M,
		matA_N = instance->matrixA_N,
		matB_M = instance->matrixB_M,
		matB_N = instance->matrixB_N,
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
		result_dim0 = instance->C_dimensions[0],
		result_dim1 = instance->C_dimensions[1],
		result_dim2 = instance->C_dimensions[2],
		result_dim3 = instance->C_dimensions[3],
		retain = instance->numberOfIndexRetain,
		perThreadBufferSize = instance->perThreadBufferSize,
		*index_x = instance->index_x, *index_y = instance->index_y,
		*index_x_buffer = instance->index_x_sorting_buffer, *index_y_buffer = instance->index_y_sorting_buffer,
		*index_buffer = instance->common_buffer,
		*index_buffer_sort = instance->index_raw_sorting_buffer;

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
		if (method == LIB_MATCH_MSE)
			if (numberOfIndexRetain)
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
		else if (method == LIB_MATCH_CC)
			if (numberOfIndexRetain)
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
		memcpy(dimensionOfResult, instance->C_dimensions, sizeof(*dimensionOfResult) * 4);
	}

	return !isFailed;
}
