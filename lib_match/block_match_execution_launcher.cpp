#include "block_match_execute.hpp"

template <typename Type>
void blockMatchExecute(void *_instance, Type *A, Type *B,
	Type *C,
	Type *padded_A, Type *padded_B,
	int *index_x, int *index_y)
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type>*>(_instance);

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
		B_N_padPost = instance->matrixBPadding_N_post,
		numberOfThreads = instance->numberOfThreads;
	/*
	if (padded_A == nullptr)
	{
		if (instance->optionalBuffer.matrixA_padded_internal == nullptr)
			if (!allocateInternalBuffer(instance, InternalBufferType::MatrixA_Padded_Buffer))
			{
				setLastErrorString("Error: allocation failed");

				return false;
			}

		padded_A = instance->optionalBuffer.matrixA_padded_internal;
	}
	if (padded_B == nullptr)
	{
		if (instance->optionalBuffer.matrixB_padded_internal == nullptr)
			if (!allocateInternalBuffer(instance, InternalBufferType::MatrixA_Padded_Buffer))
			{
				setLastErrorString("Error: allocation failed");

				return false;
			}

		padded_B = instance->optionalBuffer.matrixB_padded_internal;
	}*/
	// TODO: Fix
	instance->padMethodA(A, padded_A, A_N, A_M, A_N_padPre, A_N_padPost, A_M_padPre, A_M_padPost);
	instance->padMethodB(B, padded_B, B_N, B_M, B_N_padPre, B_N_padPost, B_M_padPre, B_M_padPost);
	/*
	if (index_x == nullptr)
	{
		if (instance->optionalBuffer.index_x_internal == nullptr)
		{
			if (!allocateInternalBuffer(instance, InternalBufferType::Index_X_Internal))
			{
				setLastErrorString("Error: allocation failed");
				return false;
			}
		}

		index_x = instance->optionalBuffer.index_x_internal;
	}
	else
		initializeWorkerInternalBuffer(instance, index_x, InternalBufferType::Index_X_Internal);

	if (index_y == nullptr)
	{
		if (instance->optionalBuffer.index_y_internal == nullptr)
		{
			if (!allocateInternalBuffer(instance, InternalBufferType::Index_Y_Internal))
			{
				setLastErrorString("Error: allocation failed");
				return false;
			}
		}

		index_y = instance->optionalBuffer.index_y_internal;
	}
	else
		initializeWorkerInternalBuffer(instance, index_y, InternalBufferType::Index_Y_Internal);
		*/

	if (instance->perThreadBuffer[0].matrixA_buffer.get() == nullptr)
	{
		for (int indexOfThread = 0; indexOfThread != instance->numberOfThreads;++indexOfThread)
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

	execution_service &pool = globalContext.pool;

	for (int i = 0; i < numberOfThreads; ++i)
	{
		ExecutionContext<Type> *executionContext = instance->workerContext[i].executionContext.get();
		executionContext->block_M = instance->block_M;
		executionContext->block_N = instance->block_N;
		executionContext->matrixA_M = instance->matrixA_padded_M;
		executionContext->matrixA_N = instance->matrixA_padded_N;
		executionContext->matrixB_M = instance->matrixB_padded_M;
		executionContext->matrixB_N = instance->matrixB_padded_N;
		executionContext->matrixA = padded_A;
		executionContext->matrixB = padded_B;
		executionContext->numberOfIteration = instance->workerContext[i].numberOfIteration;
		executionContext->index_x = index_x + instance->workerContext[i].rawMatrixCIndex_begin * instance->C_dimensions[2];
		executionContext->index_y = index_y + instance->workerContext[i].rawMatrixCIndex_begin * instance->C_dimensions[2];
		executionContext->index_x_buffer = instance->perThreadBuffer[i].index_x_sorting_buffer.get();
		executionContext->index_y_buffer = instance->perThreadBuffer[i].index_y_sorting_buffer.get();
		executionContext->numberOfIndexRetain = instance->numberOfIndexRetain;
		executionContext->lengthOfGpuTaskQueue = instance->sizeOfGpuTaskQueue;
		executionContext->numberOfSubmitProcessors = instance->numberOfSubmitProcessors;
		executionContext->numberOfSubmitThreadsPerProcessor = instance->numberOfSubmitThreadsPerProcessor;
		executionContext->matrixA_buffer = instance->perThreadBuffer[i].matrixA_buffer.get();
		executionContext->matrixB_buffer = instance->perThreadBuffer[i].matrixB_buffer.get();
		executionContext->matrixC_buffer = instance->perThreadBuffer[i].matrixC_buffer.get();
		executionContext->matrixA_deviceBuffer = instance->perThreadBuffer[i].matrixA_deviceBuffer.get();
		executionContext->matrixB_deviceBuffer = instance->perThreadBuffer[i].matrixB_deviceBuffer.get();
		executionContext->matrixC_deviceBuffer = instance->perThreadBuffer[i].matrixC_deviceBuffer.get();
		executionContext->maxNumberOfThreadsPerProcessor = globalContext.numberOfGPUProcessorThread;
		executionContext->neighbour_M = instance->searchRegion_M;
		executionContext->neighbour_N = instance->searchRegion_N;
		executionContext->numberOfBlockBPerBlockA = instance->numberOfBlockBPerBlockA;
		executionContext->rawIndexBuffer = instance->perThreadBuffer[i].index_raw_sorting_buffer.get();
		executionContext->rawIndexTemplate = instance->common_buffer.get();
		executionContext->startIndexOfMatrixA_M = instance->workerContext[i].beginMatrixAIndex_M;
		executionContext->indexA_M_begin = instance->indexA_M_begin;
		executionContext->indexA_M_end = instance->indexA_M_end;
		executionContext->indexA_N_begin = instance->indexA_N_begin;
		executionContext->indexA_N_end = instance->indexA_N_end;
		executionContext->startIndexOfMatrixA_N = instance->workerContext[i].beginMatrixAIndex_N;
		executionContext->stream = instance->streams[i];
		executionContext->strideA_M = instance->strideA_M;
		executionContext->strideA_N = instance->strideA_N;
		executionContext->strideB_M = instance->strideB_M;
		executionContext->strideB_N = instance->strideB_N;
		executionContext->matrixC = C + instance->workerContext[i].rawMatrixCIndex_begin * instance->C_dimensions[2];

		instance->threadPoolTaskHandle[i] = pool.submit(reinterpret_cast<unsigned(*)(void*)>(instance->executionMethod),
			static_cast<void*>(executionContext));
	}

	for (int i = 0; i < numberOfThreads; ++i)
	{
		pool.join(instance->threadPoolTaskHandle[i]);
	}
	bool isFailed = false;

	std::string error_message;
	for (int i = 0; i < numberOfThreads; ++i)
	{
		if (pool.get_rc(instance->threadPoolTaskHandle[i]) != 0)
			isFailed = true;
		error_message += pool.get_exp_what(instance->threadPoolTaskHandle[i]);
		pool.release(instance->threadPoolTaskHandle[i]);
	}

	if (isFailed)
		throw std::runtime_error(error_message);
}

LIB_MATCH_EXPORT
template
void blockMatchExecute(void *_instance, float *A, float *B,
	float *C,
	float *padded_A, float *padded_B,
	int *index_x, int *index_y);

LIB_MATCH_EXPORT
template
void blockMatchExecute(void *_instance, double *A, double *B,
	double *C,
	double *padded_A, double *padded_B,
	int *index_x, int *index_y);