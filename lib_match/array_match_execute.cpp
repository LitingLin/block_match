#include "lib_match_internal.h"
#include "sorting.hpp"

template <typename Type>
using ProcessFunction = cudaError_t(*)(Type *A, Type *B, int numberOfArray,
	int size, Type *C, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
using ProcessFunctionCPU = void(*)(Type *A, Type *B, int size, Type *C);

template <typename Type, ProcessFunction<Type> processFunction>
void submitGpuTask(Type *bufferA, Type *bufferB, Type *resultBuffer, Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
	int numberOfArray, int size,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferA, bufferA, numberOfArray * size * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferB, bufferB, numberOfArray * size * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(processFunction(deviceBufferA, deviceBufferB, numberOfArray, size, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfArray * sizeof(Type), cudaMemcpyDeviceToHost, stream));
}


#ifdef _MSC_VER
#pragma warning( disable : 4800 )  
#endif
template <typename ComputingDataType, typename ResultDataType, typename IndexDataType, RawSortMethod_WithIndex<ComputingDataType> sortType>
inline void
sort_recordIndex(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArray, int sizeOfArray, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		memcpy(index_sorting_buffer, index_template, sizeOfArray * sizeof(*index_template));

		sortType(index_sorting_buffer, result_buffer, sizeOfArray, retain);

		for (int j = 0; j < retain; ++j)
		{
			*result++ = static_cast<ResultDataType>(result_buffer[index_sorting_buffer[j]]);
			*index++ = static_cast<IndexDataType>(index_sorting_buffer[j]);
		}

		result_buffer += sizeOfArray;
	}
	*index = c_index;
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType, RawSortMethod_WithIndex<ComputingDataType> sortType>
inline void
sort_recordIndexPlusOne(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArray, int sizeOfArray, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		memcpy(index_sorting_buffer, index_template, sizeOfArray * sizeof(*index_template));

		sortType(index_sorting_buffer, result_buffer, sizeOfArray, retain);

		for (int j = 0; j < retain; ++j)
		{
			*result++ = static_cast<ResultDataType>(result_buffer[index_sorting_buffer[j]]);
			*index++ = static_cast<IndexDataType>(index_sorting_buffer[j] + 1);
		}

		result_buffer += sizeOfArray;
	}
	*index = c_index;
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, RawSortMethod<ComputingDataType> sortType>
inline void
sort_noRecordIndex(void **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArray, int sizeOfArray, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		sortType(result_buffer, sizeOfArray, retain);

		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(*result_buffer++);
		}
	}
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
inline void
noSort_recordIndex(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArray, int sizeOfArray, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(*result_buffer++);
			*c_index++ = static_cast<IndexDataType>(j);
		}
	}
	*result = c_result;
	*index = c_index;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
inline void
noSort_recordIndexPlusOne(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArray, int sizeOfArray, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(*result_buffer++);
			*c_index++ = static_cast<IndexDataType>(j + 1);
		}
	}
	*result = c_result;
	*index = c_index;
}

template <typename ComputingDataType, typename ResultDataType>
inline void
noSort_noRecordIndex(void **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArray, int sizeOfArray, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	ResultDataType *c_result = result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(*result_buffer++);
		}
	}
	*result = c_result;
}

#ifdef _MSC_VER
#pragma warning( default : 4800 )  
#endif


template <typename Type, ProcessFunction<Type> processFunction>
unsigned arrayMatchWorker(ArrayMatchExecutionContext<Type>* context)
{
	void *A = context->A, *B = context->B, *C = context->C;
	Type *bufferA = context->bufferA, *bufferB = context->bufferB, *bufferC = context->bufferC,
		*deviceBufferA = context->deviceBufferA, *deviceBufferB = context->deviceBufferB, *deviceBufferC = context->deviceBufferC;
	int	numberOfArrayA = context->numberOfArrayA, numberOfArrayB = context->numberOfArrayB, sizeOfArray = context->sizeOfArray,
		startIndexA = context->startIndexA, numberOfIteration = context->numberOfIteration,
		numberOfGPUDeviceMultiProcessor = context->numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread = context->numberOfGPUProcessorThread;

	void *index = context->index;
	int *index_template = context->index_template;
	int *index_sorting_buffer = context->index_sorting_buffer;

	cudaStream_t stream = context->stream;

	int sizeOfGpuTaskQueue = numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread;
	int indexOfGpuTaskQueue = 0;

	ArrayCopyMethod *arrayCopyingA = context->arrayCopyingAFunction;
	ArrayCopyMethod *arrayCopyingB = context->arrayCopyingBFunction;
	ArrayMatchDataPostProcessingMethod *dataPostProcessing = context->dataPostProcessingFunction;

	char *c_A = static_cast<char*>(A);
	char *c_B = static_cast<char*>(B);
	void *c_C = C;
	int elementSizeOfTypeA = context->elementSizeOfTypeA, elementSizeOfTypeB = context->elementSizeOfTypeB;

	int retain = context->retain;
	if (retain == 0)
		retain = sizeOfArray;

	void *c_index = index;

	Type *c_bufferA = bufferA;
	Type *c_bufferB = bufferB;
	
	int indexOfIteration = 0;
	int numberOfFilledTaskQueue = 0;

	for (int indexOfA = startIndexA; indexOfA < numberOfArrayA; ++indexOfA)
	{
		for (int indexOfB = 0; indexOfB < numberOfArrayB; ++indexOfB)
		{
			arrayCopyingA(c_bufferA, c_A, sizeOfArray);
			c_bufferA += sizeOfArray;
			arrayCopyingB(c_bufferB, c_B, sizeOfArray);
			c_bufferB += sizeOfArray;
			c_B += elementSizeOfTypeB * sizeOfArray;

			numberOfFilledTaskQueue++;
			if (numberOfFilledTaskQueue == sizeOfGpuTaskQueue)
			{
				submitGpuTask<processFunction>(bufferA, bufferB, bufferC,
					deviceBufferA, deviceBufferB, deviceBufferC,
					numberOfFilledTaskQueue, sizeOfArray,
					numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, stream);

				dataPostProcessing(&c_index, &c_C, bufferC,
					numberOfFilledTaskQueue, sizeOfArray, retain,
					index_template, index_sorting_buffer);
				
				numberOfFilledTaskQueue = 0;
				c_bufferA = bufferA;
				c_bufferB = bufferB;
			}

			if (indexOfIteration == numberOfIteration)
				goto JumpOut;
		}
		c_B = static_cast<char*>(B);
		c_A += elementSizeOfTypeA * sizeOfArray;
	}
	JumpOut:

	if (numberOfFilledTaskQueue)
	{
		submitGpuTask<processFunction>(bufferA, bufferB, bufferC,
			deviceBufferA, deviceBufferB, deviceBufferC,
			numberOfFilledTaskQueue, sizeOfArray,
			numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, stream);

		dataPostProcessing(&c_index, &c_C, bufferC,
			numberOfFilledTaskQueue, sizeOfArray, retain,
			index_template, index_sorting_buffer);
	}
	return 0;
}
