#pragma once
#include "lib_match_internal.h"
#include "sorting.hpp"
#include <cstring>

template <typename Type>
using ArrayMatchProcessFunction = cudaError_t(const Type *A, const  Type *B, const int numberOfArray,
	const int size, Type *C, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
using ProcessFunctionCPU = void(const Type *A, const Type *B, const int size, Type *C);

template <typename Type, ArrayMatchProcessFunction<Type> processFunction>
void submitGpuTask(Type *bufferA, Type *bufferB, Type *resultBuffer, Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
	const int numberOfArray, const int size,
	const int numberOfGpuProcessors, const int numberOfGpuThreads,
	const cudaStream_t stream)
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
	int numberOfArrayA, int numberOfArrayB, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		memcpy(index_sorting_buffer, index_template, numberOfArrayB * sizeof(*index_template));

		sortType(index_sorting_buffer, result_buffer, numberOfArrayB, retain);

		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(result_buffer[index_sorting_buffer[j]]);
			*c_index++ = static_cast<IndexDataType>(index_sorting_buffer[j]);
		}

		result_buffer += numberOfArrayB;
	}
	*index = c_index;
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType, RawSortMethod_WithIndex<ComputingDataType> sortType>
inline void
sort_recordIndexPlusOne(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		memcpy(index_sorting_buffer, index_template, numberOfArrayB * sizeof(*index_template));

		sortType(index_sorting_buffer, result_buffer, numberOfArrayB, retain);

		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(result_buffer[index_sorting_buffer[j]]);
			*c_index++ = static_cast<IndexDataType>(index_sorting_buffer[j] + 1);
		}

		result_buffer += numberOfArrayB;
	}
	*index = c_index;
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, RawSortMethod<ComputingDataType> sortType>
inline void
sort_noRecordIndex(void **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		sortType(result_buffer, numberOfArrayB, retain);

		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(result_buffer[j]);
		}
	}
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
inline void
noSort_recordIndex(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(result_buffer[j]);
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
			*c_result++ = static_cast<ResultDataType>(result_buffer[j]);
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
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArray; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(result_buffer[j]);
		}
	}
	*result = c_result;
}

#ifdef _MSC_VER
#pragma warning( default : 4800 )  
#endif


template <typename Type, ArrayMatchProcessFunction<Type> processFunction>
unsigned arrayMatchWorker(ArrayMatchExecutionContext<Type>* context)
{
	void *A = context->A, *B = context->B, *C = context->C;
	Type *bufferA = context->bufferA, *bufferB = context->bufferB, *bufferC = context->bufferC,
		*deviceBufferA = context->deviceBufferA, *deviceBufferB = context->deviceBufferB, *deviceBufferC = context->deviceBufferC;
	const int numberOfArrayA = context->numberOfArrayA, numberOfArrayB = context->numberOfArrayB, sizeOfArray = context->sizeOfArray,
		startIndexA = context->startIndexA, startIndexB = context->startIndexB, numberOfIteration = context->numberOfIteration,
		numberOfGPUDeviceMultiProcessor = context->numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread = context->numberOfGPUProcessorThread;

	void *index = context->index;
	int *index_template = context->index_template;
	int *index_sorting_buffer = context->index_sorting_buffer;

	cudaStream_t stream = context->stream;

	const int sizeOfGpuTaskQueue = numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread;

	ArrayCopyMethod *arrayCopyingA = context->arrayCopyingAFunction;
	ArrayCopyMethod *arrayCopyingB = context->arrayCopyingBFunction;
	ArrayMatchDataPostProcessingMethod *dataPostProcessing = context->dataPostProcessingFunction;

	const int elementSizeOfTypeA = context->elementSizeOfTypeA, elementSizeOfTypeB = context->elementSizeOfTypeB,
		elementSizeOfTypeC = context->elementSizeOfTypeC, elementSizeOfIndex = context->elementSizeOfIndex;

	char *c_A = static_cast<char*>(A) + startIndexA * elementSizeOfTypeA;
	char *c_B = static_cast<char*>(B) + startIndexB * elementSizeOfTypeB;
	void *c_C = static_cast<char*>(C) + (startIndexA * numberOfArrayB + startIndexB) * elementSizeOfTypeC;

	int retain = context->retain;
	if (retain == 0)
		retain = numberOfArrayB;

	void *c_index = static_cast<char*>(index) + (startIndexA * numberOfArrayB + startIndexB) * elementSizeOfIndex;

	Type *c_bufferA = bufferA;
	Type *c_bufferB = bufferB;
	
	int indexOfIteration = 0;
	int numberOfFilledTaskQueue = 0;

	int indexOfA = startIndexA, indexOfB = startIndexB;

	goto JumpIn;

	for (/*indexOfA = 0*/; indexOfA < numberOfArrayA; ++indexOfA)
	{
		for (indexOfB = 0; indexOfB < numberOfArrayB; ++indexOfB)
		{
			JumpIn:
			arrayCopyingA(c_bufferA, c_A, sizeOfArray);
			c_bufferA += sizeOfArray;
			arrayCopyingB(c_bufferB, c_B, sizeOfArray);
			c_bufferB += sizeOfArray;
			c_B += elementSizeOfTypeB * sizeOfArray;

			numberOfFilledTaskQueue++;
			if (numberOfFilledTaskQueue == sizeOfGpuTaskQueue)
			{
				submitGpuTask<Type, processFunction>(bufferA, bufferB, bufferC,
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

			++indexOfIteration;

			if (indexOfIteration == numberOfIteration)
				goto JumpOut;
		}
		c_B = static_cast<char*>(B);
		c_A += elementSizeOfTypeA * sizeOfArray;
	}
	JumpOut:

	if (numberOfFilledTaskQueue)
	{
		submitGpuTask<Type, processFunction>(bufferA, bufferB, bufferC,
			deviceBufferA, deviceBufferB, deviceBufferC,
			numberOfFilledTaskQueue, sizeOfArray,
			numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, stream);

		dataPostProcessing(&c_index, &c_C, bufferC,
			numberOfFilledTaskQueue, sizeOfArray, retain,
			index_template, index_sorting_buffer);
	}
	return 0;
}
