#pragma once
#include "lib_match_internal.h"
#include "lib_match_execute.hpp"
#include <cstring>

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
		result_buffer += numberOfArrayB;
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
		result_buffer += numberOfArrayB;
	}
	*result = c_result;
	*index = c_index;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType>
inline void
noSort_recordIndexPlusOne(IndexDataType **index, ResultDataType **result,
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
			*c_index++ = static_cast<IndexDataType>(j + 1);
		}
		result_buffer += numberOfArrayB;
	}
	*result = c_result;
	*index = c_index;
}

template <typename ComputingDataType, typename ResultDataType>
inline void
noSort_noRecordIndex(void **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	const int *index_template, int *index_sorting_buffer)
{
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			*c_result++ = static_cast<ResultDataType>(result_buffer[j]);
		}
		result_buffer += numberOfArrayB;
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
	const int numberOfArrayA = context->numberOfArrayA, numberOfArrayB = context->numberOfArrayB, sizeOfArray = context->sizeOfArray,
		startIndexA = context->startIndexA, numberOfIteration = context->numberOfIteration,
		numberOfGPUDeviceMultiProcessor = context->numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread = context->numberOfGPUProcessorThread;

	void *index = context->index;
	int *index_template = context->index_template;
	int *index_sorting_buffer = context->index_sorting_buffer;

	cudaStream_t stream = context->stream;

	const int sizeOfGpuTaskQueue = context->sizeOfGpuTaskQueue;

	ArrayCopyMethod *arrayCopyingA = context->arrayCopyingAFunction;
	ArrayCopyMethod *arrayCopyingB = context->arrayCopyingBFunction;
	ArrayMatchDataPostProcessingMethod *dataPostProcessing = context->dataPostProcessingFunction;

	const int elementSizeOfTypeA = context->elementSizeOfTypeA, elementSizeOfTypeB = context->elementSizeOfTypeB,
		elementSizeOfTypeC = context->elementSizeOfTypeC, elementSizeOfIndex = context->elementSizeOfIndex;

	char *c_A = static_cast<char*>(A) + startIndexA * elementSizeOfTypeA;
	char *c_B = static_cast<char*>(B);
	void *c_C = static_cast<char*>(C) + startIndexA * numberOfArrayB * elementSizeOfTypeC;

	int retain = context->retain;
	if (retain == 0)
		retain = numberOfArrayB;

	void *c_index = static_cast<char*>(index) + startIndexA * numberOfArrayB * elementSizeOfIndex;

	Type *c_bufferA = bufferA;
	Type *c_bufferB = bufferB;
	
	int indexOfIteration = 0;
	int numberOfAInQueue = 0;
/*
	for (int indexOfB = 0; indexOfB < numberOfArrayB; ++indexOfB)
	{
		arrayCopyingB(c_bufferB, c_B, sizeOfArray);
		c_bufferB += sizeOfArray;
		c_B += elementSizeOfTypeB * sizeOfArray;
	}
	c_B = static_cast<char*>(B);
	c_bufferB = bufferB;*/
	arrayCopyingB(bufferB, B, sizeOfArray * numberOfArrayB);

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferB, bufferB, numberOfArrayB * sizeOfArray * sizeof(Type), cudaMemcpyHostToDevice, stream));

	for (int indexOfA = startIndexA; indexOfA < numberOfArrayA || outOfIndexError(); ++indexOfA)
	{
		arrayCopyingA(c_bufferA, c_A, sizeOfArray);
		c_bufferA += sizeOfArray;
		++numberOfAInQueue;
		if (numberOfAInQueue == sizeOfGpuTaskQueue)
		{
			submitGpuTask_global<Type, processFunction>(bufferA, bufferC,
				deviceBufferA, deviceBufferB, deviceBufferC,
				sizeOfArray, numberOfAInQueue, numberOfArrayB,
				numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, stream);

			CUDA_CHECK_POINT(cudaStreamSynchronize(stream));

			dataPostProcessing(&c_index, &c_C, bufferC,
				numberOfAInQueue, numberOfArrayB, retain,
				index_template, index_sorting_buffer);

			numberOfAInQueue = 0;
			c_bufferA = bufferA;
		}

		++indexOfIteration;

		if (indexOfIteration == numberOfIteration)
			break;
		c_A += elementSizeOfTypeA * sizeOfArray;
	}

	if (numberOfAInQueue)
	{
		submitGpuTask_global<Type, processFunction>(bufferA, bufferC,
			deviceBufferA, deviceBufferB, deviceBufferC,
			sizeOfArray, numberOfAInQueue, numberOfArrayB,
			numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, stream);

		CUDA_CHECK_POINT(cudaStreamSynchronize(stream));

		dataPostProcessing(&c_index, &c_C, bufferC,
			numberOfAInQueue, numberOfArrayB, retain,
			index_template, index_sorting_buffer);
	}
	return 0;
}
