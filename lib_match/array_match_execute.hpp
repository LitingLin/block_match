#pragma once
#include "lib_match_internal.h"
#include "lib_match_execute.hpp"
#include <cstring>

#ifdef _MSC_VER
#pragma warning( disable : 4800 )  
#endif
template <typename ComputingDataType, typename ResultDataType, typename IndexDataType, 
RawSortMethod_WithIndex<ComputingDataType> sortType, ThresholdMethod<ComputingDataType> thresholdFunction,
IndexValueOffsetMethod<IndexDataType> indexValueOffset >
inline void
sort_recordIndex(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	ComputingDataType threshold, ComputingDataType replacementValue,
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
			ComputingDataType value = result_buffer[index_sorting_buffer[j]];
			thresholdFunction(&value, threshold, replacementValue);
			*c_result++ = static_cast<ResultDataType>(value);
			*c_index = static_cast<IndexDataType>(index_sorting_buffer[j]);
			indexValueOffset(c_index++);
		}

		result_buffer += numberOfArrayB;
	}
	*index = c_index;
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType,
RawSortMethod<ComputingDataType> sortType, ThresholdMethod<ComputingDataType> thresholdFunction>
inline void
sort_noRecordIndex(void **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	ComputingDataType threshold, ComputingDataType replacementValue,
	const int *index_template, int *index_sorting_buffer)
{
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		sortType(result_buffer, numberOfArrayB, retain);

		for (int j = 0; j < retain; ++j)
		{
			ComputingDataType value = result_buffer[j];
			thresholdFunction(&value, threshold, replacementValue);
			*c_result++ = static_cast<ResultDataType>(value);
		}
		result_buffer += numberOfArrayB;
	}
	*result = c_result;
}

template <typename ComputingDataType, typename ResultDataType, typename IndexDataType,
	ThresholdMethod<ComputingDataType> thresholdFunction,
	IndexValueOffsetMethod<IndexDataType> indexValueOffset>
inline void
noSort_recordIndex(IndexDataType **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	ComputingDataType threshold, ComputingDataType replacementValue,
	const int *index_template, int *index_sorting_buffer)
{
	IndexDataType *c_index = *index;
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			ComputingDataType value = result_buffer[j];
			thresholdFunction(&value, threshold, replacementValue);
			*c_result++ = static_cast<ResultDataType>(value);
			*c_index = static_cast<IndexDataType>(j);
			indexValueOffset(c_index++);
		}
		result_buffer += numberOfArrayB;
	}
	*result = c_result;
	*index = c_index;
}

template <typename ComputingDataType, typename ResultDataType,
	ThresholdMethod<ComputingDataType> thresholdFunction>
inline void
noSort_noRecordIndex(void **index, ResultDataType **result,
	ComputingDataType *result_buffer,
	int numberOfArrayA, int numberOfArrayB, int retain,
	ComputingDataType threshold, ComputingDataType replacementValue,
	const int *index_template, int *index_sorting_buffer)
{
	ResultDataType *c_result = *result;
	for (int i = 0; i < numberOfArrayA; ++i)
	{
		for (int j = 0; j < retain; ++j)
		{
			ComputingDataType value = result_buffer[j];
			thresholdFunction(&value, threshold, replacementValue);
			*c_result++ = static_cast<ResultDataType>(value);
		}
		result_buffer += numberOfArrayB;
	}
	*result = c_result;
}

#ifdef _MSC_VER
#pragma warning( default : 4800 )  
#endif


template <typename Type, ArrayMatchProcessFunction<Type> processFunction>
unsigned arrayMatchWorker(ArrayMatchExecutionContext<Type>* context)
{
	CUDA_CHECK_POINT(cudaSetDevice(context->indexOfDevice));

	void *A = context->A, *B = context->B, *C = context->C;
	Type *bufferA = context->bufferA, *bufferB = context->bufferB, *bufferC = context->bufferC,
		*deviceBufferA = context->deviceBufferA, *deviceBufferB = context->deviceBufferB, *deviceBufferC = context->deviceBufferC;

	Type threshold = context->threshold;
	Type replacementValue = context->replacementValue;

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
	ArrayMatchDataPostProcessingMethod<Type> *dataPostProcessing = context->dataPostProcessingFunction;

	const int elementSizeOfTypeA = context->elementSizeOfTypeA, elementSizeOfTypeB = context->elementSizeOfTypeB,
		elementSizeOfTypeC = context->elementSizeOfTypeC, elementSizeOfIndex = context->elementSizeOfIndex;

	int retain = context->retain;
	if (retain == 0)
		retain = numberOfArrayB;

	char *c_A = static_cast<char*>(A) + startIndexA * sizeOfArray * elementSizeOfTypeA;
	char *c_B = static_cast<char*>(B);
	void *c_C = static_cast<char*>(C) + startIndexA * retain * elementSizeOfTypeC;

	void *c_index = static_cast<char*>(index) + startIndexA * retain * elementSizeOfIndex;

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
				threshold, replacementValue,
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
			threshold, replacementValue,
			index_template, index_sorting_buffer);
	}
	return 0;
}
