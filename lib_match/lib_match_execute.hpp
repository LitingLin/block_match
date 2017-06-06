#pragma once

#include <cuda_runtime.h>

template <typename Type>
using ArrayMatchProcessFunction = cudaError_t(const Type *A, const Type *B, const int numberOfA,
	const int numberOfBPerA, const int size, Type *C, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
using ProcessFunction = cudaError_t(const Type *A, const Type *B, const int *offsetA,
	const int numberOfB, const int size, Type *C, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
using ProcessFunctionCPU = void(const Type *A, const Type *B, const int size, Type *C);
//
//template <typename Type, ProcessFunction<Type> processFunction>
//void submitGpuTask(Type *bufferA, Type *bufferB, Type *resultBuffer, Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
//	int size,
//	int numberOfA, int numberOfBPerA,
//	int numberOfGpuProcessors, int numberOfGpuThreads,
//	cudaStream_t stream)
//{
//	int numberOfB = numberOfA * numberOfBPerA;
//
//	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferA, bufferA, numberOfA * size * sizeof(Type), cudaMemcpyHostToDevice, stream));
//
//	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferB, bufferB, numberOfB * size * sizeof(Type), cudaMemcpyHostToDevice, stream));
//
//	CUDA_CHECK_POINT(processFunction(deviceBufferA, deviceBufferB, numberOfA, numberOfBPerA, size, deviceResultBuffer,
//		numberOfGpuProcessors, numberOfGpuThreads, stream));
//
//	CUDA_CHECK_POINT(cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfB * sizeof(Type), cudaMemcpyDeviceToHost, stream));
//}

template <typename Type, ProcessFunction<Type> processFunction>
void submitGpuTask_offset(Type *bufferA, Type *bufferB, Type *resultBuffer,
	Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
	int size, int numberOfA, int numberOfB,
	int *offsetA, int *offsetADevice,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferA, bufferA, numberOfA * size * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferB, bufferB, numberOfB * size * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(offsetADevice, offsetA, numberOfB * size * sizeof(int), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(processFunction(deviceBufferA, deviceBufferB, offsetADevice, numberOfB, size, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfB * sizeof(Type), cudaMemcpyDeviceToHost, stream));
}

template <typename Type, ArrayMatchProcessFunction<Type> processFunction>
void submitGpuTask_global(Type *bufferA, Type *resultBuffer, Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
	int size,
	int numberOfA, int numberOfB,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferA, bufferA, numberOfA * size * sizeof(Type), cudaMemcpyHostToDevice, stream));
	
	CUDA_CHECK_POINT(processFunction(deviceBufferA, deviceBufferB, numberOfA, numberOfB, size, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfA * numberOfB * sizeof(Type), cudaMemcpyDeviceToHost, stream));
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

template<typename ComputingDataType>
using ThresholdMethod =
void(ComputingDataType*, ComputingDataType, ComputingDataType);

template <typename ComputingDataType>
void threshold(ComputingDataType *value, ComputingDataType threshold, ComputingDataType replacementValue)
{
	if (*value > threshold)
		*value = replacementValue;
}

template <typename ComputingDataType>
void noThreshold(ComputingDataType *value, ComputingDataType threshold, ComputingDataType replacementValue)
{
}

template<typename IndexDataType>
using IndexValueOffsetMethod =
void(IndexDataType*);

template<typename IndexDataType>
void indexValuePlusOne(IndexDataType* value)
{
	++(*value);
}

template<typename IndexDataType>
void noChangeIndexValue(IndexDataType* value)
{
}

inline
bool outOfIndexError()
{
#ifndef NDEBUG
	logger.critical("Internal logical error: indexA_M out of index");
#endif
	return false;
}