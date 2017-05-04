#pragma once
#include <cuda_runtime.h>

template <typename Type>
using ProcessFunction = cudaError_t(*)(Type *A, Type *B, int numberOfA,
	int numberOfBPerA, int size, Type *C, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
using ProcessFunctionCPU = void(*)(Type *A, Type *B, int size, Type *C);

template <typename Type, ProcessFunction<Type> processFunction>
void submitGpuTask(Type *bufferA, Type *bufferB, Type *resultBuffer, Type *deviceBufferA, Type *deviceBufferB, Type *deviceResultBuffer,
	int size,
	int numberOfA, int numberOfBPerA,
	int numberOfGpuProcessors, int numberOfGpuThreads,
	cudaStream_t stream)
{
	int numberOfB = numberOfA * numberOfBPerA;

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferA, bufferA, numberOfA * size * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(deviceBufferB, bufferB, numberOfB * size * sizeof(Type), cudaMemcpyHostToDevice, stream));

	CUDA_CHECK_POINT(processFunction(deviceBufferA, deviceBufferB, numberOfA, numberOfBPerA, size, deviceResultBuffer,
		numberOfGpuProcessors, numberOfGpuThreads, stream));

	CUDA_CHECK_POINT(cudaMemcpyAsync(resultBuffer, deviceResultBuffer, numberOfB * sizeof(Type), cudaMemcpyDeviceToHost, stream));
}