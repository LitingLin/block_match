#include "lib_match_internal.h"

#include "lib_match.h"

size_t getGpuMemoryAllocationSize(int lengthOfArray)
{
	const int numberOfGpuDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGpuProcessorThread = globalContext.numberOfGPUProcessorThread;
	const int numberOfThreads = globalContext.numberOfThreads;

	size_t deviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * numberOfThreads;
	size_t deviceBufferBSize = deviceBufferASize;
	size_t deviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread) * numberOfThreads;

	return (deviceBufferASize + deviceBufferBSize + deviceBufferCSize) * sizeof(float);
}

size_t getPageLockedMemoryAllocationSize(int numberOfArray)
{
	return numberOfArray * sizeof(float);
}

extern "C"
enum ErrorCode arrayMatchInitialize(void **instance,
	int numberOfArray, int lengthOfArray)
{
	if (!globalContext.hasGPU)
		return LibMatchErrorCuda;

	ErrorCode errorCode;

	ArrayMatchContext *context = (ArrayMatchContext *)malloc(sizeof(ArrayMatchContext));
	if (context == nullptr) {
		errorCode = LibMatchErrorMemoryAllocation;

		setLastErrorString("Error in memory allocation.");

		goto ContextAllocationFailed;
	}

	float *result;

	cudaError_t cudaError = cudaMallocHost(&result, getPageLockedMemoryAllocationSize(numberOfArray));
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorPageLockedMemoryAllocation;

		setCudaLastErrorString(cudaError, "Error in page locked memory allocation.\n");

		goto PageLockedMemoryAllocationFailed;
	}

	float *deviceBufferA, *deviceBufferB, *deviceBufferC;

	const int numberOfGpuDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGpuProcessorThread = globalContext.numberOfGPUProcessorThread;

#ifndef NDEBUG
	const int numberOfThreads = 1;
#else
	const int numberOfThreads = 2; // enough
#endif

	size_t deviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * numberOfThreads;
	size_t deviceBufferBSize = deviceBufferASize;
	size_t deviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread) * numberOfThreads;

	cudaError = cudaMalloc(&deviceBufferA, (deviceBufferASize + deviceBufferBSize + deviceBufferCSize) * sizeof(float));
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorGpuMemoryAllocation;

		setCudaLastErrorString(cudaError, "Error in gpu memory allocation.\n");

		goto GpuMemoryAllocationFailed;
	}

	deviceBufferB = deviceBufferA + deviceBufferASize;
	deviceBufferC = deviceBufferB + deviceBufferBSize;

	context->deviceBufferA = deviceBufferA;
	context->deviceBufferB = deviceBufferB;
	context->deviceBufferC = deviceBufferC;
	context->lengthOfArray = lengthOfArray;
	context->numberOfArray = numberOfArray;
	context->result = result;
	context->numberOfThreads = numberOfThreads;

	*instance = context;
	errorCode = LibMatchErrorOk;

	return errorCode;

GpuMemoryAllocationFailed:

PageLockedMemoryAllocationFailed:

MemoryAllocationFailed:
	free(context);
ContextAllocationFailed:
	return errorCode;
}

extern "C"
size_t arrayMatchGetMaximumMemoryAllocationSize(int numberOfArray, int lengthOfArray)
{
	return sizeof(ArrayMatchContext);
}

extern "C"
size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int numberOfArray, int lengthOfArray)
{
	return getGpuMemoryAllocationSize(lengthOfArray);
}

extern "C"
size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArray, int lengthOfArray)
{
	return getPageLockedMemoryAllocationSize(numberOfArray);
}