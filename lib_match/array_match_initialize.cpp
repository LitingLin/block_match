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

size_t getArrayCLength(int numberOfArrayA, int numberOfArrayB)
{
	return numberOfArrayA * numberOfArrayB;
}

size_t getPageLockedMemoryAllocationSize(int numberOfArrayA, int numberOfArrayB, int lengthOfArray)
{
	const int numberOfGpuDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGpuProcessorThread = globalContext.numberOfGPUProcessorThread;
	const int numberOfThreads = globalContext.numberOfThreads;

	return getArrayCLength(numberOfArrayA, numberOfArrayB) * sizeof(float) +
		arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * numberOfThreads;
}

LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArrayA, int numberOfArrayB, int lengthOfArray)
{
	if (!globalContext.hasGPU)
		return LibMatchErrorCode::errorCuda;

#ifndef NDEBUG
	const int numberOfThreads = 1;
#else
	const int numberOfThreads = 2; // enough
#endif

	LibMatchErrorCode errorCode;

	ArrayMatchContext *context = static_cast<ArrayMatchContext *>(malloc(sizeof(ArrayMatchContext) +
		sizeof(ArrayMatchExecutionContext) * numberOfThreads +
		sizeof(void*) * numberOfThreads));
	if (context == nullptr) {
		errorCode = LibMatchErrorCode::errorMemoryAllocation;

		setLastErrorString("Error in memory allocation.");

		goto ContextAllocationFailed;
	}

	context->executionContext = reinterpret_cast<ArrayMatchExecutionContext*>(reinterpret_cast<char*>(context) + sizeof(ArrayMatchContext));
	context->taskHandle = reinterpret_cast<void**>(reinterpret_cast<char*>(context->executionContext) + sizeof(ArrayMatchExecutionContext) * numberOfThreads);

	float *result;

	cudaError_t cudaError = cudaMallocHost(&result, getPageLockedMemoryAllocationSize(numberOfArrayA, numberOfArrayB, lengthOfArray));
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorCode::errorPageLockedMemoryAllocation;

		setCudaLastErrorString(cudaError, "Error in page locked memory allocation.\n");

		goto PageLockedMemoryAllocationFailed;
	}

	float *bufferA = result + getArrayCLength(numberOfArrayA, numberOfArrayB);

	float *deviceBufferA, *deviceBufferB, *deviceBufferC;

	const int numberOfGpuDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGpuProcessorThread = globalContext.numberOfGPUProcessorThread;

	size_t deviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * numberOfThreads;
	size_t deviceBufferBSize = deviceBufferASize;
	size_t deviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread) * numberOfThreads;

	cudaError = cudaMalloc(&deviceBufferA, (deviceBufferASize + deviceBufferBSize + deviceBufferCSize) * sizeof(float));
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorCode::errorGpuMemoryAllocation;

		setCudaLastErrorString(cudaError, "Error in gpu memory allocation.\n");

		goto GpuMemoryAllocationFailed;
	}

	deviceBufferB = deviceBufferA + deviceBufferASize;
	deviceBufferC = deviceBufferB + deviceBufferBSize;

	context->deviceBufferA = deviceBufferA;
	context->deviceBufferB = deviceBufferB;
	context->deviceBufferC = deviceBufferC;
	context->bufferA = bufferA;
	context->lengthOfArray = lengthOfArray;
	context->numberOfArrayA = numberOfArrayA;
	context->numberOfArrayB = numberOfArrayB;
	context->result = result;
	context->numberOfThreads = numberOfThreads;

	*instance = context;
	errorCode = LibMatchErrorCode::success;

	return errorCode;

GpuMemoryAllocationFailed:

PageLockedMemoryAllocationFailed:
	cudaFreeHost(result);
MemoryAllocationFailed:
	free(context);
ContextAllocationFailed:
	return errorCode;
}

size_t arrayMatchGetMaximumMemoryAllocationSize()
{
	return sizeof(ArrayMatchContext);
}

size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int lengthOfArray)
{
	return getGpuMemoryAllocationSize(lengthOfArray);
}

size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArrayA, int numberOfArrayB, int lengthOfArray)
{
	return getPageLockedMemoryAllocationSize(numberOfArrayA, numberOfArrayB, lengthOfArray);
}