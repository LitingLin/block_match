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

size_t getPageLockedMemoryAllocationSize(int numberOfArrayA, int numberOfArrayB, int lengthOfArray, int numberOfThreads)
{
	const int numberOfGpuDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGpuProcessorThread = globalContext.numberOfGPUProcessorThread;

	return getArrayCLength(numberOfArrayA, numberOfArrayB) * sizeof(float) +
		arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * sizeof(float) * numberOfThreads +
		arrayMatchPerThreadDeviceBufferBSize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * sizeof(float) * numberOfThreads;
}

template <typename Type>
LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArrayA, int numberOfArrayB, int lengthOfArray)
{
	if (!globalContext.hasGPU)
		return LibMatchErrorCode::errorCuda;

#ifndef NDEBUG
	const int numberOfThreads = 2;
#else
	int numberOfThreads = 2; // enough
#endif

	if (numberOfArrayA == 1)
		numberOfThreads = 1;

	LibMatchErrorCode errorCode;

	ArrayMatchContext<Type> *context = static_cast<ArrayMatchContext<Type> *>(malloc(sizeof(ArrayMatchContext<Type>) +
		sizeof(ArrayMatchExecutionContext<Type>) * numberOfThreads +
		sizeof(void*) * numberOfThreads));

	if (context == nullptr) {
		errorCode = LibMatchErrorCode::errorMemoryAllocation;

		setLastErrorString("Error in memory allocation.");

		goto ContextAllocationFailed;
	}

	context->executionContext = reinterpret_cast<ArrayMatchExecutionContext<Type>*>(reinterpret_cast<char*>(context) + sizeof(ArrayMatchContext<Type>));
	context->taskHandle = reinterpret_cast<void**>(reinterpret_cast<char*>(context->executionContext) + sizeof(ArrayMatchExecutionContext<Type>) * numberOfThreads);
	
	const int numberOfGpuDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGpuProcessorThread = globalContext.numberOfGPUProcessorThread;

	float *deviceBufferA, *deviceBufferB, *deviceBufferC;

	size_t deviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * numberOfThreads;
	size_t deviceBufferBSize = deviceBufferASize;
	size_t deviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread) * numberOfThreads;

	cudaError_t cudaError = cudaMalloc(&deviceBufferA, (deviceBufferASize + deviceBufferBSize + deviceBufferCSize) * sizeof(float));
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorCode::errorGpuMemoryAllocation;

		setCudaLastErrorString(cudaError, "Error in gpu memory allocation.\n");

		goto GpuMemoryAllocationFailed;
	}

	float *result;

	cudaError = cudaMallocHost(&result, getPageLockedMemoryAllocationSize(numberOfArrayA, numberOfArrayB, lengthOfArray, numberOfThreads));
	if (cudaError != cudaSuccess) {
		errorCode = LibMatchErrorCode::errorPageLockedMemoryAllocation;

		setCudaLastErrorString(cudaError, "Error in page locked memory allocation.\n");

		goto PageLockedMemoryAllocationFailed;
	}

	float *bufferA = result + getArrayCLength(numberOfArrayA, numberOfArrayB);
	float *bufferB = bufferA + arrayMatchPerThreadDeviceBufferASize(numberOfGpuDeviceMultiProcessor, numberOfGpuProcessorThread, lengthOfArray) * numberOfThreads;

	deviceBufferB = deviceBufferA + deviceBufferASize;
	deviceBufferC = deviceBufferB + deviceBufferBSize;

	context->deviceBufferA = deviceBufferA;
	context->deviceBufferB = deviceBufferB;
	context->deviceBufferC = deviceBufferC;
	context->bufferA = bufferA;
	context->bufferB = bufferB;
	context->lengthOfArray = lengthOfArray;
	context->numberOfArrayA = numberOfArrayA;
	context->numberOfArrayB = numberOfArrayB;
	context->result = result;
	context->numberOfThreads = numberOfThreads;

	*instance = context;
	errorCode = LibMatchErrorCode::success;

	return errorCode;


PageLockedMemoryAllocationFailed:
	cudaFreeHost(result);

GpuMemoryAllocationFailed:
	free(context);
ContextAllocationFailed:
	return errorCode;
}

size_t arrayMatchGetMaximumMemoryAllocationSize()
{
	return sizeof(ArrayMatchContext<Type>);
}

size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int lengthOfArray)
{
	return getGpuMemoryAllocationSize(lengthOfArray);
}

size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArrayA, int numberOfArrayB, int lengthOfArray, int numberOfThreads)
{
	return getPageLockedMemoryAllocationSize(numberOfArrayA, numberOfArrayB, lengthOfArray, numberOfThreads);
}