#include "block_match_internal.h"

#include "block_match.h"

#include "stack_vector.hpp"

typedef cudaError_t(ProcessFunction)(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);

template <ProcessFunction processFunction>
cudaError_t submitGpuTask(float *A, float *B, float *C,
	float *deviceBufferA, float *deviceBufferB, float *deviceBufferC,
	int numberOfArray, int lengthOfArray,
	int numberOfGPUDeviceMultiProcessor, int numberOfGPUProcessorThread)
{
	cudaError_t cudaError;
	size_t dataSizeA, dataSizeB;
	dataSizeA = dataSizeB = numberOfArray * lengthOfArray * sizeof(float);
	size_t dataSizeC = numberOfArray * sizeof(float);
	cudaError = cudaMemcpyAsync(deviceBufferA, A, dataSizeA, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
		return cudaError;
	cudaError = cudaMemcpyAsync(deviceBufferB, B, dataSizeB, cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = processFunction(deviceBufferA, deviceBufferB, deviceBufferC,
		lengthOfArray, numberOfArray,
		numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);
	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = cudaMemcpy(C,deviceBufferC ,dataSizeC, cudaMemcpyDeviceToHost);
	return cudaError;
}

template <ProcessFunction processFunction>
unsigned arrayMatchWorker(float *A, float *B, float *C,
	float *deviceBufferA, float *deviceBufferB, float *deviceBufferC,
	int numberOfArray, int lengthOfArray,
	int numberOfGPUDeviceMultiProcessor, int numberOfGPUProcessorThread)
{
	int perIterationNumberOfArray = numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread; 
	int indexOfArray = 0;
	
	cudaError_t cudaError;

	for (; indexOfArray < numberOfArray; indexOfArray += perIterationNumberOfArray)
	{
		if (indexOfArray + perIterationNumberOfArray >= numberOfArray) {
			perIterationNumberOfArray = numberOfArray - indexOfArray;
			numberOfGPUDeviceMultiProcessor = (perIterationNumberOfArray + numberOfGPUProcessorThread - 1) / numberOfGPUProcessorThread;
		}

		cudaError = submitGpuTask<processFunction>(A, B, C, 
			deviceBufferA, deviceBufferB, deviceBufferC,
			perIterationNumberOfArray, lengthOfArray,
			numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

		if (cudaError != cudaSuccess)
			return 1;
		
		size_t offsetA, offsetB;
		offsetA = offsetB = perIterationNumberOfArray * lengthOfArray;
		size_t offsetC = perIterationNumberOfArray;

		A += offsetA;
		B += offsetB;
		C += offsetC;
	}
	return 0;
}

extern "C"
enum ErrorCode arrayMatchExecute(void *instance, float *A, float *B, enum Method method,
	float **_result)
{
	ErrorCode errorCode = LibMatchErrorOk;
	ArrayMatchContext *context = (ArrayMatchContext *)instance;

	int numberOfArray = context->numberOfArray;
	int lengthOfArray = context->lengthOfArray;
	float *result = context->result;

	float *deviceBufferA = context->deviceBufferA;
	float *deviceBufferB = context->deviceBufferB;
	float *deviceBufferC = context->deviceBufferC;

	int numberOfThreads = context->numberOfThreads;

	if (numberOfArray == 1)
		numberOfThreads = 1;

	ThreadPool &pool = globalContext.pool;

	StackVector<std::tuple<float*, float*, float*,
		float*, float*, float*,
		int, int,
		int, int>,
		2> parameterBuffer(numberOfThreads);

	if (parameterBuffer.bad_alloc()) {
		setLastErrorString("Error: in allocate memory for parameterBuffer");
		return LibMatchErrorMemoryAllocation;
	}

	StackVector<void *, 2>taskHandle(numberOfThreads);
	if (taskHandle.bad_alloc()) {
		setLastErrorString("Error: in allocate memory for taskHandle");
		return LibMatchErrorMemoryAllocation;
	}

	int perThreadNumberOfArray = numberOfArray / numberOfThreads;

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	size_t perThreadDeviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, lengthOfArray);
	size_t perThreadDeviceBufferBSize = perThreadDeviceBufferASize;
	size_t perThreadDeviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		int c_numberOfArray;
		if (indexOfThread + 1 != numberOfThreads)
			c_numberOfArray = perThreadNumberOfArray;
		else
			c_numberOfArray = numberOfArray - indexOfThread * perThreadNumberOfArray;

		parameterBuffer[indexOfThread] = { A + perThreadNumberOfArray * indexOfThread * lengthOfArray,
		B + perThreadNumberOfArray * indexOfThread * lengthOfArray,
		result + perThreadNumberOfArray * indexOfThread,
		deviceBufferA + perThreadDeviceBufferASize * indexOfThread,
		deviceBufferB + indexOfThread * perThreadDeviceBufferBSize,
		deviceBufferC + indexOfThread * perThreadDeviceBufferCSize,
		c_numberOfArray, lengthOfArray,
		numberOfGPUDeviceMultiProcessor,numberOfGPUProcessorThread };

		if (method == MSE)
			taskHandle[indexOfThread] =
			thread_pool_launcher(pool, (arrayMatchWorker<arrayMatchMse>), parameterBuffer[indexOfThread]);
		else if (method == CC)
			taskHandle[indexOfThread] =
			thread_pool_launcher(pool, (arrayMatchWorker<arrayMatchCc>), parameterBuffer[indexOfThread]);
		else
		{
			setLastErrorString("Measure Method hasn't implement yet");
			return LibMatchErrorInternal;
		}
	}

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		pool.join(taskHandle[indexOfThread]);
	}

	for (int i = 0; i < numberOfThreads; ++i)
	{
		if (pool.get_rc(taskHandle[i]) != 0) {
			setLastErrorString("Internal CUDA error");
			errorCode = LibMatchErrorCuda;
		}
		pool.release(taskHandle[i]);
	}

	if (errorCode == LibMatchErrorOk)
	{
		*_result = result;
	}

	return errorCode;
}