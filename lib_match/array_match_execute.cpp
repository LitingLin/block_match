#include "lib_match_internal.h"

#include "lib_match.h"

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

	cudaError = cudaMemcpy(C, deviceBufferC, dataSizeC, cudaMemcpyDeviceToHost);
	return cudaError;
}

template <ProcessFunction processFunction>
unsigned arrayMatchWorker(ArrayMatchExecutionContext* context)
{
	float *A = context->A, *B = context->B, *C = context->C,
		*bufferA = context->bufferA,
		*deviceBufferA = context->deviceBufferA, *deviceBufferB = context->deviceBufferB, *deviceBufferC = context->deviceBufferC;
	int	numberOfArray = context->numberOfArray, lengthOfArray = context->lengthOfArray,
		startIndexA = context->startIndexA, numberOfIteration = context->numberOfIteration,
		numberOfGPUDeviceMultiProcessor = context->numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread = context->numberOfGPUProcessorThread;

	int sizeOfGpuTaskQueue = numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread;
	int indexOfGpuTaskQueue = 0;

	cudaError_t cudaError;

	float *c_A = A + startIndexA * lengthOfArray;
	float *c_B = B;
	float *c_C = C + startIndexA * lengthOfArray;
	float *c_deviceBufferA = deviceBufferA, *c_deviceBufferB = deviceBufferB, *c_deviceBufferC = deviceBufferC;
	float *c_bufferA = bufferA;

	int sizeOfThisIteration = 0;
	int indexOfArray = 0;
	int sizeOfFilledGpuTaskQueue = 0;

	for (int indexOfIteration = 0; indexOfIteration < numberOfIteration; indexOfIteration += sizeOfThisIteration)
	{
		if (sizeOfFilledGpuTaskQueue + numberOfArray - indexOfArray <= sizeOfGpuTaskQueue)
			sizeOfThisIteration = numberOfArray - indexOfArray;
		else
			sizeOfThisIteration = sizeOfGpuTaskQueue - sizeOfFilledGpuTaskQueue;

		for (int indexOfA = 0; indexOfA < sizeOfThisIteration; ++indexOfA)
		{
			memcpy(c_bufferA, c_A, lengthOfArray * sizeof(float));
			c_bufferA += lengthOfArray;
		}

		cudaError = cudaMemcpyAsync(c_deviceBufferB, c_B, sizeOfThisIteration * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaError != cudaSuccess)
			return cudaError;

		c_deviceBufferB += sizeOfThisIteration;
		c_B += sizeOfThisIteration;

		indexOfArray += sizeOfThisIteration;
		sizeOfFilledGpuTaskQueue += sizeOfThisIteration;
		if (indexOfArray == lengthOfArray)
		{
			c_A += lengthOfArray;
			c_B = B;
		}

		if (sizeOfFilledGpuTaskQueue == sizeOfGpuTaskQueue)
		{
			cudaError = cudaMemcpyAsync(deviceBufferA, bufferA, sizeOfGpuTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaError != cudaSuccess)
				return cudaError;
			c_bufferA = bufferA;

			cudaError = processFunction(deviceBufferA, deviceBufferB, deviceBufferC,
				lengthOfArray, sizeOfGpuTaskQueue,
				numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

			if (cudaError != cudaSuccess)
				return cudaError;

			cudaError = cudaMemcpy(c_C, deviceBufferC, sizeOfGpuTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyDeviceToHost);

			if (cudaError != cudaSuccess)
				return cudaError;

			c_deviceBufferB = deviceBufferB;
			c_C += sizeOfGpuTaskQueue * lengthOfArray;
		}
	}
	cudaError = cudaMemcpyAsync(deviceBufferA, bufferA, sizeOfFilledGpuTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = processFunction(deviceBufferA, deviceBufferB, deviceBufferC,
		lengthOfArray, sizeOfFilledGpuTaskQueue,
		(sizeOfFilledGpuTaskQueue + numberOfGPUProcessorThread - 1) / numberOfGPUProcessorThread, numberOfGPUProcessorThread);

	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = cudaMemcpy(c_C, deviceBufferC, sizeOfFilledGpuTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess)
		return cudaError;

	return 0;
}

LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, LibMatchMeasureMethod method,
	float **_result)
{
	LibMatchErrorCode errorCode = LibMatchErrorCode::success;
	ArrayMatchContext *context = static_cast<ArrayMatchContext *>(instance);

	int numberOfArray = context->numberOfArrayA;
	int lengthOfArray = context->lengthOfArray;
	float *result = context->result;

	float *deviceBufferA = context->deviceBufferA;
	float *deviceBufferB = context->deviceBufferB;
	float *deviceBufferC = context->deviceBufferC;

	int numberOfThreads = context->numberOfThreads;

	if (numberOfArray == 1)
		numberOfThreads = 1;

	ThreadPool &pool = globalContext.pool;
		
	int perThreadNumberOfArray = numberOfArray / numberOfThreads;

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	size_t perThreadDeviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, lengthOfArray);
	size_t perThreadDeviceBufferBSize = perThreadDeviceBufferASize;
	size_t perThreadDeviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		ArrayMatchExecutionContext &executionContext = context->executionContext[indexOfThread];

		int c_numberOfArray;
		if (indexOfThread + 1 != numberOfThreads)
			c_numberOfArray = perThreadNumberOfArray;
		else
			c_numberOfArray = numberOfArray - indexOfThread * perThreadNumberOfArray;

		executionContext.lengthOfArray = lengthOfArray;
		executionContext.numberOfArray = numberOfArray;
		executionContext.startIndexA = indexOfThread * perThreadNumberOfArray;
		executionContext.numberOfIteration = c_numberOfArray;
		executionContext.bufferA = context->bufferA + indexOfThread * perThreadDeviceBufferASize;
		executionContext.A = A;
		executionContext.B = B;
		executionContext.C = result;
		executionContext.deviceBufferA = deviceBufferA + indexOfThread * perThreadDeviceBufferASize;
		executionContext.deviceBufferB = deviceBufferB + indexOfThread * perThreadDeviceBufferBSize;
		executionContext.deviceBufferC = deviceBufferC + indexOfThread * perThreadDeviceBufferCSize;
		executionContext.numberOfGPUDeviceMultiProcessor = numberOfGPUDeviceMultiProcessor;
		executionContext.numberOfGPUProcessorThread = numberOfGPUProcessorThread;
		
		if (method == LibMatchMeasureMethod::mse)
			context->taskHandle[indexOfThread] = pool.submit(reinterpret_cast<unsigned(*)(void*)>(arrayMatchWorker<arrayMatchMse>), &executionContext);
		else if (method == LibMatchMeasureMethod::cc)
			context->taskHandle[indexOfThread] = pool.submit(reinterpret_cast<unsigned(*)(void*)>(arrayMatchWorker<arrayMatchCc>), &executionContext);
		else
		{
			setLastErrorString("Measure Method hasn't implement yet");
			return LibMatchErrorCode::errorInternal;
		}
	}

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		pool.join(context->taskHandle[indexOfThread]);
	}

	for (int i = 0; i < numberOfThreads; ++i)
	{
		if (pool.get_rc(context->taskHandle[i]) != 0) {
			setLastErrorString("Internal CUDA error");
			errorCode = LibMatchErrorCode::errorCuda;
		}
		pool.release(context->taskHandle[i]);
	}

	if (errorCode == LibMatchErrorCode::success)
	{
		*_result = result;
	}

	return errorCode;
}