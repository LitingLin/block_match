#include "lib_match_internal.h"

#include "lib_match_execute.hpp"

template <typename Type, ProcessFunction<Type> processFunction>
unsigned arrayMatchWorker(ArrayMatchExecutionContext<Type>* context)
{
	void *A = context->A, *B = context->B, *C = context->C;
	Type *bufferA = context->bufferA, *bufferB = context->bufferB,
		*deviceBufferA = context->deviceBufferA, *deviceBufferB = context->deviceBufferB, *deviceBufferC = context->deviceBufferC;
	int	numberOfArrayA = context->numberOfArrayA, numberOfArrayB = context->numberOfArrayB, sizeOfArray = context->sizeOfArray,
		startIndexA = context->startIndexA, numberOfIteration = context->numberOfIteration,
		numberOfGPUDeviceMultiProcessor = context->numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread = context->numberOfGPUProcessorThread;

	int sizeOfGpuTaskQueue = numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread;
	int indexOfGpuTaskQueue = 0;

	ArrayCopyMethod *arrayCopyingA = context->arrayCopyingAFunction;
	ArrayCopyMethod *arrayCopyingB = context->arrayCopyingBFunction;

	char *c_A = static_cast<char*>(A);
	char *c_B = static_cast<char*>(B);
	int elementSizeOfTypeA = context->elementSizeOfTypeA, elementSizeOfTypeB = context->elementSizeOfTypeB;

	Type *c_bufferA = bufferA;
	Type *c_bufferB = bufferB;

	cudaError_t cudaError;

	int indexOfIteration = 0;
	int indexOfArrayB = 0;
	int numberOfFilledTaskQueue = 0;
	int numberOfAInQueue = 0;
	int numberOfBInQueue = 0;

	for (int indexOfA = startIndexA; indexOfA < numberOfArrayA; ++indexOfA)
	{
		arrayCopyingA(c_bufferA, c_A, sizeOfArray);
		c_bufferA += sizeOfArray;
		c_A += elementSizeOfTypeA * sizeOfArray;
		numberOfAInQueue++;
		for (int indexOfB = 0; indexOfB < numberOfArrayB; ++indexOfB)
		{
			arrayCopyingB(c_bufferB, c_B, sizeOfArray);
			c_bufferB += sizeOfArray;
			c_B += elementSizeOfTypeB * sizeOfArray;
			numberOfBInQueue++;

			numberOfFilledTaskQueue++;
			if (numberOfFilledTaskQueue == sizeOfGpuTaskQueue)
			{
				
				if (numberOfAInQueue != 1)
				{
					memcpy(bufferA, c_bufferA - sizeOfArray, sizeof(Type) * sizeOfArray);
					c_bufferA = bufferA + sizeOfArray;
					numberOfArrayA = 1;
				}
			}

			if (indexOfIteration == numberOfIteration)
				goto JumpOut;
		}
	}
	JumpOut:

	for (int indexOfIteration = 0; indexOfIteration < numberOfIteration; indexOfIteration += indexOfIteration)
	{
		if (numberOfFilledTaskQueue + numberOfArrayB - indexOfArrayB <= sizeOfGpuTaskQueue)
			indexOfIteration = numberOfArrayB - indexOfArrayB;
		else
			indexOfIteration = sizeOfGpuTaskQueue - numberOfFilledTaskQueue;

		for (int indexOfA = 0; indexOfA < indexOfIteration; ++indexOfA)
		{
			memcpy(c_bufferA, c_A, lengthOfArray * sizeof(float));
			c_bufferA += lengthOfArray;
		}

		memcpy(c_bufferB, c_B, indexOfIteration * lengthOfArray * sizeof(float));

		c_bufferB += indexOfIteration * lengthOfArray;
		c_B += indexOfIteration * lengthOfArray;

		indexOfArrayB += indexOfIteration;
		numberOfFilledTaskQueue += indexOfIteration;
		if (indexOfArrayB == numberOfArrayB)
		{
			c_A += lengthOfArray;
			c_B = B;
			indexOfArrayB = 0;
		}

		if (numberOfFilledTaskQueue == sizeOfGpuTaskQueue)
		{
			cudaError = cudaMemcpyAsync(deviceBufferA, bufferA, sizeOfGpuTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaError != cudaSuccess)
				return cudaError;
			c_bufferA = bufferA;

			cudaError = cudaMemcpyAsync(deviceBufferB, bufferB, sizeOfGpuTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaError != cudaSuccess)
				return cudaError;
			c_bufferB = bufferB;

			cudaError = processFunction(deviceBufferA, deviceBufferB, deviceBufferC,
				lengthOfArray, sizeOfGpuTaskQueue,
				numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

			if (cudaError != cudaSuccess)
				return cudaError;

			cudaError = cudaMemcpy(c_C, deviceBufferC, sizeOfGpuTaskQueue * sizeof(float), cudaMemcpyDeviceToHost);

			if (cudaError != cudaSuccess)
				return cudaError;

			c_C += sizeOfGpuTaskQueue;
			numberOfFilledTaskQueue = 0;
		}
	}
	cudaError = cudaMemcpyAsync(deviceBufferA, bufferA, numberOfFilledTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = cudaMemcpyAsync(deviceBufferB, bufferB, numberOfFilledTaskQueue * lengthOfArray * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = processFunction(deviceBufferA, deviceBufferB, deviceBufferC,
		lengthOfArray, numberOfFilledTaskQueue,
		(numberOfFilledTaskQueue + numberOfGPUProcessorThread - 1) / numberOfGPUProcessorThread, numberOfGPUProcessorThread);

	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = cudaMemcpy(c_C, deviceBufferC, numberOfFilledTaskQueue * sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess)
		return cudaError;

	return 0;
}

LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, LibMatchMeasureMethod method,
	float **_result)
{
	LibMatchErrorCode errorCode = LibMatchErrorCode::success;
	ArrayMatchContext *context = static_cast<ArrayMatchContext *>(instance);

	int numberOfArrayA = context->numberOfArrayA;
	int numberOfArrayB = context->numberOfArrayB;
	int lengthOfArray = context->lengthOfArray;
	float *result = context->result;

	float *deviceBufferA = context->deviceBufferA;
	float *deviceBufferB = context->deviceBufferB;
	float *deviceBufferC = context->deviceBufferC;

	int numberOfThreads = context->numberOfThreads;

	if (numberOfArrayA == 1)
		numberOfThreads = 1;

	execution_service &pool = globalContext.pool;

	int perThreadNumberOfArrayA = numberOfArrayA / numberOfThreads;

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;

	size_t perThreadDeviceBufferASize = arrayMatchPerThreadDeviceBufferASize(numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, lengthOfArray);
	size_t perThreadDeviceBufferBSize = perThreadDeviceBufferASize;
	size_t perThreadDeviceBufferCSize = arrayMatchPerThreadDeviceBufferCSize(numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread);

	for (int indexOfThread = 0; indexOfThread < numberOfThreads; ++indexOfThread)
	{
		ArrayMatchExecutionContext &executionContext = context->executionContext[indexOfThread];

		int c_numberOfArrayA;
		if (indexOfThread + 1 != numberOfThreads)
			c_numberOfArrayA = perThreadNumberOfArrayA;
		else
			c_numberOfArrayA = numberOfArrayA - indexOfThread * perThreadNumberOfArrayA;

		executionContext.lengthOfArray = lengthOfArray;
		executionContext.numberOfArrayA = numberOfArrayA;
		executionContext.numberOfArrayB = numberOfArrayB;
		executionContext.startIndexA = indexOfThread * perThreadNumberOfArrayA;
		executionContext.numberOfIteration = c_numberOfArrayA * numberOfArrayB;
		executionContext.bufferA = context->bufferA + indexOfThread * perThreadDeviceBufferASize;
		executionContext.bufferB = context->bufferB + indexOfThread * perThreadDeviceBufferBSize;
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