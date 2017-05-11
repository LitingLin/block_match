#include "lib_match_internal.h"

template <typename Type>
void ArrayMatch<Type>::execute(void* A, void* B, void* C, void* index)
{
	ArrayMatchContext<Type> *instance = static_cast<ArrayMatchContext<Type>*>(m_instance);
	
	const int numberOfThreads = instance->numberOfThreads;
	execution_service &executionService = globalContext.exec_serv;

	for (int indexOfThreads = 0;indexOfThreads < numberOfThreads;++indexOfThreads)
	{
		typename ArrayMatchContext<Type>::PerThreadBuffer &perThreadBuffer = instance->perThreadBuffer[indexOfThreads];
		ArrayMatchExecutionContext<Type> *executionContext = instance->executionContexts[indexOfThreads].context.get();
		executionContext->A = A;
		executionContext->B = B;
		executionContext->C = C;
		executionContext->index = index;

		executionContext->bufferA = perThreadBuffer.matrixA_buffer.get();
		executionContext->bufferB = perThreadBuffer.matrixB_buffer.get();
		executionContext->bufferC = perThreadBuffer.matrixC_buffer.get();
		executionContext->deviceBufferA = perThreadBuffer.matrixA_deviceBuffer.get();
		executionContext->deviceBufferB = perThreadBuffer.matrixB_deviceBuffer.get();
		executionContext->deviceBufferC = perThreadBuffer.matrixC_deviceBuffer.get();

		executionContext->numberOfArrayA = instance->numberOfArrayA;
		executionContext->numberOfArrayB = instance->numberOfArrayB;
		executionContext->sizeOfArray = instance->lengthOfArray;
		executionContext->startIndexA = instance->executionContexts[indexOfThreads].startIndexA;
		executionContext->startIndexB = instance->executionContexts[indexOfThreads].startIndexB;
		executionContext->numberOfIteration = instance->executionContexts[indexOfThreads].numberOfIteration;

		executionContext->index_sorting_buffer = perThreadBuffer.index_sorting_buffer.get();
		executionContext->index_template = perThreadBuffer.index_sorting_template.get();
		executionContext->elementSizeOfTypeA = getTypeSize(inputADataType);
		executionContext->elementSizeOfTypeB = getTypeSize(inputBDataType);
		executionContext->elementSizeOfTypeC = getTypeSize(outputDataType);
		if (indexDataType != typeid(nullptr))
			executionContext->elementSizeOfIndex = getTypeSize(indexDataType);
		else
			executionContext->elementSizeOfIndex = 0;
		executionContext->retain = instance->numberOfResultRetain;
		executionContext->arrayCopyingAFunction = instance->arrayCopyingAFunction;
		executionContext->arrayCopyingBFunction = instance->arrayCopyingBFunction;
		executionContext->dataPostProcessingFunction = instance->dataPostProcessingFunction;

		executionContext->stream = instance->streams[indexOfThreads];
		executionContext->sizeOfGpuTaskQueue = instance->sizeOfGpuTaskQueue;
		executionContext->numberOfGPUDeviceMultiProcessor = instance->numberOfGPUDeviceMultiProcessor;
		executionContext->numberOfGPUProcessorThread = instance->numberOfGPUProcessorThread;

		instance->threadPoolTaskHandle[indexOfThreads] =
			executionService.submit(reinterpret_cast<unsigned(*)(void*)>(instance->executionFunction), executionContext);
	}
	
	for (int indexOfThreads = 0; indexOfThreads < numberOfThreads; ++indexOfThreads)
	{
		executionService.join(instance->threadPoolTaskHandle[indexOfThreads]);
	}

	bool isFailed = false;

	std::string error_message;
	for (int i = 0; i < numberOfThreads; ++i)
	{
		if (executionService.get_rc(instance->threadPoolTaskHandle[i]) != 0)
			isFailed = true;
		error_message += executionService.get_exp_what(instance->threadPoolTaskHandle[i]);
		executionService.release(instance->threadPoolTaskHandle[i]);
	}

	if (isFailed)
		throw std::runtime_error(error_message);
}

template
LIB_MATCH_EXPORT
void ArrayMatch<float>::execute(void*, void*, void*, void*);
template
LIB_MATCH_EXPORT
void ArrayMatch<double>::execute(void*, void*, void*, void*);