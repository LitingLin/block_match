#include "lib_match_internal.h"

#include "lib_match.h"

#include "array_match_execute.hpp"

template <typename Type>
void initializeExecutionContext(ArrayMatchContext<Type> *instance)
{
	const int numberOfThreads = instance->numberOfThreads;
	const int numberOfTasks = instance->numberOfArrayA;
	const int minimumNumberOfTaskPerThread = numberOfTasks / numberOfThreads;
	for (int indexOfThread=  0;indexOfThread<numberOfThreads;++indexOfThread)
	{
		const int beginIndexOfA = minimumNumberOfTaskPerThread * indexOfThread;
		instance->executionContexts.emplace_back(ArrayMatchContext<Type>::ExecutionContext{
			beginIndexOfA, minimumNumberOfTaskPerThread,
			std::make_unique<ArrayMatchExecutionContext<Type>>() });
	}
	instance->executionContexts[numberOfThreads - 1].numberOfIteration += (numberOfTasks - minimumNumberOfTaskPerThread * numberOfThreads);
}

template <typename Type>
ArrayMatch<Type>::ArrayMatch(std::type_index inputADataType, std::type_index inputBDataType,
	std::type_index outputDataType,
	std::type_index indexDataType,
	MeasureMethod measureMethod,
	bool sort,
	int numberOfA, int numberOfB,
	int size,
	int numberOfResultRetain)
	: m_instance(nullptr), inputADataType(inputADataType), inputBDataType(inputBDataType),
	outputDataType(outputDataType), indexDataType(indexDataType)
{
	int numberOfThreads = globalContext.numberOfThreads;
	if (!sort)
	{
		if (numberOfThreads > 2)
			numberOfThreads = 2;
	}

	ArrayMatchExecutionFunction<Type> *executionFunction = nullptr;
	ArrayCopyMethod *arrayACopyFunction = nullptr;
	ArrayCopyMethod *arrayBCopyFunction = nullptr;
	ArrayMatchDataPostProcessingMethod *dataPostProcessingFunction = nullptr;

	if (measureMethod == MeasureMethod::mse)
	{
		executionFunction = arrayMatchWorker<Type, lib_match_mse_global<Type>>;
	}
	else if (measureMethod == MeasureMethod::cc)
	{
		executionFunction = arrayMatchWorker<Type, lib_match_cc_global<Type>>;
	}

#define EXP(type) \
	arrayACopyFunction = (ArrayCopyMethod*)copyArray<Type, type>
	RuntimeTypeInference(inputADataType, EXP);
#undef EXP
	
#define EXP(type) \
	arrayBCopyFunction = (ArrayCopyMethod*)copyArray<Type, type>
	RuntimeTypeInference(inputBDataType, EXP);
#undef EXP

	if (sort)
	{
		if (indexDataType == typeid(nullptr))
		{
			if (measureMethod == MeasureMethod::mse && numberOfResultRetain)
			{
#define exp(type) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_noRecordIndex<Type, type, sortPartialAscend<Type>>
				RuntimeTypeInference(outputDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::mse && !numberOfResultRetain)
			{
#define exp(type) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_noRecordIndex<Type, type, sortAscend<Type>>
				RuntimeTypeInference(outputDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::cc && numberOfResultRetain)
			{
#define exp(type) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_noRecordIndex<Type, type, sortDescend<Type>>
				RuntimeTypeInference(outputDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::cc && !numberOfResultRetain)
			{
#define exp(type) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_noRecordIndex<Type, type, sortPartialDescend<Type>>
				RuntimeTypeInference(outputDataType, exp);
#undef exp
			}
			else
				NOT_IMPLEMENTED_ERROR;
		}
		else
		{
			if (measureMethod == MeasureMethod::mse && numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndexPlusOne<Type, type1, type2, sortPartialAscend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::mse && !numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndexPlusOne<Type, type1, type2, sortAscend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::cc && numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndexPlusOne<Type, type1, type2, sortPartialDescend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::cc && !numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndexPlusOne<Type, type1, type2, sortDescend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
		}
	}
	else
	{
		if (indexDataType == typeid(nullptr))
		{
#define exp(type) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)noSort_noRecordIndex<Type, type>
			RuntimeTypeInference(outputDataType, exp);
#undef exp
		}
		else
		{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)noSort_recordIndexPlusOne<Type, type1, type2>
			RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
		}
	}

	int numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread, sizeOfGpuTaskQueue;
	determineGpuTaskConfiguration(globalContext.numberOfGPUProcessorThread, globalContext.numberOfGPUDeviceMultiProcessor,
		numberOfB, &numberOfGPUProcessorThread, &numberOfGPUDeviceMultiProcessor, &sizeOfGpuTaskQueue);
	
	ArrayMatchContext<Type> *instance = new ArrayMatchContext<Type>{
		numberOfA, numberOfB,
		size,
		numberOfResultRetain,
		executionFunction,
		arrayACopyFunction, arrayBCopyFunction,
		dataPostProcessingFunction,
		std::vector<typename ArrayMatchContext<Type>::PerThreadBuffer>(), // perThreadBuffer
		numberOfThreads,
		std::vector<typename ArrayMatchContext<Type>::ExecutionContext>(), // executionContexts
		std::vector<void *>(numberOfThreads),
		std::vector<cudaStream_guard>(numberOfThreads),
		numberOfGPUDeviceMultiProcessor, numberOfGPUProcessorThread,
		sizeOfGpuTaskQueue,
	};

	initializeExecutionContext(instance);

	const int bufferASize = sizeOfGpuTaskQueue * size;
	const int bufferBSize = numberOfB * size;
	const int bufferCSize = sizeOfGpuTaskQueue * numberOfB;

	if (sort && indexDataType != typeid(nullptr))
	{
		for (int indexOfThread = 0; indexOfThread<numberOfThreads; ++indexOfThread)
		{
			instance->perThreadBuffer.emplace_back(ArrayMatchContext<Type>::PerThreadBuffer{
				memory_allocator<Type, memory_type::page_locked>(bufferASize), // matrixA_buffer
				memory_allocator<Type, memory_type::page_locked>(bufferBSize), // matrixB_buffer
				memory_allocator<Type, memory_type::page_locked>(bufferCSize), // matrixC_buffer
				memory_allocator<Type, memory_type::gpu>(bufferASize), // matrixA_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(bufferBSize), // matrixB_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(bufferCSize), // matrixC_deviceBuffer
				memory_allocator<int, memory_type::system>(numberOfB), // index_sorting_buffer
				memory_allocator<int, memory_type::system>(numberOfB), // index_sorting_template
			});
		}
	}
	else
	{
		for (int indexOfThread = 0; indexOfThread<numberOfThreads; ++indexOfThread)
		{
			instance->perThreadBuffer.emplace_back(ArrayMatchContext<Type>::PerThreadBuffer{
				memory_allocator<Type, memory_type::page_locked>(bufferASize), // matrixA_buffer
				memory_allocator<Type, memory_type::page_locked>(bufferBSize), // matrixB_buffer
				memory_allocator<Type, memory_type::page_locked>(bufferCSize), // matrixC_buffer
				memory_allocator<Type, memory_type::gpu>(bufferASize), // matrixA_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(bufferBSize), // matrixB_deviceBuffer
				memory_allocator<Type, memory_type::gpu>(bufferCSize), // matrixC_deviceBuffer
				memory_allocator<int, memory_type::system>(0), // index_sorting_buffer
				memory_allocator<int, memory_type::system>(0), // index_sorting_template
			});
		}
	}

	m_instance = static_cast<void*>(instance);
}

template
LIB_MATCH_EXPORT
ArrayMatch<float>::ArrayMatch(std::type_index, std::type_index,
	std::type_index,
	std::type_index,
	MeasureMethod,
	bool,
	int, int,
	int,
	int);
template
LIB_MATCH_EXPORT
ArrayMatch<double>::ArrayMatch(std::type_index, std::type_index,
	std::type_index,
	std::type_index,
	MeasureMethod,
	bool,
	int, int,
	int,
	int);

template <typename Type>
void ArrayMatch<Type>::initialize()
{
	ArrayMatchContext<Type> *instance = static_cast<ArrayMatchContext<Type> *>(m_instance);
	const int numberOfThreads = instance->numberOfThreads;

	for (int indexOfThread = 0; indexOfThread<numberOfThreads; ++indexOfThread)
	{
		typename ArrayMatchContext<Type>::PerThreadBuffer &threadBuffer = instance->perThreadBuffer[indexOfThread];
		threadBuffer.matrixA_buffer.alloc();
		threadBuffer.matrixB_buffer.alloc();
		threadBuffer.matrixC_buffer.alloc();
		threadBuffer.matrixA_deviceBuffer.alloc();
		threadBuffer.matrixB_deviceBuffer.alloc();
		threadBuffer.matrixC_deviceBuffer.alloc();
		threadBuffer.index_sorting_buffer.alloc();
		threadBuffer.index_sorting_template.alloc();

		if (threadBuffer.index_sorting_template.allocated())
			generateIndexSequence(threadBuffer.index_sorting_template.get(), instance->numberOfArrayB);
	}
}

template
LIB_MATCH_EXPORT
void ArrayMatch<float>::initialize();
template
LIB_MATCH_EXPORT
void ArrayMatch<double>::initialize();