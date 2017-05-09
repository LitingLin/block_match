#include "lib_match_internal.h"

#include "lib_match.h"

#include "array_match_execute.hpp"

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
	
	ArrayCopyMethod *arrayACopyFunction;
	ArrayCopyMethod *arrayBCopyFunction;
	ArrayMatchDataPostProcessingMethod *dataPostProcessingFunction;

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
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndex<Type, type1, type2, sortPartialAscend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::mse && !numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndex<Type, type1, type2, sortAscend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::cc && numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndex<Type, type1, type2, sortPartialDescend<Type>>
				RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
			}
			else if (measureMethod == MeasureMethod::cc && !numberOfResultRetain)
			{
#define exp(type1, type2) \
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)sort_recordIndex<Type, type1, type2, sortDescend<Type>>
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
	dataPostProcessingFunction = (ArrayMatchDataPostProcessingMethod*)noSort_recordIndex<Type, type1, type2>
			RuntimeTypeInference2(outputDataType, indexDataType, exp);
#undef exp
		}
	}

	const int numberOfGPUDeviceMultiProcessor = globalContext.numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread = globalContext.numberOfGPUProcessorThread;
	const int sizeOfGpuTaskQueue = numberOfGPUDeviceMultiProcessor * numberOfGPUProcessorThread;



	ArrayMatchContext<Type> *instance = new ArrayMatchContext<Type>{
		numberOfA, numberOfB,
		size,
		std::vector<typename ArrayMatchContext<Type>::PerThreadBuffer>(),
		numberOfThreads,
		std::vector<ArrayMatchExecutionContext<Type>>(),
		std::vector<void *>(numberOfThreads)
	};

	for (int indexOfThread = 0;indexOfThread<numberOfThreads;++indexOfThread)
	{
		instance->executionContext.emplace_back(typename ArrayMatchContext<Type>::PerThreadBuffer{

		});
	}

	m_instance = static_cast<void*>(instance);
}

template <typename Type>
void ArrayMatch<Type>::initialize()
{
}
