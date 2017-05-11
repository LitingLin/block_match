#include <cmath>

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations)
{
	double numberOfBlockAPerProcessor = static_cast<double>(maxNumberOfGpuThreads) / static_cast<double>(numberOfBlockBPerBlockA);
	if (numberOfBlockAPerProcessor > 1.0)
	{
		int fixedNumberOfBlockAPerProcessor = static_cast<int>(numberOfBlockAPerProcessor);
		*numberOfSubmitThreadsPerProcessor = static_cast<int>(numberOfBlockAPerProcessor) * numberOfBlockBPerBlockA;
		*numberOfSubmitProcessors = numberOfGpuProcessors;
		*numberOfIterations = fixedNumberOfBlockAPerProcessor * numberOfGpuProcessors;
	}
	else
	{
		double numberOfProcessorPerBlockA = 1.0 / numberOfBlockAPerProcessor;
		if (numberOfProcessorPerBlockA < numberOfGpuProcessors)
		{
			int _numberOfIterations = static_cast<int>(static_cast<double>(numberOfGpuProcessors) / numberOfProcessorPerBlockA);
			int _numberOfSubmitProcessors = static_cast<int>(std::ceil(_numberOfIterations * numberOfProcessorPerBlockA));
			*numberOfSubmitThreadsPerProcessor = maxNumberOfGpuThreads;
			*numberOfSubmitProcessors = _numberOfSubmitProcessors;
			*numberOfIterations = _numberOfIterations;
		}
		else
		{
			*numberOfSubmitThreadsPerProcessor = maxNumberOfGpuThreads;
			*numberOfSubmitProcessors = static_cast<int>(std::ceil(numberOfProcessorPerBlockA));
			*numberOfIterations = 1;
		}
	}
}

int determineNumberOfThreads(bool sort,
	const int numberOfTasks,
	const int maxNumberOfThreads)
{
	if (sort) {
		if (numberOfTasks < maxNumberOfThreads)
			return numberOfTasks;
		else
			return maxNumberOfThreads;
	}
	else
		if (2 <= maxNumberOfThreads)
			return 2;
		else
			return 1;
}