#include <lib_match.h>
#include <sstream>
#include <mex.h>

void logging_function(const char* msg)
{
	mexPrintf(msg);
}

void reportMemoryAllocationFailed(size_t currentMxMemorySize, size_t maxMxMemorySize)
{
	size_t currentSystemMemorySize, currentPageLockedMemorySize, currentGpuMemorySize,
		maxSystemMemorySize, maxPageLockedMemorySize, maxGpuMemorySize;

	LibMatchDiagnose::getCurrentMemoryUsage(&currentSystemMemorySize, &currentPageLockedMemorySize, &currentGpuMemorySize);
	LibMatchDiagnose::getMaxMemoryUsage(&maxSystemMemorySize, &maxPageLockedMemorySize, &maxGpuMemorySize);

	std::ostringstream message;
	message << "MATLAB memory allocation failed.\n"
		<< "\tCurrent\tMax(estimated)\n"
		<< "System:\t" << currentSystemMemorySize << "\t" << maxSystemMemorySize << "\n"
		<< "Page Locked:\t" << currentPageLockedMemorySize << "\t" << maxPageLockedMemorySize << "\n"
		<< "GPU:\t" << currentGpuMemorySize << "\t" << maxGpuMemorySize << "\n"
		<< "MATLAB:\t" << currentMxMemorySize << "\t" << maxMxMemorySize << "\n";

	mexPrintf(message.str().c_str());
}