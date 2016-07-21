#include "block_match_internal.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stddef.h>

const int numberOfGPUProcessorThread = 512;

// Hyper-Threading do harms to arithmetic computation
unsigned getNumberOfPhysicalProcessor()
{
	DWORD bufferLength = 0;
	GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferLength);

	// Guess actual size
	const unsigned structSize = offsetof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, DUMMYUNIONNAME.Processor) + sizeof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX::DUMMYUNIONNAME.Processor);

#ifndef NDEBUG
	// Check it
	if (bufferLength % structSize) abort();
#endif

	return bufferLength / structSize;
}

GlobalContext::GlobalContext()
	: numberOfThreads(getNumberOfPhysicalProcessor()), pool(numberOfThreads), numberOfGPUProcessorThread(numberOfGPUProcessorThread)
{
	cudaError_t cuda_error = cudaDeviceGetAttribute(&numberOfGPUDeviceMultiProcessor, cudaDevAttrMultiProcessorCount, 0);
	if (cuda_error != cudaSuccess) hasGPU = false;
	else hasGPU = true;
}

GlobalContext globalContext;