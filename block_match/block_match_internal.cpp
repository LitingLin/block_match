#include "block_match_internal.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stddef.h>

#include <spdlog/spdlog.h>

const int numberOfGPUProcessorThread = 512;

// Hyper-Threading do harms to arithmetic computation
unsigned getNumberOfPhysicalProcessor()
{
	DWORD bufferLength = 0;
	GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferLength);

	// Guess actual size
	const unsigned structSize = offsetof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor)
		+ sizeof(_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX::Processor);

#ifndef NDEBUG
	// Check it
	if (bufferLength % structSize) abort();
#endif

	return bufferLength / structSize;
}

GlobalContext::GlobalContext()
	: 
#ifdef NDEBUG
	numberOfThreads(getNumberOfPhysicalProcessor()),
#else
	numberOfThreads(1),
#endif
	pool(numberOfThreads), numberOfGPUProcessorThread(::numberOfGPUProcessorThread)
{
}

bool GlobalContext::initialize()
{
	cudaError_t cuda_error = cudaDeviceGetAttribute(&numberOfGPUDeviceMultiProcessor, cudaDevAttrMultiProcessorCount, 0);
	if (cuda_error != cudaSuccess) {
		hasGPU = false;
		return false;
	}
	else {
		hasGPU = true;
		cuda_error = cudaSetDeviceFlags(cudaDeviceScheduleYield); // save cpu time
		if (cuda_error != cudaSuccess) {
			logger.warn("cudaSetDeviceFlags(cudaDeviceScheduleYield) return {}, message:{}.", cuda_error, cudaGetErrorString(cuda_error));
			return false;
		}
	}

	return true;
}

GlobalContext globalContext;
