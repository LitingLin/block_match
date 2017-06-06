#include "lib_match_internal.h"

#ifdef _MSC_VER

#define VC_EXTRALEAN
#include <Windows.h>
#include <stddef.h>

const int numberOfGPUProcessorThread = 512;

#define MAX_NUMBER_OF_PROCESSOR 1

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

#elif defined __unix__

#include <stdio.h>

int command_get_int(const char *command)
{
	FILE *handle = popen(command, "r");
	char buf[11]; // length of INT_MAX in string
	size_t size = 11;
	CHECK_POINT_EQ(fgets(buf, size, handle), buf);

	int number;
	number = atoi(buf);

	pclose(handle);

	return number;
}

int get_number_of_sockets()
{
	return command_get_int("grep -i \"physical id\" /proc/cpuinfo | sort -u | wc -l");
}

int get_number_of_processors()
{
	return command_get_int("grep -i processor /proc/cpuinfo | sort -u | wc -l");
}

int get_number_of_threads_per_core()
{
	return command_get_int("lscpu | grep -i thread | tail -c 2");
}

unsigned getNumberOfPhysicalProcessor()
{
	return get_number_of_processors() / get_number_of_threads_per_core();
}

#endif

#define LIMIT_NUMBER_OF_THREAD_IN_DEBUG_MODE 0

unsigned getNumberOfProcessor()
{
	unsigned numberOfThreads;
#if (LIMIT_NUMBER_OF_THREAD_IN_DEBUG_MODE && !(defined NDEBUG))
	numberOfThreads = 1;
#elif defined(MAX_NUMBER_OF_PROCESSOR)
	numberOfThreads = getNumberOfPhysicalProcessor();
	if (numberOfThreads > MAX_NUMBER_OF_PROCESSOR)
		numberOfThreads = MAX_NUMBER_OF_PROCESSOR;
#else
	numberOfThreads = getNumberOfPhysicalProcessor();
#endif

	return numberOfThreads;
}


GlobalContext::GlobalContext()
	: numberOfThreads(getNumberOfProcessor()),
	exec_serv(numberOfThreads), numberOfGPUProcessorThread(::numberOfGPUProcessorThread),
	hasGPU(false)
{
	initialize();
}

bool GlobalContext::initialize()
{
	int nDevices;

	cudaError_t cuda_error = cudaGetDeviceCount(&nDevices);
	if (cuda_error != cudaSuccess)
		return false;
/*
	if (cuda_error != cudaSuccess)
	{
		logger.critical("cudaGetDeviceCount() return {}, message: {}", cuda_error, cudaGetErrorString(cuda_error));
		logger.critical() << "GPU initialization failed.";
		return false;
	}*/
	numberOfGPUDeviceMultiProcessor.resize(nDevices);

	for (int i = 0; i < nDevices; ++i)
	{
		cuda_error = cudaDeviceGetAttribute(&numberOfGPUDeviceMultiProcessor[i], cudaDevAttrMultiProcessorCount, i);
		if (cuda_error != cudaSuccess)
			return false;
		/*
		if (cuda_error != cudaSuccess) {
			logger.critical("cudaDeviceGetAttribute(&numberOfGPUDeviceMultiProcessor[i], cudaDevAttrMultiProcessorCount, i)"
				"return {}, message: {}, i={}", cuda_error, cudaGetErrorString(cuda_error), i);
			return false;
		}*/
	}
	hasGPU = true;
	cuda_error = cudaSetDeviceFlags(cudaDeviceScheduleYield); // save cpu time
	if (cuda_error != cudaSuccess)
		return false;
	/*if (cuda_error != cudaSuccess) {
		logger.warn("cudaSetDeviceFlags(cudaDeviceScheduleYield) return {}, message: {}", cuda_error, cudaGetErrorString(cuda_error));
		return false;
	}*/

	return true;
}

GlobalContext globalContext;
