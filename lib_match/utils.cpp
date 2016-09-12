#include "lib_match.h"

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

int getLength(int matSize, int paddingSize, int blockSize, int strideSize)
{
	return (matSize + 2 * paddingSize - blockSize) / strideSize + 1;
}

int determineEndOfIndex(int matSize, int paddingSize, int blockSize)
{
	return matSize + paddingSize - blockSize + 1;
}

void generateIndexSequence(int *index, int size)
{
	for (int i = 0; i < size; ++i)
	{
		index[i] = i;
	}
}

template <typename TypeA, typename TypeB>
void type_convert(TypeA *a, TypeB *b, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		a[i] = b[i];
}

char errorStringBuffer[LIB_MATCH_MAX_MESSAGE_LENGTH];

void libMatchGetLastErrorString(char *buffer, size_t size)
{
	strncpy_s(buffer, size, errorStringBuffer, size);
}

void setLastErrorString(const char *string, ...)
{
	va_list args;
	va_start(args, string);
	snprintf(errorStringBuffer, LIB_MATCH_MAX_MESSAGE_LENGTH, string, args);
	va_end(args);
}

void setCudaLastErrorString(cudaError_t cudaError, char *message)
{
	setLastErrorString("%s"
		"Cuda Error Code: %d, Message: %s",
		message, cudaError, cudaGetErrorString(cudaError));
}

size_t arrayMatchPerThreadASize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread,
	const int lengthOfArray)
{
	return numberOfGpuDeviceMultiProcessor * numberOfGpuProcessorThread * lengthOfArray;
}

size_t arrayMatchPerThreadBSize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread,
	const int lengthOfArray)
{
	return numberOfGpuDeviceMultiProcessor * numberOfGpuProcessorThread * lengthOfArray;
}

size_t arrayMatchPerThreadCSize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread,
	const int lengthOfArray)
{
	return numberOfGpuDeviceMultiProcessor * numberOfGpuProcessorThread * lengthOfArray;
}

size_t arrayMatchPerThreadDeviceBufferASize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread,
	const int lengthOfArray)
{
	return numberOfGpuDeviceMultiProcessor * numberOfGpuProcessorThread * lengthOfArray;
}

size_t arrayMatchPerThreadDeviceBufferBSize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread,
	const int lengthOfArray)
{
	return numberOfGpuDeviceMultiProcessor * numberOfGpuProcessorThread * lengthOfArray;
}

size_t arrayMatchPerThreadDeviceBufferCSize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread)
{
	return numberOfGpuDeviceMultiProcessor * numberOfGpuProcessorThread;
}