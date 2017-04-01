#include "lib_match_internal.h"

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

int getLength(int matSize, int paddingSize, int blockSize, int strideSize)
{
	return (matSize + 2 * paddingSize - blockSize) / strideSize + 1;
}

int getLength(int matSize, int prePaddingSize, int postPaddingSize, int blockSize, int strideSize)
{
	return (matSize + prePaddingSize + postPaddingSize - blockSize) / strideSize + 1;
}

int determineEndOfIndex(int matSize, int blockSize)
{
	return matSize - blockSize + 1;
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

double diagnose::getFinishedPercentage(void* instance)
{
	
}