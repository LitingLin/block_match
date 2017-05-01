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

int getTypeSize(std::type_index type)
{
	if (type == typeid(uint8_t))
		return sizeof(uint8_t);
	else if (type == typeid(int8_t))
		return sizeof(int8_t);
	else if (type == typeid(uint16_t))
		return sizeof(uint16_t);
	else if (type == typeid(int16_t))
		return sizeof(int16_t);
	else if (type == typeid(uint32_t))
		return sizeof(uint32_t);
	else if (type == typeid(int32_t))
		return sizeof(int32_t);
	else if (type == typeid(uint64_t))
		return sizeof(uint64_t);
	else if (type == typeid(int64_t))
		return sizeof(int64_t);
	else if (type == typeid(float))
		return sizeof(float);
	else if (type == typeid(double))
		return sizeof(double);
	else
		NOT_IMPLEMENTED_ERROR;
}