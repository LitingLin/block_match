#include <stdio.h>
#include <string.h>
#include <stdarg.h>

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

const size_t errorStringBufferSize = 128;
char errorStringBuffer[errorStringBufferSize];

void getLastErrorString(char *buffer, size_t size)
{
	strncpy_s(buffer, size, errorStringBuffer, size);
}

void setLastErrorString(const char *string, ...)
{
	va_list args;
	va_start(args, string);
	snprintf(errorStringBuffer, errorStringBufferSize, string, args);
	va_end(args);
}