#include <stdlib.h>

size_t getMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArrayA, int numberOfArrayB)
{
	return lengthOfArray * (numberOfArrayA + numberOfArrayB) * sizeof(float);
}