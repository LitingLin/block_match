#include <stdlib.h>

size_t getMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArray)
{
	return (lengthOfArray * numberOfArray * 2 + numberOfArray) * sizeof(float);
}