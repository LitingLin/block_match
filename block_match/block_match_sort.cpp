#include <algorithm>

void block_sort(int *index, float *value, int size)
{
	std::sort(index, index + size, [value](size_t i1, size_t i2) {return value[i1] < value[i2]; });
}