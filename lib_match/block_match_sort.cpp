#include <algorithm>

void block_sort(int *index, float *value, int size)
{
	std::sort(index, index + size, [value](size_t i1, size_t i2) {return value[i1] < value[i2]; });
}

void block_sort_partial(int *index, float *value, int size, int retain)
{
	std::partial_sort(index, index + retain, index + size, [value](size_t i1, size_t i2) {return value[i1] < value[i2]; });
}

void block_sort_descend(int *index, float *value, int size)
{
	std::sort(index, index + size, [value](size_t i1, size_t i2) {return value[i1] > value[i2]; });
}

void block_sort_partial_descend(int *index, float *value, int size, int retain)
{
	std::partial_sort(index, index + retain, index + size, [value](size_t i1, size_t i2) {return value[i1] > value[i2]; });
}
