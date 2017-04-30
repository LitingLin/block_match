#include <algorithm>

template <typename Type>
void lib_match_sort(Type *value, int size)
{
	std::sort(value, value + size);
}

template <typename Type>
void lib_match_sort_partial(Type *value, int size, int retain)
{
	std::partial_sort(value, value + retain, value + size);
}

template <typename Type>
void lib_match_sort_descend(Type *value, int size)
{
	std::sort(value, value + size);
}

template <typename Type>
void lib_match_sort_partial_descend(Type *value, int size, int retain)
{
	std::partial_sort(value, value + retain, value + size);
}

template <typename Type>
void lib_match_sort(int *index, Type *value, int size)
{
	std::sort(index, index + size, [value](size_t i1, size_t i2) {return value[i1] < value[i2]; });
}

template <typename Type>
void lib_match_sort_partial(int *index, Type *value, int size, int retain)
{
	std::partial_sort(index, index + retain, index + size, [value](size_t i1, size_t i2) {return value[i1] < value[i2]; });
}

template <typename Type>
void lib_match_sort_descend(int *index, Type *value, int size)
{
	std::sort(index, index + size, [value](size_t i1, size_t i2) {return value[i1] > value[i2]; });
}

template <typename Type>
void lib_match_sort_partial_descend(int *index, Type *value, int size, int retain)
{
	std::partial_sort(index, index + retain, index + size, [value](size_t i1, size_t i2) {return value[i1] > value[i2]; });
}

template
void lib_match_sort(float *, int );
template
void lib_match_sort(double *, int );
template
void lib_match_sort(int *, float *, int);
template
void lib_match_sort(int *, double *, int);
template
void lib_match_sort_partial(float *, int, int);
template
void lib_match_sort_partial(double *, int, int);
template
void lib_match_sort_partial(int *, float *, int, int);
template
void lib_match_sort_partial(int *, double *, int, int);
template
void lib_match_sort_descend(float *, int);
template
void lib_match_sort_descend(double *, int);
template
void lib_match_sort_descend(int *, float *, int);
template
void lib_match_sort_descend(int *, double *, int);
template
void lib_match_sort_partial_descend(float *, int, int);
template
void lib_match_sort_partial_descend(double *, int, int);
template
void lib_match_sort_partial_descend(int *, float *, int, int);
template
void lib_match_sort_partial_descend(int *, double *, int, int);