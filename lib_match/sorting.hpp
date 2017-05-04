#pragma once

template <typename Type>
using RawSortMethod_WithIndex = void(*)(int *index, Type *value, int size, int retain);
template <typename Type>
using RawSortMethod = void(*)(Type *value, int size, int retain);

template <typename Type>
void sortAscend(int *index, Type *value, int size, int retain)
{
	lib_match_sort(index, value, size);
}

template <typename Type>
void sortPartialAscend(int *index, Type *value, int size, int retain)
{
	lib_match_sort_partial(index, value, size, retain);
}

template <typename Type>
void sortDescend(int *index, Type *value, int size, int retain)
{
	lib_match_sort_descend(index, value, size);
}

template <typename Type>
void sortPartialDescend(int *index, Type *value, int size, int retain)
{
	lib_match_sort_partial_descend(index, value, size, retain);
}

template <typename Type>
void sortAscend(Type *value, int size, int retain)
{
	lib_match_sort(value, size);
}

template <typename Type>
void sortPartialAscend(Type *value, int size, int retain)
{
	lib_match_sort_partial(value, size, retain);
}

template <typename Type>
void sortDescend(Type *value, int size, int retain)
{
	lib_match_sort_descend(value, size);
}

template <typename Type>
void sortPartialDescend(Type *value, int size, int retain)
{
	lib_match_sort_partial_descend(value, size, retain);
}
