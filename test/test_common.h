#pragma once

#include <boost/test/unit_test.hpp>

#include <lib_match.h>
#include <cstdlib>

template <typename T1, typename T2, typename T3>
bool inRange(T1 x, T2 a, T3 b)
{
	return (x >= static_cast<T1>(a) && x < static_cast<T1>(b));
}

template <typename T1, typename T2, size_t size>
bool inRange(T1 x, T2 (&a)[size])
{
	for (size_t i=0;i<size;++i)
	{
		if (x == static_cast<T1>(a[i]))
			return true;
	}
	return false;
}