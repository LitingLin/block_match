#pragma once

#include <boost/test/unit_test.hpp>

#include <lib_match.h>
#include <cstdlib>

const float singleFloatingPointErrorTolerance = 0.0001f;
const double doubleFloatingPointErrorTolerance = 0.0001;

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

typedef void * HANDLE;

class MemoryMappedIO
{
public:
	MemoryMappedIO(const wchar_t *filename);
	~MemoryMappedIO() noexcept(false);
	const void *getPtr() const;
private:
	HANDLE hFile;
	HANDLE hFileMapping;
	void *ptr;
};

void checkFloatPointEqual(const double* a, const double* b, size_t n);

void checkIndexEqual(const uint8_t* m, const uint8_t* n, const uint8_t* groundtruth, size_t elemsize, size_t num);
