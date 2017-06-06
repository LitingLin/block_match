#define BOOST_TEST_MODULE block_match
#include <boost/test/unit_test.hpp>

#include <lib_match.h>
#include "test_common.h"
#include <iostream>

void logging_sink(const char *msg)
{
	puts(msg);
}

bool dummyIsInterruptPending()
{
	return false;
}

class GlobalContextInitializer
{
public:
	GlobalContextInitializer()
	{
		libMatchRegisterLoggingSinkFunction(logging_sink);
		libMatchRegisterInterruptPeddingFunction(dummyIsInterruptPending);
		libMatchOnLoad();
	}
	~GlobalContextInitializer()
	{
		libMatchAtExit();
	}
}initializer;

#define VC_EXTRALEAN
#include <windows.h>

MemoryMappedIO::MemoryMappedIO(const wchar_t* filename)
{
	hFile =  CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
		throw std::exception();
	hFileMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hFileMapping == NULL)
		throw std::exception();
	ptr = MapViewOfFileEx(hFileMapping, FILE_MAP_READ, 0, 0, 0, NULL);
	if (ptr == NULL)
		throw std::exception();
}

MemoryMappedIO::~MemoryMappedIO() noexcept(false)
{
	if (UnmapViewOfFile(ptr) != TRUE)
		throw std::exception();
	if (CloseHandle(hFileMapping) != TRUE)
		throw std::exception();
	if (CloseHandle(hFile) != TRUE)
		throw std::exception();
}

const void* MemoryMappedIO::getPtr() const 
{
	return ptr;
}

#include <cmath>

void checkFloatPointEqual(const double* a, const double* b, size_t n)
{
	for (size_t i = 0; i < n; ++i) {
		if (std::isnan(a[i]) && std::isnan(b[i])) continue;
		BOOST_CHECK_SMALL(a[i] - b[i], doubleFloatingPointErrorTolerance);
	}
}

void checkIndexEqual(const uint8_t* m, const uint8_t* n, const uint8_t* groundtruth, size_t elemsize, size_t num)
{
	size_t ptroffset = 0;
	size_t m_offset = 0;
	size_t n_offset = 0;
	for (size_t i = 0; i < num; ++i)
	{
		for (size_t j = 0; j < elemsize; ++j) {
			BOOST_CHECK_EQUAL(groundtruth[ptroffset], m[m_offset]);
			ptroffset++; m_offset++;
		}
		for (size_t j = 0; j < elemsize; ++j) {
			BOOST_CHECK_EQUAL(groundtruth[ptroffset], n[n_offset]);
			ptroffset++; n_offset++;
		}
	}
}
