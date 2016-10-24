#pragma once

#include <lib_match_internal.h>

#include <boost/test/unit_test.hpp>

template <typename Type>
void isNormal(Type *ptr, int size);

void fillWithSequence(float *ptr, size_t size);

void fillWithSequence(int *ptr, size_t size);