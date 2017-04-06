#pragma once

#include <lib_match_internal.h>

#include <boost/test/unit_test.hpp>

const float singleFloatingPointErrorTolerance = 0.0001f;
const double doubleFloatingPointErrorTolerance = 0.0001;

template <typename Type>
void checkIsNormal(Type *ptr, int size);

void fillWithSequence(float *ptr, size_t size);

void fillWithSequence(int *ptr, size_t size);