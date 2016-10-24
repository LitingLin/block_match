#pragma once

#include <typeindex>
#include <mex.h>

void convertArrayType(std::type_index inType, std::type_index outType, const void *in, void *out, size_t size);
template <typename Type1, typename Type2>
void convertArrayTypeAndPlusOne(const Type1 *in, Type2 *out, size_t n);

std::type_index getTypeIndex(mxClassID mxTypeId);
mxClassID getMxClassId(std::type_index type);