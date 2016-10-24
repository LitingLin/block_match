#include <typeindex>
#include <mex.h>
#include <stdint.h>
#include <memory>
#include <lib_match_mex_common.h>

template <typename Type1, typename Type2>
void convertArrayTypeAndPlusOne(const Type1 *in, Type2 *out, size_t n)
{
	for (size_t i = 0; i<n; i++)
	{
		out[i] = in[i] + 1;
	}
}
template
void convertArrayTypeAndPlusOne(const int *in, double *out, size_t n);
template
void convertArrayTypeAndPlusOne(const int *in, float *out, size_t n);

std::type_index getTypeIndex(mxClassID mxTypeId)
{
	switch (mxTypeId)
	{
	case mxDOUBLE_CLASS:
		return typeid(double);
	case mxSINGLE_CLASS:
		return typeid(float);
	case mxLOGICAL_CLASS:
		return typeid(bool);
	case mxCHAR_CLASS:
		return typeid(char);
	case mxINT8_CLASS:
		return typeid(int8_t);
	case mxUINT8_CLASS:
		return typeid(uint8_t);
	case mxINT16_CLASS:
		return typeid(int16_t);
	case mxUINT16_CLASS:
		return typeid(uint16_t);
	case mxINT32_CLASS:
		return typeid(int32_t);
	case mxUINT32_CLASS:
		return typeid(uint32_t);
	case mxINT64_CLASS:
		return typeid(int64_t);
	case mxUINT64_CLASS:
		return typeid(uint64_t);
	case mxUNKNOWN_CLASS:
	case mxCELL_CLASS:
	case mxSTRUCT_CLASS:
	case mxVOID_CLASS:
	case mxFUNCTION_CLASS:
	case mxOPAQUE_CLASS:
	case mxOBJECT_CLASS:
	default:
		return typeid(nullptr);
	}
}

mxClassID getMxClassId(std::type_index type)
{
	if (type == typeid(float))
		return mxSINGLE_CLASS;
	else if (type == typeid(double))
		return mxDOUBLE_CLASS;
	else
		abort();
}

void convertArrayType(std::type_index inType, std::type_index outType, const void *in, void *out, size_t size)
{
	if (inType == typeid(float) && outType == typeid(double))
	{
		convertArrayType((float*)in, (double*)out, size);
	}
	else if (inType == typeid(double) && outType == typeid(float))
	{
		convertArrayType((double*)in, (float*)out, size);
	}
	else if (inType == typeid(float) && outType == typeid(float))
	{
		memcpy(out, in, size * sizeof(float));
	}
	else if (inType == typeid(double) && outType == typeid(double))
	{
		memcpy(out, in, size * sizeof(double));
	}
	else
		abort();
}