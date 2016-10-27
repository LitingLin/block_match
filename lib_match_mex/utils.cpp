#include "lib_match_mex_common.h"
#include <stdarg.h>
#include <memory>
#include <typeindex>
#include <stdint.h>

LibMatchMexErrorWithMessage generateErrorMessage(LibMatchMexError error, char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH], ...)
{
	LibMatchMexErrorWithMessage errorWithMessage = { error, "" };
	va_list args;
	va_start(args, message);
	vsnprintf(errorWithMessage.message, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, message, args);
	va_end(args);
	return errorWithMessage;
}

template <typename Type1, typename Type2, typename std::enable_if<!std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n)
{
	for (size_t i = 0; i<n; i++)
	{
		out[i] = in[i];
	}
}

template <typename Type1, typename Type2, typename std::enable_if<std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n)
{
	memcpy(out, in, n * sizeof(Type1));
}

LibMatchMexErrorWithMessage internalErrorMessage()
{
	return generateErrorMessage(LibMatchMexError::errorInternal, "Unknown internal error.");
}

template
void convertArrayType(const float*, double *, size_t);
template
void convertArrayType(const double*, float *, size_t);
template
void convertArrayType(const int*, double *, size_t);
template
void convertArrayType(const int*, float *, size_t);
template
void convertArrayType(const float*, float *, size_t);
template
void convertArrayType(const double*, double *, size_t);


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

template <typename OriginType, typename DestinationType>
LibMatchMexError typeConvertWithNumericLimitsCheck(const OriginType *originValue, DestinationType *destinationValue)
{
	OriginType value = *originValue;
	if (value > std::numeric_limits<DestinationType>::max())
		return LibMatchMexError::errorOverFlow;

	*destinationValue = value;

	return LibMatchMexError::success;
}

/*
* Return:
*  success,
*  errorOverFlow,
*  errorTypeOfArgument
*/
LibMatchMexError getIntegerFromMxArray(const mxArray *pa, int *integer)
{
	mxClassID classId = mxGetClassID(pa);
	const void *data = mxGetData(pa);
	switch (classId)
	{
	case mxLOGICAL_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const mxLogical*>(data), integer);
	case mxCHAR_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const mxChar*>(data), integer);
	case mxDOUBLE_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const double*>(data), integer);
	case mxSINGLE_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const float*>(data), integer);
	case mxINT8_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const int8_t*>(data), integer);
	case mxUINT8_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const uint8_t*>(data), integer);
	case mxINT16_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const int16_t*>(data), integer);
	case mxUINT16_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const uint16_t*>(data), integer);
	case mxINT32_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const int32_t*>(data), integer);
	case mxUINT32_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const uint32_t*>(data), integer);
	case mxINT64_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const int64_t*>(data), integer);
	case mxUINT64_CLASS:
		return typeConvertWithNumericLimitsCheck(static_cast<const uint64_t*>(data), integer);
	default:
		return LibMatchMexError::errorTypeOfArgument;
	}
}


LibMatchMexError getTwoIntegerFromMxArray(const mxArray *pa,
	int *integerA, int *integerB)
{
	mxClassID classId = mxGetClassID(pa);
	void *data = mxGetData(pa);
	LibMatchMexError error;
	switch (classId)
	{
	case mxLOGICAL_CLASS: {
		mxLogical* dataWithType = static_cast<mxLogical*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxCHAR_CLASS: {
		mxChar* dataWithType = static_cast<mxChar*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxDOUBLE_CLASS: {
		double* dataWithType = static_cast<double*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxSINGLE_CLASS: {
		float* dataWithType = static_cast<float*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxINT8_CLASS: {
		int8_t* dataWithType = static_cast<int8_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxUINT8_CLASS: {
		uint8_t* dataWithType = static_cast<uint8_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxINT16_CLASS: {
		int16_t* dataWithType = static_cast<int16_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxUINT16_CLASS: {
		uint16_t* dataWithType = static_cast<uint16_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxINT32_CLASS: {
		int32_t* dataWithType = static_cast<int32_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxUINT32_CLASS: {
		uint32_t* dataWithType = static_cast<uint32_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxINT64_CLASS: {
		int64_t* dataWithType = static_cast<int64_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	case mxUINT64_CLASS: {
		uint64_t* dataWithType = static_cast<uint64_t*>(data);
		error = typeConvertWithNumericLimitsCheck(dataWithType, integerA);
		if (error != LibMatchMexError::success)
			return error;
		return typeConvertWithNumericLimitsCheck(dataWithType + 1, integerB);
	}
	default:
		return LibMatchMexError::errorTypeOfArgument;
	}
}