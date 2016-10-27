#pragma once

#include <mex.h>

#include <lib_match.h>
#include <type_traits>

namespace std {
	class type_index;
}

enum class LibMatchMexError
{
	success = 0,
	errorNumberOfArguments,
	errorTypeOfArgument,
	errorNumberOfMatrixDimension,
	errorNumberOfMatrixDimensionMismatch,
	errorSizeOfMatrixMismatch,
	errorSizeOfArray,
	errorInvalidValue,
	errorNotImplemented,
	errorOverFlow,
	errorInvalidParameterCombination,
	errorInternal
};

#define LIB_MATCH_MEX_MAX_MESSAGE_LENGTH 128
#define LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH 128

struct LibMatchMexErrorWithMessage
{
	LibMatchMexError error;
	char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH];
};

void libMatchMexInitalize();

// Utilities

struct LibMatchMexErrorWithMessage generateErrorMessage(LibMatchMexError error, char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH], ...);
struct LibMatchMexErrorWithMessage internalErrorMessage();

void convertArrayType(std::type_index inType, std::type_index outType, const void *in, void *out, size_t size);

template <typename Type1, typename Type2, typename std::enable_if<!std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n);
template <typename Type1, typename Type2, typename std::enable_if<std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n);

/* 
 * Return:
 * errorTypeOfArgument
 * errorSizeOfArray
 * success
 */
LibMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength);

/*
* Return:
*  errorSizeOfArray
*  errorOverFlow
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);

/*
* Return:
*  errorSizeOfArray
*  errorOverFlow
*  errorInvalidValue
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementPositiveIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);

/*
* Return:
*  errorSizeOfArray
*  errorOverFlow
*  errorInvalidValue
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementNonNegativeIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);
/* 
* Return:
*  errorTypeOfArgument
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  success
*/
LibMatchMexError parse2DMatrixParameter(const mxArray *pa,
	void **pointer,
	int *size_M, int *size_N);


std::type_index getTypeIndex(mxClassID mxTypeId);
mxClassID getMxClassId(std::type_index type);

/*
* Return:
*  success,
*  errorOverFlow,
*  errorTypeOfArgument
*/
LibMatchMexError getIntegerFromMxArray(const mxArray *pa, int *integer);

/*
* Return:
*  success,
*  errorOverFlow,
*  errorTypeOfArgument
*/
LibMatchMexError getTwoIntegerFromMxArray(const mxArray *pa,
	int *integerA, int *integerB);