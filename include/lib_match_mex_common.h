#pragma once

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define DLL_EXPORT_SYM __attribute__ ((dllexport))
#else
#define DLL_EXPORT_SYM __declspec(dllexport)
#endif
#else
#define DLL_EXPORT_SYM __attribute__ ((visibility ("default")))
#endif

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
/*
void convertArrayType(std::type_index inType, std::type_index outType, const void *in, void *out, size_t size);

template <typename Type1, typename Type2, typename std::enable_if<!std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n);
template <typename Type1, typename Type2, typename std::enable_if<std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n);
*/
/* 
 * Return:
 * errorTypeOfArgument
 * errorSizeOfArray
 * success
 */
LibMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength);

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementPositiveIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementNonNegativeIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  success
*/
LibMatchMexError parse4ElementIntegerParameter(const mxArray *pa,
	int *parameterA1, int *parameterA2,
	int *parameterB1, int *parameterB2);

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  success
*/
LibMatchMexError parse4ElementNonNegativeIntegerParameter(const mxArray *pa,
	int *parameterA1, int *parameterA2,
	int *parameterB1, int *parameterB2);

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

/*
* Return:
*  success,
*  errorOverFlow,
*  errorTypeOfArgument
*/
LibMatchMexError getFourIntegerFromMxArray(const mxArray *pa,
	int *integerA1, int *integerA2,
	int *integerB1, int *integerB2);