#pragma once

#include <mex.h>

#include <lib_match.h>

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

void convertArrayFromDoubleToFloat(const double *source, float *destination, size_t size);
void convertArrayFromFloatToDouble(const float *source, double *destination, size_t size);

/* Return:
 * errorTypeOfArgument
 * errorSizeOfArray
 * success
 */
LibMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength);

/* Return:
 * errorSizeOfArray
 * errorOverFlow
 * success
 */
LibMatchMexError parse2ElementIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB);

/* Return:
* errorTypeOfArgument
* errorNumberOfMatrixDimension
* errorOverFlow
* success
*/
LibMatchMexError parse2DMatrixParameter(const mxArray *pa,
	double **pointer,
	int *size_M, int *size_N);