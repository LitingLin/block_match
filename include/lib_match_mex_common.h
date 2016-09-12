#pragma once

#include <stdbool.h>
#include <mex.h>

#include <lib_match.h>

enum LibMatchMexError
{
	libMatchMexOk = 0,
	libMatchMexErrorNumberOfArguments,
	libMatchMexErrorTypeOfArgument,
	libMatchMexErrorNumberOfMatrixDimension,
	libMatchMexErrorNumberOfMatrixDimensionMismatch,
	libMatchMexErrorSizeOfMatrixMismatch,
	libMatchMexErrorSizeOfMatrix,
	libMatchMexErrorInvalidValue,
	libMatchMexErrorNotImplemented,
	libMatchMexErrorOverFlow,
	libMatchMexErrorInternal
};

#define LIB_MATCH_MEX_MAX_MESSAGE_LENGTH 128
#define LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH 128

struct LibMatchMexErrorWithMessage
{
	enum LibMatchMexError error;
	char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH];
};

void libMatchMexInitalize();

// Utilities

struct LibMatchMexErrorWithMessage generateErrorMessage(enum ArrayMatchMexError error, char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH], ...);

void convertArrayFromDoubleToFloat(const double *source, float *destination, size_t size);
void convertArrayFromFloatToDouble(const float *source, double *destination, size_t size);

enum LibMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength);