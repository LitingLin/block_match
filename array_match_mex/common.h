#pragma once

#include <block_match.h>

#include <mex.h>

enum ArrayMatchMexError
{
	arrayMatchMexErrorInternal,
	arrayMatchMexErrorOverFlow,
	arrayMatchMexErrorTypeOfArgument,
	arrayMatchMexErrorNumberOfArguments,
	arrayMatchMexErrorNumberOfMatrixDimensions,
	arrayMatchMexErrorSizeOfMatrixMismatch,
	arrayMatchMexErrorInvalidValue,
	arrayMatchMexOk
};

#define ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH 128
#define ARRAY_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH 128

struct ArrayMatchMexErrorWithMessage
{
	enum ArrayMatchMexError error;
	char message[ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH];
};

struct ArrayMatchMexErrorWithMessage generateErrorMessage(enum ArrayMatchMexError error, 
	char message[ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH], ...);

struct ArrayMatchMexContext
{
	enum Method method;
	int numberOfArray;
	int lengthOfArray;

	double *A;
	double *B;
};

struct ArrayMatchMexErrorWithMessage parseParameter(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);


void convertDoubleToFloat(const double *source, float *destination, size_t size);
void convertFloatToDouble(const float *source, double *destination, size_t size);

size_t getMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArray);

void logging_function(const char* msg);