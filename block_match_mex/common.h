#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <mex.h>

#include <block_match.h>

enum LibBlockMatchMexError
{
	blockMatchMexOk = 0,
	blockMatchMexErrorNumberOfArguments,
	blockMatchMexErrorTypeOfArgument,
	blockMatchMexErrorNumberOfMatrixDimension,
	blockMatchMexErrorNumberOfMatrixDimensionMismatch,
	blockMatchMexErrorSizeOfMatrix,
	blockMatchMexErrorInvalidValue,
	blockMatchMexErrorNotImplemented
};

#define LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH 128
#define LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH 128

struct LibBlockMatchMexErrorWithMessage
{
	enum LibBlockMatchMexError error;
	char message[LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH];
};

struct LibBlockMatchMexContext
{
	enum Method method;
	int sequenceMatrixNumberOfDimensions;

	int sequenceAMatrixDimensions[4];
	double *sequenceAMatrixPointer;

	int sequenceBMatrixDimensions[4];
	double *sequenceBMatrixPointer;

	int blockWidth;
	int blockHeight;
	int searchRegionWidth;
	int searchRegionHeight;

	int sequenceAStrideWidth;
	int sequenceAStrideHeight;
	int sequenceBStrideWidth;
	int sequenceBStrideHeight;

	int sequenceAPaddingWidth;
	int sequenceAPaddingHeight;
	int sequenceBPaddingWidth;
	int sequenceBPaddingHeight;

	bool sort;
	int retain;
};

struct LibBlockMatchMexErrorWithMessage generateErrorMessage(enum LibBlockMatchMexError error, char message[LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH]);

struct LibBlockMatchMexErrorWithMessage parseParameter(struct LibBlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

struct LibBlockMatchMexErrorWithMessage validateParameter(struct LibBlockMatchMexContext *context);

bool generate_result(struct LibBlockMatchMexContext *context, mxArray **_pa, const int *index, const float *value);