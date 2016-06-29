#include "common.h"
#include <string.h>

enum LibBlockMatchMexError
{
	blockMatchMexOk = 0,
	blockMatchMexErrorNumberOfArguments,
	blockMatchMexErrorTypeOfArgument,
	blockMatchMexErrorNumberOfMatrixDimension,
	blockMatchMexErrorNumberOfMatrixDimensionMismatch,
	blockMatchMexErrorSizeOfMatrixDimension,
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
	size_t sequenceMatrixNumberOfDimensions;

	size_t sequenceAMatrixDimensions[4];
	double *sequenceAMatrixPointer;

	size_t sequenceBMatrixDimensions[4];
	double *sequenceBMatrixPointer;

	size_t blockWidth;
	size_t blockHeight;
	size_t searchRegionWidth;
	size_t searchRegionHeight;

	size_t sequenceBStrideWidth;
	size_t sequenceBStrideHeight;
};

enum LibBlockMatchMexError parseSequenceBStrideSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t sequenceBStrideWidth, sequenceBStrideHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		sequenceBStrideWidth = sequenceBStrideHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		sequenceBStrideHeight = pr[0];
		sequenceBStrideWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceBStrideWidth = sequenceBStrideWidth;
	context->sequenceBStrideHeight = sequenceBStrideHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSearchRegionSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t searchRegionWidth, searchRegionHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		searchRegionWidth = searchRegionHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		searchRegionHeight = pr[0];
		searchRegionWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	if (context->sequenceAMatrixDimensions[0] - context->blockWidth < searchRegionWidth || context->sequenceAMatrixDimensions[1] - context->blockHeight < searchRegionHeight)
	{
		return blockMatchMexErrorSizeOfMatrixDimension;
	}
	context->searchRegionWidth = searchRegionWidth;
	context->searchRegionHeight = searchRegionHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseBlockSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t blockWidth, blockHeight;
	if (mxGetNumberOfElements(pa) == 1)
	{
		blockWidth = blockHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		blockHeight = pr[0];
		blockWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	if (context->sequenceAMatrixDimensions[0] < blockWidth || context->sequenceAMatrixDimensions[1] < blockHeight)
	{
		return blockMatchMexErrorSizeOfMatrixDimension;
	}
	
	context->blockWidth = blockWidth;
	context->blockHeight = blockHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceB(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return blockMatchMexErrorTypeOfArgument;
	}

	size_t sequenceBMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);

	if (sequenceBMatrixNumberOfDimensions != context->sequenceMatrixNumberOfDimensions)
	{
		return blockMatchMexErrorNumberOfMatrixDimensionMismatch;
	}

	const size_t *sequenceBMatrixDimensions = mxGetDimensions(pa);
	context->sequenceBMatrixDimensions[1] = sequenceBMatrixDimensions[0];
	context->sequenceBMatrixDimensions[0] = sequenceBMatrixDimensions[1];

	if (sequenceBMatrixDimensions[0] < context->sequenceAMatrixDimensions[0] || sequenceBMatrixDimensions[1] < context->sequenceAMatrixDimensions[1])
	{
		return blockMatchMexErrorSizeOfMatrixDimension;
	}

	context->sequenceBMatrixPointer = mxGetData(pa);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceA(struct LibBlockMatchMexContext *context, 
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return blockMatchMexErrorTypeOfArgument;
	}

	size_t sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceMatrixNumberOfDimensions = sequenceAMatrixNumberOfDimensions;

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	context->sequenceAMatrixDimensions[1] = sequenceAMatrixDimensions[0];
	context->sequenceAMatrixDimensions[0] = sequenceAMatrixDimensions[1];

	context->sequenceAMatrixPointer = mxGetData(pa);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseOutputArgument(struct LibBlockMatchMexContext *context, 
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return blockMatchMexErrorNumberOfArguments;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError getString(const mxArray *pa, char *buffer, size_t bufferLength)
{
	if (mxGetString(pa, buffer, bufferLength) == 0)
		return blockMatchMexErrorTypeOfArgument;

	return blockMatchMexOk;
}

inline
struct LibBlockMatchMexErrorWithMessage generateErrorMessage(enum LibBlockMatchMexError error, char message[LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH])
{
	struct LibBlockMatchMexErrorWithMessage error_with_message = { error, "" };
	strncpy_s(error_with_message.message, LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH, message, LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH);
	return error_with_message;
}

struct LibBlockMatchMexErrorWithMessage parseParameter(struct LibBlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	enum LibBlockMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == blockMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Too many output arguments\n");

	if (nrhs < 3)
		return generateErrorMessage(blockMatchMexErrorNumberOfArguments, "Too few input arguments\n");

	int index = 0;

	error = parseSequenceA(context, prhs[index]);
	if (error == blockMatchMexErrorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix A must be double\n");
	}
	else if (error == blockMatchMexErrorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix A must be 2\n");
	}

	++index;

	error = parseSequenceB(context, prhs[index]);
	if (error == blockMatchMexErrorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix B must be double\n");
	}
	else if (error == blockMatchMexErrorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be 2\n");
	}
	else if (error == blockMatchMexErrorNumberOfMatrixDimensionMismatch)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be the same as Matrix A\n");
	}

	++index;

	error = parseBlockSize(context, prhs[index]);
	if (error == blockMatchMexErrorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of BlockSize must be 1 or 2\n");
	}
	else if (error == blockMatchMexErrorSizeOfMatrixDimension)
	{
		return generateErrorMessage(error, "BlockSize cannot be smaller then Matrix A\n");
	}

	if ((nrhs - index) % 2)
	{
		return generateErrorMessage(error, "Wrong number of input arguments");
	}

	while (index < nrhs)
	{
		char buffer[LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH];
		error = getString(prhs[index], buffer, LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH);
		if (error == blockMatchMexErrorTypeOfArgument)
		{
			sprintf_s(buffer, LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH,
				"Argument %d should be the name of parameter(string)\n", index);
			return generateErrorMessage(error, buffer);
		}

		++index;

		if (strncmp(buffer, "SearchRegionSize", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSearchRegionSize(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SearchRegionSize must be 2\n");
			}
			else if (error == blockMatchMexErrorSizeOfMatrixDimension)
			{
				return generateErrorMessage(error, "SearchRegionSize cannot be smaller then the size of Matrix A - BlockSize");
			}
		}

		else if (strncmp(buffer, "SequenceBStrideSize", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBStrideSize(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBStrideSize must be 2\n");
			}
		}
	}

	struct LibBlockMatchMexErrorWithMessage error_message = { 0 };
	
	return error_message;
}