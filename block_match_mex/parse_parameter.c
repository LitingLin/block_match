#include "common.h"
#include <string.h>


enum LibBlockMatchMexError getString(const mxArray *pa, char *buffer, int bufferLength)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId != mxCHAR_CLASS)
		return blockMatchMexErrorTypeOfArgument;

	mxGetNChars(pa, buffer, bufferLength);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseRetain(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int retain;
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else if (classId == mxDOUBLE_CLASS)
		goto ValueClass;
	else
		return blockMatchMexErrorTypeOfArgument;

StringClass:

	char buffer[4];
	getString(pa, buffer, 4);
	if (strncmp(buffer, "all", 4) != 0)
		return blockMatchMexErrorInvalidValue;

	context->retain = 0;

	return blockMatchMexOk;
ValueClass:
	
	size_t numberOfDimensions = mxGetNumberOfDimensions(pa);
	if (numberOfDimensions != 1 && numberOfDimensions != 2)
		return blockMatchMexErrorNumberOfMatrixDimension;

	retain = mxGetScalar(pa);

	if (retain <= 0)
		return blockMatchMexErrorInvalidValue;

	context->retain = retain;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSort(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	bool sort;

	if (mxGetClassID(pa) != mxLOGICAL_CLASS)
		return blockMatchMexErrorTypeOfArgument;

	mxLogical* logicals = mxGetLogicals(pa);
	sort = logicals[0];

	if (sort == false)
		return blockMatchMexErrorNotImplemented;

	context->sort = sort;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceAPaddingSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int sequenceAPaddingWidth, sequenceAPaddingHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		sequenceAPaddingWidth = sequenceAPaddingHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		sequenceAPaddingHeight = pr[0];
		sequenceAPaddingWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceAPaddingWidth = sequenceAPaddingWidth;
	context->sequenceAPaddingHeight = sequenceAPaddingHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceBPaddingSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int sequenceBPaddingWidth, sequenceBPaddingHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		sequenceBPaddingWidth = sequenceBPaddingHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		sequenceBPaddingHeight = pr[0];
		sequenceBPaddingWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceBPaddingWidth = sequenceBPaddingWidth;
	context->sequenceBPaddingHeight = sequenceBPaddingHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceAStrideSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int sequenceAStrideWidth, sequenceAStrideHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		sequenceAStrideWidth = sequenceAStrideHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		sequenceAStrideHeight = pr[0];
		sequenceAStrideWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	if (sequenceAStrideHeight == 0 || sequenceAStrideWidth == 0)
		return blockMatchMexErrorInvalidValue;

	context->sequenceAStrideWidth = sequenceAStrideWidth;
	context->sequenceAStrideHeight = sequenceAStrideHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceBStrideSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int sequenceBStrideWidth, sequenceBStrideHeight;

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

	if (sequenceBStrideHeight == 0 || sequenceBStrideWidth == 0)
		return blockMatchMexErrorInvalidValue;

	context->sequenceBStrideWidth = sequenceBStrideWidth;
	context->sequenceBStrideHeight = sequenceBStrideHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSearchRegion(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int searchRegionWidth, searchRegionHeight;

	char buffer[6];

	if (getString(pa, buffer, 6) == blockMatchMexOk && strncmp(buffer, "full", 6) == 0)
		return blockMatchMexOk;
	else
		return blockMatchMexErrorNotImplemented;

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
		return blockMatchMexErrorSizeOfMatrix;
	}
	context->searchRegionWidth = searchRegionWidth;
	context->searchRegionHeight = searchRegionHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseBlockSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int blockWidth, blockHeight;
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
		return blockMatchMexErrorSizeOfMatrix;
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

	int sequenceBMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);

	if (sequenceBMatrixNumberOfDimensions != context->sequenceMatrixNumberOfDimensions)
	{
		return blockMatchMexErrorNumberOfMatrixDimensionMismatch;
	}

	const size_t *sequenceBMatrixDimensions = mxGetDimensions(pa);
	context->sequenceBMatrixDimensions[0] = sequenceBMatrixDimensions[0];
	context->sequenceBMatrixDimensions[1] = sequenceBMatrixDimensions[1];

	if (sequenceBMatrixDimensions[0] < context->sequenceAMatrixDimensions[0] || sequenceBMatrixDimensions[1] < context->sequenceAMatrixDimensions[1])
	{
		return blockMatchMexErrorSizeOfMatrix;
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

	int sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceMatrixNumberOfDimensions = sequenceAMatrixNumberOfDimensions;

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	context->sequenceAMatrixDimensions[0] = sequenceAMatrixDimensions[0];
	context->sequenceAMatrixDimensions[1] = sequenceAMatrixDimensions[1];

	context->sequenceAMatrixPointer = mxGetData(pa);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseMethod(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	enum LibBlockMatchMexError error;
	char buffer[4];
	error = getString(pa, buffer, 4);
	if (error != blockMatchMexOk)
		return error;

	if (strncmp(buffer, "mse", 4) == 0)
		context->method = MSE;
	else if (strncmp(buffer, "cc", 4) == 0)
		context->method = CC;
	else
		return blockMatchMexErrorInvalidValue;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseOutputArgument(struct LibBlockMatchMexContext *context, 
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return blockMatchMexErrorNumberOfArguments;

	return blockMatchMexOk;
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
	else if (error == blockMatchMexErrorSizeOfMatrix)
	{
		return generateErrorMessage(error, "BlockSize cannot be smaller then Matrix A\n");
	}

	++index;

	if ((nrhs - index) % 2)
	{
		return generateErrorMessage(blockMatchMexErrorNumberOfArguments, "Wrong number of input arguments\n");
	}

	while (index < nrhs)
	{
		char buffer[LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH];
		char messageBuffer[LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH];
		error = getString(prhs[index], buffer, LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH);
		if (error == blockMatchMexErrorTypeOfArgument)
		{
			sprintf_s(messageBuffer, LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %d should be the name of parameter(string)\n", index);
			return generateErrorMessage(error, messageBuffer);
		}

		++index;

		if (strncmp(buffer, "SearchRegion", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSearchRegion(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SearchRegionSize must be 2\n");
			}
			else if (error == blockMatchMexErrorSizeOfMatrix)
			{
				return generateErrorMessage(error, "SearchRegionSize cannot be smaller then the size of Matrix A - BlockSize\n");
			}
			else if (error == blockMatchMexErrorNotImplemented)
				goto NotImplemented;
		}

		else if (strncmp(buffer, "SequenceAStride", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAStrideSize(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceAStrideSize must be 2\n");
			}
			else if (error == blockMatchMexErrorInvalidValue)
			{
				return generateErrorMessage(error, "SequenceAStride can not be zero\n");
			}
		}
		else if (strncmp(buffer, "SequenceBStride", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBStrideSize(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBStrideSize must be 2\n");
			}
			else if (error == blockMatchMexErrorInvalidValue)
			{
				return generateErrorMessage(error, "SequenceBStride can not be zero\n");
			}
		}
		else if (strncmp(buffer, "SequenceAPadding", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingSize(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceAPadding must be 2\n");
			}
		}
		else if (strncmp(buffer, "SequenceBPadding", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingSize(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBPadding must be 2\n");
			}
		}
		else if (strncmp(buffer, "MeasureMethod", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseMethod(context, prhs[index]);
			if (error == blockMatchMexErrorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument MeasureMethod must be string\n");
			}
			else if (error == blockMatchMexErrorInvalidValue)
			{
				return generateErrorMessage(error, "Invalid value of argument MeasureMethod\n");
			}
		}
		else if (strncmp(buffer, "SequenceAPaddingMethod", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0); // TODO
		else if (strncmp(buffer, "SequenceBPaddingMethod", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "Threshold", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "Sort", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSort(context, prhs[index]);
			if (error == blockMatchMexErrorNotImplemented)
			{
				return generateErrorMessage(error, "Argument Sort must be true(false Not implemented yet)\n");
			}
			else if (error == blockMatchMexErrorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument Sort must be logical(boolean)\n");
			}
		}
		else if (strncmp(buffer, "Retain", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseRetain(context, prhs[index]);
			if (error == blockMatchMexErrorNumberOfMatrixDimension || error == blockMatchMexErrorSizeOfMatrix)
				return generateErrorMessage(error, "Argument Retain must be scalar\n");
			else if (error == blockMatchMexErrorTypeOfArgument)
				return generateErrorMessage(error, "Argument Retain must be integer or string\n");
			else if (error == blockMatchMexErrorInvalidValue)
				return generateErrorMessage(error, "Value of argument Retain must be 'all' or positive integer\n");
		}
		else if (strncmp(buffer, "ResultDataType", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "IntermediateDataType", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "Sparse", LIB_BLOCK_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);

		else
		{
			NotImplemented:
			sprintf_s(messageBuffer, LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %s has not implemented yet, or has wrong name\n", buffer);
			return generateErrorMessage(blockMatchMexErrorNotImplemented, messageBuffer);
		}
		++index;
	}

	struct LibBlockMatchMexErrorWithMessage error_message = { 0 };
	
	return error_message;
}