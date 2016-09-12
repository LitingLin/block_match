#include "common.h"
#include <string.h>

enum LibMatchMexError parseRetain(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int retain;
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else if (classId == mxDOUBLE_CLASS)
		goto ValueClass;
	else
		return libMatchMexErrorTypeOfArgument;

StringClass:

	char buffer[4];
	getStringFromMxArray(pa, buffer, 4);
	if (strncmp(buffer, "all", 4) != 0)
		return libMatchMexErrorInvalidValue;

	context->retain = 0;

	return libMatchMexOk;
ValueClass:
	
	size_t numberOfDimensions = mxGetNumberOfDimensions(pa);
	if (numberOfDimensions != 1 && numberOfDimensions != 2)
		return libMatchMexErrorNumberOfMatrixDimension;

	retain = mxGetScalar(pa);

	if (retain <= 0)
		return libMatchMexErrorInvalidValue;

	context->retain = retain;

	return libMatchMexOk;
}

enum LibMatchMexError parseSort(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	bool sort;

	if (mxGetClassID(pa) != mxLOGICAL_CLASS)
		return libMatchMexErrorTypeOfArgument;

	mxLogical* logicals = mxGetLogicals(pa);
	sort = logicals[0];

	if (sort == false)
		return libMatchMexErrorNotImplemented;

	context->sort = sort;

	return libMatchMexOk;
}

enum LibMatchMexError parseSequenceAPaddingSize(struct LibBlockMatchMexContext *context,
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
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceAPaddingWidth = sequenceAPaddingWidth;
	context->sequenceAPaddingHeight = sequenceAPaddingHeight;

	return libMatchMexOk;
}

enum LibMatchMexError parseSequenceBPaddingSize(struct LibBlockMatchMexContext *context,
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
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceBPaddingWidth = sequenceBPaddingWidth;
	context->sequenceBPaddingHeight = sequenceBPaddingHeight;

	return libMatchMexOk;
}

enum LibMatchMexError parseSequenceAStrideSize(struct LibBlockMatchMexContext *context,
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
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	if (sequenceAStrideHeight == 0 || sequenceAStrideWidth == 0)
		return libMatchMexErrorInvalidValue;

	context->sequenceAStrideWidth = sequenceAStrideWidth;
	context->sequenceAStrideHeight = sequenceAStrideHeight;

	return libMatchMexOk;
}

enum LibMatchMexError parseSequenceBStrideSize(struct LibBlockMatchMexContext *context,
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
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	if (sequenceBStrideHeight == 0 || sequenceBStrideWidth == 0)
		return libMatchMexErrorInvalidValue;

	context->sequenceBStrideWidth = sequenceBStrideWidth;
	context->sequenceBStrideHeight = sequenceBStrideHeight;

	return libMatchMexOk;
}

// SearchRegion size 0 for full search
enum LibMatchMexError parseSearchRegion(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	int searchRegionWidth, searchRegionHeight;

	char buffer[6];

	if (getStringFromMxArray(pa, buffer, 6) == libMatchMexOk)
	{
		if (strncmp(buffer, "full", 6) != 0)
			return libMatchMexErrorInvalidValue;

		searchRegionWidth = searchRegionHeight = 0;
	}
	else
	{
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
			return libMatchMexErrorNumberOfMatrixDimension;
		}

		if (context->sequenceAMatrixDimensions[0] - context->blockWidth < searchRegionWidth || context->sequenceAMatrixDimensions[1] - context->blockHeight < searchRegionHeight)
		{
			return libMatchMexErrorSizeOfMatrix;
		}
	}

	context->searchRegionWidth = searchRegionWidth;
	context->searchRegionHeight = searchRegionHeight;

	return libMatchMexOk;
}

enum LibMatchMexError parseBlockSize(struct LibBlockMatchMexContext *context,
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
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	if (context->sequenceAMatrixDimensions[0] < blockWidth || context->sequenceAMatrixDimensions[1] < blockHeight)
	{
		return libMatchMexErrorSizeOfMatrix;
	}
	
	context->blockWidth = blockWidth;
	context->blockHeight = blockHeight;

	return libMatchMexOk;
}

enum LibMatchMexError parseSequenceB(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return libMatchMexErrorTypeOfArgument;
	}

	int sequenceBMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);

	if (sequenceBMatrixNumberOfDimensions != context->sequenceMatrixNumberOfDimensions)
	{
		return libMatchMexErrorNumberOfMatrixDimensionMismatch;
	}

	const size_t *sequenceBMatrixDimensions = mxGetDimensions(pa);
	context->sequenceBMatrixDimensions[0] = sequenceBMatrixDimensions[0];
	context->sequenceBMatrixDimensions[1] = sequenceBMatrixDimensions[1];

	if (sequenceBMatrixDimensions[0] < context->sequenceAMatrixDimensions[0] || sequenceBMatrixDimensions[1] < context->sequenceAMatrixDimensions[1])
	{
		return libMatchMexErrorSizeOfMatrix;
	}

	context->sequenceBMatrixPointer = mxGetData(pa);

	return libMatchMexOk;
}

enum LibMatchMexError parseSequenceA(struct LibBlockMatchMexContext *context, 
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return libMatchMexErrorTypeOfArgument;
	}

	int sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceMatrixNumberOfDimensions = sequenceAMatrixNumberOfDimensions;

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	context->sequenceAMatrixDimensions[0] = sequenceAMatrixDimensions[0];
	context->sequenceAMatrixDimensions[1] = sequenceAMatrixDimensions[1];

	context->sequenceAMatrixPointer = mxGetData(pa);

	return libMatchMexOk;
}

enum LibMatchMexError parseMethod(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	enum LibMatchMexError error;
	char buffer[4];
	error = getStringFromMxArray(pa, buffer, 4);
	if (error != libMatchMexOk)
		return error;

	if (strncmp(buffer, "mse", 4) == 0)
		context->method = LIB_MATCH_MSE;
	else if (strncmp(buffer, "cc", 4) == 0)
		context->method = LIB_MATCH_CC;
	else
		return libMatchMexErrorInvalidValue;

	return libMatchMexOk;
}

enum LibMatchMexError parseOutputArgument(struct LibBlockMatchMexContext *context, 
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return libMatchMexErrorNumberOfArguments;

	return libMatchMexOk;
}


struct LibMatchMexErrorWithMessage parseParameter(struct LibBlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	enum LibMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == libMatchMexErrorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments\n");

	if (nrhs < 3)
		return generateErrorMessage(libMatchMexErrorNumberOfArguments, "Too few input arguments\n");

	int index = 0;

	error = parseSequenceA(context, prhs[index]);
	if (error == libMatchMexErrorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix A must be double\n");
	}
	else if (error == libMatchMexErrorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix A must be 2\n");
	}

	++index;

	error = parseSequenceB(context, prhs[index]);
	if (error == libMatchMexErrorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix B must be double\n");
	}
	else if (error == libMatchMexErrorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be 2\n");
	}
	else if (error == libMatchMexErrorNumberOfMatrixDimensionMismatch)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be the same as Matrix A\n");
	}

	++index;

	error = parseBlockSize(context, prhs[index]);
	if (error == libMatchMexErrorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of BlockSize must be 1 or 2\n");
	}
	else if (error == libMatchMexErrorSizeOfMatrix)
	{
		return generateErrorMessage(error, "BlockSize cannot be smaller then Matrix A\n");
	}

	++index;

	if ((nrhs - index) % 2)
	{
		return generateErrorMessage(libMatchMexErrorNumberOfArguments, "Wrong number of input arguments\n");
	}

	while (index < nrhs)
	{
		char buffer[LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH];
		char messageBuffer[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH];
		error = getStringFromMxArray(prhs[index], buffer, LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH);
		if (error == libMatchMexErrorTypeOfArgument)
		{
			sprintf_s(messageBuffer, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %d should be the name of parameter(string)\n", index);
			return generateErrorMessage(error, messageBuffer);
		}

		++index;

		if (strncmp(buffer, "SearchRegion", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSearchRegion(context, prhs[index]);
			if (error == libMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SearchRegionSize must be 2\n");
			}
			else if (error == libMatchMexErrorSizeOfMatrix)
			{
				return generateErrorMessage(error, "SearchRegionSize cannot be smaller then the size of Matrix A - BlockSize\n");
			}
			else if (error == libMatchMexErrorNotImplemented)
				goto NotImplemented;
		}

		else if (strncmp(buffer, "SequenceAStride", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAStrideSize(context, prhs[index]);
			if (error == libMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceAStrideSize must be 2\n");
			}
			else if (error == libMatchMexErrorInvalidValue)
			{
				return generateErrorMessage(error, "SequenceAStride can not be zero\n");
			}
		}
		else if (strncmp(buffer, "SequenceBStride", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBStrideSize(context, prhs[index]);
			if (error == libMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBStrideSize must be 2\n");
			}
			else if (error == libMatchMexErrorInvalidValue)
			{
				return generateErrorMessage(error, "SequenceBStride can not be zero\n");
			}
		}
		else if (strncmp(buffer, "SequenceAPadding", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingSize(context, prhs[index]);
			if (error == libMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceAPadding must be 2\n");
			}
		}
		else if (strncmp(buffer, "SequenceBPadding", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingSize(context, prhs[index]);
			if (error == libMatchMexErrorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBPadding must be 2\n");
			}
		}
		else if (strncmp(buffer, "MeasureMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseMethod(context, prhs[index]);
			if (error == libMatchMexErrorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument MeasureMethod must be string\n");
			}
			else if (error == libMatchMexErrorInvalidValue)
			{
				return generateErrorMessage(error, "Invalid value of argument MeasureMethod\n");
			}
		}
		else if (strncmp(buffer, "SequenceAPaddingMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0); // TODO
		else if (strncmp(buffer, "SequenceBPaddingMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "Threshold", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "Sort", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSort(context, prhs[index]);
			if (error == libMatchMexErrorNotImplemented)
			{
				return generateErrorMessage(error, "Argument Sort must be true(false Not implemented yet)\n");
			}
			else if (error == libMatchMexErrorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument Sort must be logical(boolean)\n");
			}
		}
		else if (strncmp(buffer, "Retain", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseRetain(context, prhs[index]);
			if (error == libMatchMexErrorNumberOfMatrixDimension || error == libMatchMexErrorSizeOfMatrix)
				return generateErrorMessage(error, "Argument Retain must be scalar\n");
			else if (error == libMatchMexErrorTypeOfArgument)
				return generateErrorMessage(error, "Argument Retain must be integer or string\n");
			else if (error == libMatchMexErrorInvalidValue)
				return generateErrorMessage(error, "Value of argument Retain must be 'all' or positive integer\n");
		}
		else if (strncmp(buffer, "ResultDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "IntermediateDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "Sparse", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);

		else
		{
			NotImplemented:
			sprintf_s(messageBuffer, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %s has not implemented yet, or has wrong name\n", buffer);
			return generateErrorMessage(libMatchMexErrorNotImplemented, messageBuffer);
		}
		++index;
	}

	struct LibMatchMexErrorWithMessage error_message = { 0 };
	
	return error_message;
}