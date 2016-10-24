#include "common.h"
#include <string.h>

void recheckDataType(struct BlockMatchMexContext *context)
{
	context->sourceType = typeid(double);
	if (context->intermediateType == typeid(nullptr))
		context->intermediateType = typeid(double);
	if (context->resultType == typeid(nullptr))
		context->resultType = typeid(double);
}

LibMatchMexError parseIntermediateType(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[7];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 7);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "float", 6) == 0)
		context->intermediateType = typeid(float);
	else if (strncmp(buffer, "double", 7) == 0)
		context->intermediateType = typeid(double);
	else if (strncmp(buffer, "same", 5) != 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseResultDataType(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[7];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 7);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "float", 6) == 0)
		context->resultType = typeid(float);
	else if (strncmp(buffer, "double", 7) == 0)
		context->resultType = typeid(double);
	else if (strncmp(buffer, "same", 5) != 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceAPaddingMethod(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[10];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 10);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "zero", 5) == 0)
		context->padMethodA = PadMethod::zero;
	else if (strncmp(buffer, "circular", 9) == 0)
		context->padMethodA = PadMethod::circular;
	else if (strncmp(buffer, "replicate", 10) == 0)
		context->padMethodA = PadMethod::replicate;
	else if (strncmp(buffer, "symmetric", 10) == 0)
		context->padMethodA = PadMethod::symmetric;
	else
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceBPaddingMethod(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[10];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 10);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "zero", 5) == 0)
		context->padMethodB = PadMethod::zero;
	else if (strncmp(buffer, "circular", 9) == 0)
		context->padMethodB = PadMethod::circular;
	else if (strncmp(buffer, "replicate", 10) == 0)
		context->padMethodB = PadMethod::replicate;
	else if (strncmp(buffer, "symmetric", 10) == 0)
		context->padMethodB = PadMethod::symmetric;
	else
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseRetain(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	int retain;
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else if (classId == mxDOUBLE_CLASS)
		goto ValueClass;
	else
		return LibMatchMexError::errorTypeOfArgument;

StringClass:
	{
		char buffer[4];

		getStringFromMxArray(pa, buffer, 4);
		if (strncmp(buffer, "all", 4) != 0)
			return LibMatchMexError::errorInvalidValue;

		context->retain = 0;

		return LibMatchMexError::success;
	}
ValueClass:
	{
		if (mxGetNumberOfElements(pa) != 1)
			return LibMatchMexError::errorSizeOfArray;

		retain = mxGetScalar(pa);

		if (retain <= 0)
			return LibMatchMexError::errorInvalidValue;

		context->retain = retain;

		return LibMatchMexError::success;
	}
}

LibMatchMexError parseSort(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	bool sort;

	if (mxGetClassID(pa) != mxLOGICAL_CLASS)
		return LibMatchMexError::errorTypeOfArgument;

	mxLogical* logicals = mxGetLogicals(pa);
	sort = logicals[0];

	if (sort == false)
		return LibMatchMexError::errorNotImplemented;

	context->sort = sort;

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceAPaddingSize(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2ElementIntegerParameter(pa, &context->sequenceAPaddingHeight, &context->sequenceAPaddingWidth);
}

LibMatchMexError parseSequenceBPaddingSize(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2ElementIntegerParameter(pa, &context->sequenceBPaddingHeight, &context->sequenceBPaddingWidth);
}

LibMatchMexError parseSequenceAStrideSize(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = parse2ElementIntegerParameter(pa, &context->sequenceAStrideHeight, &context->sequenceAStrideWidth);
	if (error == LibMatchMexError::success)
		if (context->sequenceAStrideHeight == 0 || context->sequenceAStrideWidth == 0)
			return LibMatchMexError::errorInvalidValue;

	return error;
}

LibMatchMexError parseSequenceBStrideSize(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = parse2ElementIntegerParameter(pa, &context->sequenceBStrideHeight, &context->sequenceBStrideWidth);
	if (error == LibMatchMexError::success)
		if (context->sequenceBStrideHeight == 0 || context->sequenceBStrideWidth == 0)
			return LibMatchMexError::errorInvalidValue;

	return error;
}

// SearchRegion size 0 for full search
LibMatchMexError parseSearchRegion(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	int searchRegionWidth, searchRegionHeight;

	char buffer[6];

	if (getStringFromMxArray(pa, buffer, 6) == LibMatchMexError::success)
	{
		if (strncmp(buffer, "full", 6) != 0)
			return LibMatchMexError::errorInvalidValue;

		searchRegionWidth = searchRegionHeight = 0;
		context->searchType = SearchType::global;
	}
	else
	{
		if (mxGetNumberOfElements(pa) == 1)
		{
			searchRegionWidth = searchRegionHeight = mxGetScalar(pa);
		}
		else if (mxGetNumberOfElements(pa) == 2)
		{
			const double *pr = static_cast<const double *>(mxGetData(pa));
			searchRegionHeight = pr[0];
			searchRegionWidth = pr[1];
		}
		else
		{
			return LibMatchMexError::errorNumberOfMatrixDimension;
		}

		if (context->sequenceAMatrixDimensions[0] - context->blockWidth < searchRegionWidth || context->sequenceAMatrixDimensions[1] - context->blockHeight < searchRegionHeight)
		{
			return LibMatchMexError::errorSizeOfArray;
		}
		context->searchType = SearchType::local;
	}

	context->searchRegionWidth = searchRegionWidth;
	context->searchRegionHeight = searchRegionHeight;

	return LibMatchMexError::success;
}

LibMatchMexError parseBlockSize(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	int blockWidth, blockHeight;
	if (mxGetNumberOfElements(pa) == 1)
	{
		blockWidth = blockHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = static_cast<const double *>(mxGetData(pa));
		blockHeight = pr[0];
		blockWidth = pr[1];
	}
	else
	{
		return LibMatchMexError::errorNumberOfMatrixDimension;
	}

	if (context->sequenceAMatrixDimensions[0] < blockWidth || context->sequenceAMatrixDimensions[1] < blockHeight)
	{
		return LibMatchMexError::errorSizeOfArray;
	}

	context->blockWidth = blockWidth;
	context->blockHeight = blockHeight;

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceB(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2DMatrixParameter(pa, &context->sequenceBMatrixPointer,
		&context->sequenceBMatrixDimensions[0], &context->sequenceBMatrixDimensions[1]);
}

// TODO: multi-dimension
LibMatchMexError parseSequenceA(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = parse2DMatrixParameter(pa, &context->sequenceAMatrixPointer,
		&context->sequenceAMatrixDimensions[0], &context->sequenceAMatrixDimensions[1]);

	context->sequenceMatrixNumberOfDimensions = 2;

	return error;
}

LibMatchMexError parseMethod(struct BlockMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error;
	char buffer[4];
	error = getStringFromMxArray(pa, buffer, 4);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "mse", 4) == 0)
		context->method = LibMatchMeasureMethod::mse;
	else if (strncmp(buffer, "cc", 4) == 0)
		context->method = LibMatchMeasureMethod::cc;
	else
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseOutputArgument(struct BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 4)
		return LibMatchMexError::errorNumberOfArguments;

	return LibMatchMexError::success;
}


struct LibMatchMexErrorWithMessage parseParameter(struct BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	LibMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == LibMatchMexError::errorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments\n");

	if (nrhs < 3)
		return generateErrorMessage(LibMatchMexError::errorNumberOfArguments, "Too few input arguments\n");

	int index = 0;

	error = parseSequenceA(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix A must be double\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix A must be 2\n");
	}

	++index;

	error = parseSequenceB(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix B must be double\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be 2\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimensionMismatch)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be the same as Matrix A\n");
	}

	++index;

	error = parseBlockSize(context, prhs[index]);
	if (error == LibMatchMexError::errorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of BlockSize must be 1 or 2\n");
	}
	else if (error == LibMatchMexError::errorSizeOfArray)
	{
		return generateErrorMessage(error, "BlockSize cannot be smaller then Matrix A\n");
	}

	++index;

	if ((nrhs - index) % 2)
	{
		return generateErrorMessage(LibMatchMexError::errorNumberOfArguments, "Wrong number of input arguments\n");
	}

	while (index < nrhs)
	{
		char buffer[LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH];
		char messageBuffer[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH];
		error = getStringFromMxArray(prhs[index], buffer, LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH);
		if (error == LibMatchMexError::errorTypeOfArgument)
		{
			sprintf_s(messageBuffer, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %d should be the name of parameter(string)\n", index);
			return generateErrorMessage(error, messageBuffer);
		}

		++index;

		if (strncmp(buffer, "SearchRegion", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSearchRegion(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SearchRegionSize must be 2\n");
			}
			else if (error == LibMatchMexError::errorSizeOfArray)
			{
				return generateErrorMessage(error, "SearchRegionSize cannot be smaller then the size of Matrix A - BlockSize\n");
			}
			else if (error == LibMatchMexError::errorNotImplemented)
				goto NotImplemented;
		}

		else if (strncmp(buffer, "SequenceAStride", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAStrideSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceAStrideSize must be 2\n");
			}
			else if (error == LibMatchMexError::errorInvalidValue)
			{
				return generateErrorMessage(error, "SequenceAStride can not be zero\n");
			}
		}
		else if (strncmp(buffer, "SequenceBStride", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBStrideSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBStrideSize must be 2\n");
			}
			else if (error == LibMatchMexError::errorInvalidValue)
			{
				return generateErrorMessage(error, "SequenceBStride can not be zero\n");
			}
		}
		else if (strncmp(buffer, "SequenceAPadding", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceAPadding must be 2\n");
			}
		}
		else if (strncmp(buffer, "SequenceBPadding", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
			{
				return generateErrorMessage(error, "Number of dimension of SequenceBPadding must be 2\n");
			}
		}
		else if (strncmp(buffer, "MeasureMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument MeasureMethod must be string\n");
			}
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
			{
				return generateErrorMessage(error, "Invalid value of argument MeasureMethod\n");
			}
		}
		else if (strncmp(buffer, "SequenceAPaddingMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument SequenceAPaddingMethod must be string\n");
			}
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
			{
				return generateErrorMessage(error, "Invalid value of argument SequenceAPaddingMethod\n");
			}
		}
		else if (strncmp(buffer, "SequenceBPaddingMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument SequenceBPaddingMethod must be string\n");
			}
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
			{
				return generateErrorMessage(error, "Invalid value of argument SequenceBPaddingMethod\n");
			}
		}
		else if (strncmp(buffer, "Threshold", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0); // TODO
		else if (strncmp(buffer, "Sort", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSort(context, prhs[index]);
			if (error == LibMatchMexError::errorNotImplemented)
			{
				return generateErrorMessage(error, "Argument Sort must be true(false Not implemented yet)\n");
			}
			else if (error == LibMatchMexError::errorTypeOfArgument)
			{
				return generateErrorMessage(error, "Argument Sort must be logical(boolean)\n");
			}
		}
		else if (strncmp(buffer, "Retain", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseRetain(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Argument Retain must be scalar or 'all'\n");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument Retain must be integer or string\n");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "Value of argument Retain must be 'all' or positive integer\n");
		}
		else if (strncmp(buffer, "ResultDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseResultDataType(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument ResultDataType must be string.\n");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument ResultDataType.\n");
		}
		else if (strncmp(buffer, "IntermediateDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseIntermediateType(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument IntermediateDataType must be string.\n");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument IntermediateDataType.\n");
		}
		else if (strncmp(buffer, "Sparse", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);

		else
		{
		NotImplemented:
			sprintf_s(messageBuffer, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %s has not implemented yet, or has wrong name\n", buffer);
			return generateErrorMessage(LibMatchMexError::errorNotImplemented, messageBuffer);
		}
		++index;
	}

	recheckDataType(context);

	LibMatchMexErrorWithMessage error_message = { error = LibMatchMexError::success };

	return error_message;
}