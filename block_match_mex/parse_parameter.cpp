#include "common.h"
#include <string.h>
#include "utils.h"

LibMatchMexError recheckSearchRegion(BlockMatchMexContext *context)
{
	if (context->searchType == SearchType::local)
		if (context->sequenceAMatrixDimensions[0] - context->blockWidth < context->searchRegionWidth ||
			context->sequenceAMatrixDimensions[1] - context->blockHeight < context->searchRegionHeight)
		{
			return LibMatchMexError::errorSizeOfArray;
		}
	return LibMatchMexError::success;
}

void recheckDataType(BlockMatchMexContext *context)
{
	std::type_index sourceType = typeid(nullptr);
	if (context->sourceAType == typeid(double) || context->sourceBType == typeid(double))
		sourceType = typeid(double);
	else
		sourceType = typeid(float);

	if (context->intermediateType == typeid(nullptr))
		context->intermediateType = sourceType;
	if (context->resultType == typeid(nullptr))
		context->resultType = sourceType;
}

void recheckSequenceBPadding(BlockMatchMexContext *context)
{
	if (context->searchType == SearchType::global)
	{
		if (context->sequenceBPaddingHeightPre == -1 || context->sequenceBPaddingHeightPre == -2)
		{
			context->sequenceBPaddingHeightPre = context->sequenceBPaddingHeightPost
				= context->sequenceBPaddingWidthPre = context->sequenceBPaddingWidthPost = 0;
		}
	}
	else
	{
		if (context->sequenceBPaddingHeightPre == -1)
		{
			context->sequenceBPaddingHeightPre = context->searchRegionHeight / 2;
			context->sequenceBPaddingHeightPost = context->searchRegionHeight - context->searchRegionHeight / 2;
			context->sequenceBPaddingWidthPre = context->searchRegionWidth / 2;
			context->sequenceBPaddingWidthPost = context->searchRegionWidth - context->searchRegionWidth / 2;
		}
		else if (context->sequenceBPaddingHeightPre == -2)
		{
			context->sequenceBPaddingHeightPre = context->sequenceBPaddingHeightPost = context->searchRegionHeight;
			context->sequenceBPaddingWidthPre = context->sequenceBPaddingWidthPost = context->searchRegionWidth;
		}
	}
}

LibMatchMexError recheckSortingParameter(BlockMatchMexContext *context)
{
	if (!context->sort && context->retain)
	{
		return LibMatchMexError::errorInvalidParameterCombination;
	}

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceABorderType(BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[17];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 17);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "includeLastBlock", 17) == 0)
		context->sequenceABorderType = BorderType::includeLastBlock;
	else if (strncmp(buffer, "normal", 7) == 0)
		context->sequenceABorderType = BorderType::normal;
	else
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseIntermediateType(BlockMatchMexContext *context,
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

LibMatchMexError parseResultDataType(BlockMatchMexContext *context,
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

LibMatchMexError parseSequenceAPaddingMethod(BlockMatchMexContext *context,
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

LibMatchMexError parseSequenceBPaddingMethod(BlockMatchMexContext *context,
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

LibMatchMexError parseRetain(BlockMatchMexContext *context,
	const mxArray *pa)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else
		goto ValueClass;

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
		if (!mxIsScalar(pa))
			return LibMatchMexError::errorSizeOfArray;

		LibMatchMexError error;
		error = getIntegerFromMxArray(pa, &context->retain);
		if (error != LibMatchMexError::success)
			return error;

		if (context->retain <= 0)
			return LibMatchMexError::errorInvalidValue;

		return LibMatchMexError::success;
	}
}

LibMatchMexError parseSort(BlockMatchMexContext *context,
	const mxArray *pa)
{
	bool sort;

	if (mxGetClassID(pa) != mxLOGICAL_CLASS)
		return LibMatchMexError::errorTypeOfArgument;

	mxLogical* logicals = mxGetLogicals(pa);
	sort = logicals[0];

	context->sort = sort;

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceAPaddingSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse4ElementNonNegativeIntegerParameter(pa,
		&context->sequenceAPaddingHeightPre, &context->sequenceAPaddingHeightPost,
		&context->sequenceAPaddingWidthPre, &context->sequenceAPaddingWidthPost);
}

LibMatchMexError parseSequenceBPaddingSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) == mxCHAR_CLASS)
	{
		char buffer[5];
		LibMatchMexError error = getStringFromMxArray(pa, buffer, 5);
		if (error != LibMatchMexError::success)
			return error;

		if (strncmp(buffer, "same", 5) == 0)
			context->sequenceBPaddingHeightPre = -1;
		else if (strncmp(buffer, "full", 5) == 0)
			context->sequenceBPaddingHeightPre = -2;
		else
			return LibMatchMexError::errorInvalidValue;

		return LibMatchMexError::success;
	}
	else
		return parse4ElementNonNegativeIntegerParameter(pa,
			&context->sequenceBPaddingHeightPre, &context->sequenceBPaddingHeightPost,
			&context->sequenceBPaddingWidthPre, &context->sequenceBPaddingWidthPost);
}

LibMatchMexError parseSequenceAStrideSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2ElementPositiveIntegerParameter(pa, &context->sequenceAStrideHeight, &context->sequenceAStrideWidth);
}

LibMatchMexError parseSequenceBStrideSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2ElementPositiveIntegerParameter(pa, &context->sequenceBStrideHeight, &context->sequenceBStrideWidth);
}

// SearchRegion size 0 for full search
LibMatchMexError parseSearchRegion(BlockMatchMexContext *context,
	const mxArray *pa)
{
	int searchRegionWidth, searchRegionHeight;

	char buffer[6];

	if (getStringFromMxArray(pa, buffer, 6) == LibMatchMexError::success)
	{
		if (strncmp(buffer, "full", 6) != 0)
			return LibMatchMexError::errorInvalidValue;

		context->searchRegionWidth = context->searchRegionHeight = 0;
		context->searchType = SearchType::global;
		return LibMatchMexError::success;
	}
	else
	{
		context->searchType = SearchType::local;
		return parse2ElementPositiveIntegerParameter(pa, &context->searchRegionWidth, &context->searchRegionHeight);
	}
}

LibMatchMexError parseBlockSize(BlockMatchMexContext *context,
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

LibMatchMexError parseSequenceB(BlockMatchMexContext *context,
	const mxArray *pa)
{
	context->sourceBType = getTypeIndex(mxGetClassID(pa));

	if (context->sourceBType != typeid(double) || context->sourceBType != typeid(double))
		return LibMatchMexError::errorTypeOfArgument;

	return parse2DMatrixParameter(pa, &context->sequenceBMatrixPointer,
		&context->sequenceBMatrixDimensions[0], &context->sequenceBMatrixDimensions[1]);
}

// TODO: multi-dimension
LibMatchMexError parseSequenceA(BlockMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = parse2DMatrixParameter(pa, &context->sequenceAMatrixPointer,
		&context->sequenceAMatrixDimensions[0], &context->sequenceAMatrixDimensions[1]);

	context->sequenceMatrixNumberOfDimensions = 2;

	context->sourceAType = getTypeIndex(mxGetClassID(pa));

	if (context->sourceAType != typeid(double) || context->sourceAType != typeid(double))
		return LibMatchMexError::errorTypeOfArgument;

	return error;
}
/*
 * Return:
 *  errorTypeOfArgument
 *  errorSizeOfArray
 *  errorInvalidValue
 *  success
 */
LibMatchMexError parseMethod(BlockMatchMexContext *context,
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

LibMatchMexError parseOutputArgument(BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 4)
		return LibMatchMexError::errorNumberOfArguments;

	return LibMatchMexError::success;
}

LibMatchMexErrorWithMessage unknownParsingError(char *parameterName)
{
	return generateErrorMessage(LibMatchMexError::errorInternal, "Unknown error occured when parsing parameter %s", parameterName);
}

LibMatchMexErrorWithMessage parseParameter(BlockMatchMexContext *context,
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
		return generateErrorMessage(error, "Type of Matrix A must be float or double\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix A must be 2\n");
	}

	++index;

	error = parseSequenceB(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix B must be float or double\n");
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
				"Argument %d should be the name of parameter(string).", index);
			return generateErrorMessage(error, messageBuffer);
		}

		++index;

		if (strncmp(buffer, "SearchRegion", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSearchRegion(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Number of dimension of SearchRegion must be 1 or 2 for numeric array.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "SearchReion must be 'full' or positive integer.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of SearchRegion overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "SearchRegion must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}

		else if (strncmp(buffer, "SequenceAStride", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAStrideSize(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Number of dimension of SequenceAStride must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "SequenceAStride must be positive.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of SequenceAStride overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "SequenceAStride must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "SequenceBStride", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBStrideSize(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Number of dimension of SequenceBStride must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "SequenceBStride must be positive.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of SequenceBStride overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "SequenceBStride must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "SequenceAPadding", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingSize(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Number of dimension of SequenceAPadding must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "SequenceAPadding must be non-negative.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of SequenceAPadding overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "SequenceAPadding must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "SequenceBPadding", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingSize(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Number of dimension of SequenceBPadding must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "SequenceBPadding must be non-negative.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of SequenceBPadding overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "SequenceBPadding must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "MeasureMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument MeasureMethod must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument MeasureMethod.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "SequenceAPaddingMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument SequenceAPaddingMethod must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument SequenceAPaddingMethod.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "SequenceBPaddingMethod", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument SequenceBPaddingMethod must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument SequenceBPaddingMethod.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "Threshold", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0); // TODO
		else if (strncmp(buffer, "Sort", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSort(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument Sort must be logical(boolean)\n");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "Retain", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseRetain(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Argument Retain must be scalar or 'all'.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument Retain must be integer or string.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "Value of argument Retain must be 'all' or positive integer.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of argument Retain overflowed.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "ResultDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseResultDataType(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument ResultDataType must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument ResultDataType.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "IntermediateDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseIntermediateType(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument IntermediateDataType must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument IntermediateDataType.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "Sparse", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0);
		else if (strncmp(buffer, "SequenceABorder", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceABorderType(context, prhs[index]);

			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument SequenceABorder must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument SequenceABorder.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}

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
	error = recheckSortingParameter(context);
	if (error == LibMatchMexError::errorInvalidParameterCombination)
		return generateErrorMessage(error, "Parameter Retain cannot be integer when Parameter Sort is given");

	error = recheckSearchRegion(context);
	if (error == LibMatchMexError::errorSizeOfArray)
		return generateErrorMessage(error, "SearchRegionSize cannot be smaller then the size of Matrix A - BlockSize\n");

	recheckSequenceBPadding(context);

	LibMatchMexErrorWithMessage error_message = { error = LibMatchMexError::success };

	return error_message;
}