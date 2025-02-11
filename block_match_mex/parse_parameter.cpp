#include "common.h"
#include <string.h>
/*
void recheckSparse(BlockMatchMexContext *context)
{
	if (context->sparse == -1)
	{
		if (context->threshold)
			context->sparse = true;
		else
			context->sparse = false;
	}
}
*/
LibMatchMexError recheckSearchRegion(BlockMatchMexContext *context)
{
	if (context->searchType == SearchType::local)
		if (context->sequenceAMatrixDimensions[0] - context->block_M < (context->searchRegion_M_pre + 1 + context->searchRegion_M_post) ||
			context->sequenceAMatrixDimensions[1] - context->block_N < (context->searchRegion_N_pre + 1 + context->searchRegion_N_post))
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

	if (context->indexDataType == typeid(nullptr)) { // auto
		uint32_t sequenceBPadded_M = uint32_t(context->sequenceBMatrixDimensions[0] + context->sequenceBPadding_M_Pre + context->sequenceBPadding_M_Post);
		uint32_t sequenceBPadded_N = uint32_t(context->sequenceBMatrixDimensions[1] + context->sequenceBPadding_N_Pre + context->sequenceBPadding_N_Post);
		std::type_index indexDataType = typeid(nullptr);
		if (sequenceBPadded_M < std::numeric_limits<uint8_t>::max() && sequenceBPadded_N < std::numeric_limits<uint8_t>::max())
			indexDataType = typeid(uint8_t);
		else if (sequenceBPadded_M < std::numeric_limits<uint16_t>::max() && sequenceBPadded_N < std::numeric_limits<uint16_t>::max())
			indexDataType = typeid(uint16_t);
		else if (sequenceBPadded_M < std::numeric_limits<uint32_t>::max() && sequenceBPadded_N < std::numeric_limits<uint32_t>::max())
			indexDataType = typeid(uint32_t);
		else
			indexDataType = typeid(uint64_t);

		context->indexDataType = indexDataType;
	}
}

void recheckSequenceBPadding(BlockMatchMexContext *context)
{
	if (context->searchType == SearchType::global)
	{
		if (context->sequenceBPadding_N_Pre == -1 || context->sequenceBPadding_N_Pre == -2)
		{
			context->sequenceBPadding_N_Pre = context->sequenceBPadding_N_Post
				= context->sequenceBPadding_M_Pre = context->sequenceBPadding_M_Post = 0;
		}
	}
	else
	{
		if (context->sequenceBPadding_N_Pre == -1) // same
		{
			context->sequenceBPadding_N_Pre = context->searchRegion_N_pre;
			context->sequenceBPadding_N_Post = context->searchRegion_N_post;
			context->sequenceBPadding_M_Pre = context->searchRegion_M_pre;
			context->sequenceBPadding_M_Post = context->searchRegion_M_post;
		}
		else if (context->sequenceBPadding_N_Pre == -2) // full
		{
			context->sequenceBPadding_N_Pre = context->sequenceBPadding_N_Post = context->searchRegion_N_pre + context->searchRegion_N_post;
			context->sequenceBPadding_M_Pre = context->sequenceBPadding_M_Post = context->searchRegion_M_pre + context->searchRegion_M_post;
		}
	}
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

LibMatchMexError parseIndexDataType(BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[8];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 8);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "single", 6) == 0)
		context->indexDataType = typeid(float);
	else if (strncmp(buffer, "double", 7) == 0)
		context->indexDataType = typeid(double);
	else if (strncmp(buffer, "logical", 8) == 0)
		context->indexDataType = typeid(bool);
	else if (strncmp(buffer, "uint8", 6) == 0)
		context->indexDataType = typeid(uint8_t);
	else if (strncmp(buffer, "int8", 5) == 0)
		context->indexDataType = typeid(int8_t);
	else if (strncmp(buffer, "uint16", 7) == 0)
		context->indexDataType = typeid(uint16_t);
	else if (strncmp(buffer, "int16", 6) == 0)
		context->indexDataType = typeid(int16_t);
	else if (strncmp(buffer, "uint32", 7) == 0)
		context->indexDataType = typeid(uint32_t);
	else if (strncmp(buffer, "int32", 6) == 0)
		context->indexDataType = typeid(int32_t);
	else if (strncmp(buffer, "uint64", 7) == 0)
		context->indexDataType = typeid(uint64_t);
	else if (strncmp(buffer, "int64", 6) == 0)
		context->indexDataType = typeid(int64_t);
	else if (strncmp(buffer, "auto", 5) != 0)
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

	if (strncmp(buffer, "single", 7) == 0)
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
	char buffer[8];
	LibMatchMexError error;
	error = getStringFromMxArray(pa, buffer, 8);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "single", 7) == 0)
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

LibMatchMexError parseThreshold(BlockMatchMexContext *context,
	const mxArray *pa)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else
		goto ValueClass;

	LibMatchMexError error;

StringClass:
	{
		char buffer[3];

		error = getStringFromMxArray(pa, buffer, 3);
		if (error != LibMatchMexError::success)
			return error;

		if (strncmp(buffer, "no", 3) != 0)
			return LibMatchMexError::errorInvalidValue;

		context->threshold = false;

		return LibMatchMexError::success;
	}
ValueClass:
	{
		if (!mxIsScalar(pa))
			return LibMatchMexError::errorSizeOfArray;

		context->threshold = true;

		context->thresholdValue = mxGetScalar(pa);

		return LibMatchMexError::success;
	}
}


LibMatchMexError parseThresholdReplacementValue(BlockMatchMexContext *context,
	const mxArray *pa)
{
	if (!mxIsNumeric(pa))
		return LibMatchMexError::errorTypeOfArgument;

	if (!mxIsScalar(pa))
		return LibMatchMexError::errorSizeOfArray;

	context->thresholdReplacementValue = mxGetScalar(pa);

	return LibMatchMexError::success;
}

//
//LibMatchMexError parseSparse(BlockMatchMexContext *context,
//	const mxArray *pa)
//{
//	mxClassID classId = mxGetClassID(pa);
//	if (classId == mxCHAR_CLASS)
//		goto StringClass;
//	else
//		goto ValueClass;
//
//	LibMatchMexError error;
//
//StringClass:
//	{
//		char buffer[5];
//
//		error = getStringFromMxArray(pa, buffer, 5);
//		if (error != LibMatchMexError::success)
//			return error;
//
//		if (strncmp(buffer, "auto", 5) != 0)
//			return LibMatchMexError::errorInvalidValue;
//
//		context->sparse = -1;
//
//		return LibMatchMexError::success;
//	}
//ValueClass:
//	{
//		if (!mxIsScalar(pa))
//			return LibMatchMexError::errorSizeOfArray;
//
//		if (!mxIsLogical(pa))
//			return LibMatchMexError::errorTypeOfArgument;
//
//		context->sparse = mxIsLogicalScalarTrue(pa);
//		
//		return LibMatchMexError::success;
//	}
//}

LibMatchMexError parseIndexOfDevice(BlockMatchMexContext *context,
	const mxArray *pa)
{
	if (!mxIsScalar(pa))
		return LibMatchMexError::errorTypeOfArgument;

	LibMatchMexError error = getIntegerFromMxArray(pa, &context->indexOfDevice);

	if (error != LibMatchMexError::success)
		return error;

	if (context->indexOfDevice < 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseNumberOfThreads(BlockMatchMexContext *context,
	const mxArray *pa)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else
		goto ValueClass;

	LibMatchMexError error;

StringClass:
	{
		char buffer[5];

		error = getStringFromMxArray(pa, buffer, 5);

		if (error != LibMatchMexError::success)
			return error;

		if (strncmp(buffer, "auto", 5) != 0)
			return LibMatchMexError::errorInvalidValue;

		context->numberOfThreads = 0;

		return LibMatchMexError::success;
	}
ValueClass:
	{
		if (!mxIsScalar(pa))
			return LibMatchMexError::errorSizeOfArray;

		error = getUnsignedIntegerFromMxArray(pa, &context->numberOfThreads);
		if (error != LibMatchMexError::success)
			return error;

		if (context->numberOfThreads == 0)
			return LibMatchMexError::errorInvalidValue;

		return LibMatchMexError::success;
	}
}

LibMatchMexError parseRetain(BlockMatchMexContext *context,
	const mxArray *pa)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId == mxCHAR_CLASS)
		goto StringClass;
	else
		goto ValueClass;

	LibMatchMexError error;

StringClass:
	{
		char buffer[4];

		error = getStringFromMxArray(pa, buffer, 4);

		if (error != LibMatchMexError::success)
			return error;

		if (strncmp(buffer, "all", 4) != 0)
			return LibMatchMexError::errorInvalidValue;

		context->retain = 0;

		return LibMatchMexError::success;
	}
ValueClass:
	{
		if (!mxIsScalar(pa))
			return LibMatchMexError::errorSizeOfArray;

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
		&context->sequenceAPadding_N_Pre, &context->sequenceAPadding_N_Post,
		&context->sequenceAPadding_M_Pre, &context->sequenceAPadding_M_Post);
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
			context->sequenceBPadding_N_Pre = -1;
		else if (strncmp(buffer, "full", 5) == 0)
			context->sequenceBPadding_N_Pre = -2;
		else
			return LibMatchMexError::errorInvalidValue;

		return LibMatchMexError::success;
	}
	else
		return parse4ElementNonNegativeIntegerParameter(pa,
			&context->sequenceBPadding_N_Pre, &context->sequenceBPadding_N_Post,
			&context->sequenceBPadding_M_Pre, &context->sequenceBPadding_M_Post);
}

LibMatchMexError parseSequenceAStrideSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2ElementPositiveIntegerParameter(pa, &context->sequenceAStride_N, &context->sequenceAStride_M);
}

LibMatchMexError parseSequenceBStrideSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	return parse2ElementPositiveIntegerParameter(pa, &context->sequenceBStride_N, &context->sequenceBStride_M);
}

// SearchRegion size 0 for full search
LibMatchMexError parseSearchRegion(BlockMatchMexContext *context,
	const mxArray *pa)
{
	char buffer[6];

	if (getStringFromMxArray(pa, buffer, 6) == LibMatchMexError::success)
	{
		if (strncmp(buffer, "full", 6) != 0)
			return LibMatchMexError::errorInvalidValue;

		context->searchRegion_M_pre = context->searchRegion_M_post
			= context->searchRegion_N_pre = context->searchRegion_N_post = 0;
		context->searchType = SearchType::global;
		return LibMatchMexError::success;
	}
	else
	{
		context->searchType = SearchType::local;
		return parse4ElementPositiveIntegerParameter(pa, &context->searchRegion_N_pre, &context->searchRegion_N_post,
			&context->searchRegion_M_pre, &context->searchRegion_M_post);
	}
}

LibMatchMexError parseBlockSize(BlockMatchMexContext *context,
	const mxArray *pa)
{
	int block_M, block_N;
	LibMatchMexError error = parse2ElementPositiveIntegerParameter(pa, &block_N, &block_M);
	if (error != LibMatchMexError::success)
		return error;

	if (context->sequenceAMatrixDimensions[0] < block_M || context->sequenceAMatrixDimensions[1] < block_N)
	{
		return LibMatchMexError::errorSizeOfArray;
	}

	context->block_M = block_M;
	context->block_N = block_N;

	return LibMatchMexError::success;
}

LibMatchMexError parseSequenceB(BlockMatchMexContext *context,
	const mxArray *pa)
{
	context->sourceBType = getTypeIndex(mxGetClassID(pa));

	if (context->sourceBType == typeid(nullptr))
		return LibMatchMexError::errorTypeOfArgument;

	if (context->sequenceMatrixNumberOfDimensions == 2)
	{
		context->sequenceBMatrixDimensions[2] = 1;
		return parse2DMatrixParameter(pa, &context->sequenceBMatrixPointer,
			&context->sequenceBMatrixDimensions[0], &context->sequenceBMatrixDimensions[1]);
	}
	else if (context->sequenceMatrixNumberOfDimensions == 3)
	{
		LibMatchMexError error = parse3DMatrixParameter(pa, &context->sequenceBMatrixPointer,
			&context->sequenceBMatrixDimensions[0], &context->sequenceBMatrixDimensions[1],
			&context->sequenceBMatrixDimensions[2]);

		if (error != LibMatchMexError::success)
			return error;

		if (context->sequenceAMatrixDimensions[2] != context->sequenceBMatrixDimensions[2])
			return LibMatchMexError::errorNumberOfMatrixDimensionMismatch;

		return LibMatchMexError::success;
	}
	else
		return LibMatchMexError::errorNumberOfMatrixDimension;
}

// TODO: multi-dimension
LibMatchMexError parseSequenceA(BlockMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = LibMatchMexError::success;
	if (mxGetNumberOfDimensions(pa) == 2)
	{
		error = parse2DMatrixParameter(pa, &context->sequenceAMatrixPointer,
			&context->sequenceAMatrixDimensions[0], &context->sequenceAMatrixDimensions[1]);

		context->sequenceAMatrixDimensions[2] = 1;

		context->sequenceMatrixNumberOfDimensions = 2;
	}
	else if (mxGetNumberOfDimensions(pa) == 3)
	{
		error = parse3DMatrixParameter(pa, &context->sequenceAMatrixPointer,
			&context->sequenceAMatrixDimensions[0], &context->sequenceAMatrixDimensions[1],
			&context->sequenceAMatrixDimensions[2]);

		context->sequenceMatrixNumberOfDimensions = 3;
	}
	else
		return LibMatchMexError::errorNumberOfMatrixDimension;

	context->sourceAType = getTypeIndex(mxGetClassID(pa));

	if (context->sourceAType == typeid(nullptr))
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
		context->method = MeasureMethod::mse;
	else if (strncmp(buffer, "cc", 4) == 0)
		context->method = MeasureMethod::cc;
	else
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseOutputArgument(BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs > 4)
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
		return generateErrorMessage(error, "Type of Matrix A must be numeric\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix A must be 2 or 3\n");
	}
	else if (error != LibMatchMexError::success)
		return unknownParsingError("A");

	++index;

	error = parseSequenceB(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
	{
		return generateErrorMessage(error, "Type of Matrix B must be numeric\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
	{
		return generateErrorMessage(error, "Number of dimension of Matrix B must be 2 or 3\n");
	}
	else if (error == LibMatchMexError::errorNumberOfMatrixDimensionMismatch)
	{
		return generateErrorMessage(error, "Number of channels of Matrix B must be the same as Matrix A\n");
	}
	else if (error != LibMatchMexError::success)
		return unknownParsingError("B");

	++index;

	error = parseBlockSize(context, prhs[index]);
	if (error == LibMatchMexError::errorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of dimension of BlockSize must be 1 or 2.");
	else if (error == LibMatchMexError::errorInvalidValue)
		return generateErrorMessage(error, "BlockSize must be positive.");
	else if (error == LibMatchMexError::errorOverFlow)
		return generateErrorMessage(error, "Value of BlockSize overflowed.");
	else if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "BlockSize must be numeric array");
	else if (error == LibMatchMexError::errorSizeOfArray)
		return generateErrorMessage(error, "BlockSize cannot be smaller then Matrix A\n");
	else if (error != LibMatchMexError::success)
		return unknownParsingError("BlockSize");

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

		if (strncmp(buffer, "SearchWindow", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSearchRegion(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Number of dimension of SearchWindow must be 1 or 2 for numeric array.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "SearchWindow must be 'full' or positive integer.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of SearchWindow overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "SearchWindow must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}

		else if (strncmp(buffer, "StrideA", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAStrideSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Number of dimension of StrideA must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "StrideA must be positive.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of StrideA overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "StrideA must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "StrideB", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBStrideSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Number of dimension of StrideB must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "StrideB must be positive.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of StrideB overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "StrideB must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "PaddingA", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Number of dimension of PaddingA must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "PaddingA must be non-negative.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of PaddingA overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "PaddingA must be numeric array");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "PaddingB", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingSize(context, prhs[index]);
			if (error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Number of dimension of PaddingB must be 1 or 2.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "PaddingB must be non-negative.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of PaddingB overflowed.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "PaddingB must be numeric array");
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
		else if (strncmp(buffer, "PaddingMethodA", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceAPaddingMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument PaddingMethodA must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray || error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Invalid value of argument PaddingMethodA.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "PaddingMethodB", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceBPaddingMethod(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument PaddingMethodB must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray || error == LibMatchMexError::errorNumberOfMatrixDimension)
				return generateErrorMessage(error, "Invalid value of argument PaddingMethodB.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "Threshold", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseThreshold(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray ||
				error == LibMatchMexError::errorTypeOfArgument ||
				error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "Argument Threshold must be scalar or 'no'.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "ThresholdReplacementValue", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseThresholdReplacementValue(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray ||
				error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument Threshold must be scalar.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "Sort", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSort(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument Sort must be logical\n");
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
		else if (strncmp(buffer, "IndexDataType", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseIndexDataType(context, prhs[index]);
			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument IntermediateDataType must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument IntermediateDataType.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}/*
		else if (strncmp(buffer, "Sparse", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSparse(context, prhs[index]);
			if (error == LibMatchMexError::errorSizeOfArray ||
				error == LibMatchMexError::errorTypeOfArgument ||
				error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "Argument Sparse must be logical or 'auto'.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}*/
		else if (strncmp(buffer, "BorderA", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseSequenceABorderType(context, prhs[index]);

			if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument BorderA must be string.");
			else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Invalid value of argument BorderA.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "NumberOfThreads", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseNumberOfThreads(context, prhs[index]);

			if (error == LibMatchMexError::errorSizeOfArray)
				return generateErrorMessage(error, "Argument NumberOfThreads must be scalar or 'auto'.");
			else if (error == LibMatchMexError::errorTypeOfArgument)
				return generateErrorMessage(error, "Argument NumberOfThreads must be integer or string.");
			else if (error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "Value of argument NumberOfThreads must be 'auto' or positive integer.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of argument NumberOfThreads overflowed.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else if (strncmp(buffer, "IndexOfDevice", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
		{
			error = parseIndexOfDevice(context, prhs[index]);

			if (error == LibMatchMexError::errorTypeOfArgument || error == LibMatchMexError::errorInvalidValue)
				return generateErrorMessage(error, "Argument IndexOfDevice must be positive integer.");
			else if (error == LibMatchMexError::errorOverFlow)
				return generateErrorMessage(error, "Value of argument IndexOfDevice overflowed.");
			else if (error != LibMatchMexError::success)
				return unknownParsingError(buffer);
		}
		else
		{
			sprintf_s(messageBuffer, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH,
				"Argument %s has not implemented yet, or has wrong name\n", buffer);
			return generateErrorMessage(LibMatchMexError::errorNotImplemented, messageBuffer);
		}
		++index;
	}
	
	error = recheckSearchRegion(context);
	if (error == LibMatchMexError::errorSizeOfArray)
		return generateErrorMessage(error, "SearchRegionSize cannot be smaller then the size of Matrix A - BlockSize\n");

	recheckSequenceBPadding(context);
	recheckDataType(context);
	// recheckSparse(context);

	LibMatchMexErrorWithMessage error_message = { error = LibMatchMexError::success };

	return error_message;
}