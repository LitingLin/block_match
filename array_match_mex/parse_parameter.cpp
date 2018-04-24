#include "common.h"

#include <string.h>

LibMatchMexError parseMeasureMethod(ArrayMatchMexContext *context,
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

LibMatchMexError parseB(ArrayMatchMexContext *context,
	const mxArray *pa)
{
	int lengthOfArrayB, numberOfArrayB;
	LibMatchMexError error = parse2DMatrixParameter(pa, (void**)&context->B, &lengthOfArrayB, &numberOfArrayB);
	context->numberOfArrayB = numberOfArrayB;

	if (error == LibMatchMexError::success)
		if (context->lengthOfArray != lengthOfArrayB)
			return LibMatchMexError::errorSizeOfMatrixMismatch;

	context->sourceBType = getTypeIndex(mxGetClassID(pa));

	if (context->sourceBType == typeid(nullptr))
		return LibMatchMexError::errorTypeOfArgument;

	return error;
}

LibMatchMexError parseA(ArrayMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = parse2DMatrixParameter(pa, (void**)&context->A, &context->lengthOfArray, &context->numberOfArrayA);
	if (error == LibMatchMexError::success)
		if (context->lengthOfArray * context->numberOfArrayA > INT_MAX)
			return LibMatchMexError::errorOverFlow;

	context->sourceAType = getTypeIndex(mxGetClassID(pa));

	if (context->sourceAType == typeid(nullptr))
		return LibMatchMexError::errorTypeOfArgument;

	return error;
}

LibMatchMexError parseSort(ArrayMatchMexContext *context,
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

LibMatchMexError parseRetain(ArrayMatchMexContext *context,
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

LibMatchMexError parseIndexDataType(ArrayMatchMexContext *context,
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

LibMatchMexError parseIntermediateType(ArrayMatchMexContext *context,
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

LibMatchMexError parseResultDataType(ArrayMatchMexContext *context,
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
	else if (strncmp(buffer, "logical", 8) == 0)
		context->resultType = typeid(bool);
	else if (strncmp(buffer, "uint8", 6) == 0)
		context->resultType = typeid(uint8_t);
	else if (strncmp(buffer, "int8", 5) == 0)
		context->resultType = typeid(int8_t);
	else if (strncmp(buffer, "uint16", 7) == 0)
		context->resultType = typeid(uint16_t);
	else if (strncmp(buffer, "int16", 6) == 0)
		context->resultType = typeid(int16_t);
	else if (strncmp(buffer, "uint32", 7) == 0)
		context->resultType = typeid(uint32_t);
	else if (strncmp(buffer, "int32", 6) == 0)
		context->resultType = typeid(int32_t);
	else if (strncmp(buffer, "uint64", 7) == 0)
		context->resultType = typeid(uint64_t);
	else if (strncmp(buffer, "int64", 6) == 0)
		context->resultType = typeid(int64_t);
	else if (strncmp(buffer, "same", 5) != 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseThreshold(ArrayMatchMexContext *context,
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


LibMatchMexError parseThresholdReplacementValue(ArrayMatchMexContext *context,
	const mxArray *pa)
{
	if (!mxIsNumeric(pa))
		return LibMatchMexError::errorTypeOfArgument;

	if (!mxIsScalar(pa))
		return LibMatchMexError::errorSizeOfArray;

	context->thresholdReplacementValue = mxGetScalar(pa);

	return LibMatchMexError::success;
}

LibMatchMexError parseIndexOfDevice(ArrayMatchMexContext *context,
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

LibMatchMexError parseNumberOfThreads(ArrayMatchMexContext *context,
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

LibMatchMexError parseOutputArgument(ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs > 2)
		return LibMatchMexError::errorNumberOfArguments;

	return LibMatchMexError::success;
}

LibMatchMexErrorWithMessage unknownParsingError(char *parameterName)
{
	return generateErrorMessage(LibMatchMexError::errorInternal, "Unknown error occured when parsing parameter %s", parameterName);
}

struct LibMatchMexErrorWithMessage parseParameter(ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	LibMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == LibMatchMexError::errorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments.");

	if (nrhs <= 3)
		return generateErrorMessage(LibMatchMexError::errorNumberOfArguments, "Too few input arguments.");

	int index = 0;
	error = parseA(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of A must be double");
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of dimensions of A must be 2");
	else if (error == LibMatchMexError::errorOverFlow)
		return generateErrorMessage(error, "Total size of A must smaller than INT_MAX(%d)", INT_MAX);
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();

	++index;

	error = parseB(context, prhs[index]);

	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of A must be double.");
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of dimensions of A must be 2.");
	else if (error == LibMatchMexError::errorSizeOfMatrixMismatch)
		return generateErrorMessage(error, "Number of rows of A and B mismatch.");
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();

	++index;
	error = parseMeasureMethod(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of MeasureMethod must be Char Array.");
	else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
		return generateErrorMessage(error, "MeasureMethod must be 'mse'(Mean Square Error) or 'cc'(Correlation Coefficient).");
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();
	
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

		if (strncmp(buffer, "Sort", LIB_MATCH_MEX_MAX_PARAMETER_NAME_LENGTH) == 0)
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

	return generateErrorMessage(LibMatchMexError::success, "");
}