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

	return error;
}

LibMatchMexError parseA(ArrayMatchMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error = parse2DMatrixParameter(pa, (void**)&context->A, &context->lengthOfArray, &context->numberOfArrayA);
	if (error == LibMatchMexError::success)
		if (context->lengthOfArray * context->numberOfArrayA > INT_MAX)
			return LibMatchMexError::errorOverFlow;

	return error;
}

LibMatchMexError parseDebug(ArrayMatchMexContext *context,
	const mxArray *pa)
{
	if (!mxIsLogical(pa))
		return LibMatchMexError::errorTypeOfArgument;

	if (!mxIsScalar(pa))
		return LibMatchMexError::errorSizeOfArray;

	context->debug = mxGetLogicals(pa)[0];

	return LibMatchMexError::success;
}

LibMatchMexError parseOutputArgument(ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return LibMatchMexError::errorNumberOfArguments;

	return LibMatchMexError::success;
}

struct LibMatchMexErrorWithMessage parseParameter(ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	LibMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == LibMatchMexError::errorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments.");

	if (!(nrhs == 3 || nrhs == 4))
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

	if (nrhs == 4) {
		++index;
		error = parseDebug(context, prhs[index]);
		if (error == LibMatchMexError::errorSizeOfArray)
			return generateErrorMessage(error, "Parameter IsDebug must be scalar");
		else if (error == LibMatchMexError::errorTypeOfArgument)
			return generateErrorMessage(error, "Parameter IsDebug must be logical");
	}

	return generateErrorMessage(LibMatchMexError::success, "");
}