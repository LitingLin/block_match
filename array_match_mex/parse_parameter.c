#include "common.h"

#include <string.h>

enum LibBlockMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId != mxCHAR_CLASS)
		return arrayMatchMexErrorTypeOfArgument;

	mxGetNChars(pa, buffer, bufferLength);

	return arrayMatchMexOk;
}

enum ArrayMatchMexError parseMeasureMethod(struct ArrayMatchMexContext *context,
	const mxArray *pa)
{
	enum LibBlockMatchMexError error;
	char buffer[4];
	error = getStringFromMxArray(pa, buffer, 4);
	if (error != arrayMatchMexOk)
		return error;

	if (strncmp(buffer, "mse", 4) == 0)
		context->method = MSE;
	else if (strncmp(buffer, "cc", 4) == 0)
		context->method = CC;
	else
		return arrayMatchMexErrorInvalidValue;

	return arrayMatchMexOk;
}

enum ArrayMatchMexError parseB(struct ArrayMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return arrayMatchMexErrorTypeOfArgument;
	}

	size_t sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return arrayMatchMexErrorNumberOfMatrixDimensions;
	}

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	if (context->numberOfArray != sequenceAMatrixDimensions[0])
	{
		return arrayMatchMexErrorSizeOfMatrixMismatch;
	}
	if (context->lengthOfArray != sequenceAMatrixDimensions[1])
	{
		return arrayMatchMexErrorSizeOfMatrixMismatch;
	}

	context->B = mxGetData(pa);

	return arrayMatchMexOk;
}

enum ArrayMatchMexError parseA(struct ArrayMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return arrayMatchMexErrorTypeOfArgument;
	}
	
	size_t sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return arrayMatchMexErrorNumberOfMatrixDimensions;
	}

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	if (sequenceAMatrixDimensions[0] > INT_MAX)
		return arrayMatchMexErrorOverFlow;
	context->numberOfArray = sequenceAMatrixDimensions[0];
	if (sequenceAMatrixDimensions[1] > INT_MAX)
		return arrayMatchMexErrorOverFlow;
	context->lengthOfArray = sequenceAMatrixDimensions[1];

	if (sequenceAMatrixDimensions[0] * sequenceAMatrixDimensions[1] > INT_MAX)
		return arrayMatchMexErrorOverFlow;

	context->A = mxGetData(pa);

	return arrayMatchMexOk;
}

enum ArrayMatchMexError parseOutputArgument(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return arrayMatchMexErrorNumberOfArguments;

	return arrayMatchMexOk;
}

struct ArrayMatchMexErrorWithMessage internalError()
{
	return generateErrorMessage(arrayMatchMexErrorInternal, "Unknown internal error.");
}

struct ArrayMatchMexErrorWithMessage parseParameter(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	enum ArrayMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == arrayMatchMexErrorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments.");

	if (nrhs != 3)
		return generateErrorMessage(arrayMatchMexErrorNumberOfArguments, "Too few input arguments.");

	int index = 0;
	error = parseA(context, prhs[index]);
	if (error == arrayMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Data type of A must be double");
	else if (error == arrayMatchMexErrorNumberOfMatrixDimensions)
		return generateErrorMessage(error, "Number of dimensions of A must be 2");
	else if (error == arrayMatchMexErrorOverFlow)
		return generateErrorMessage(error, "Total size of A must smaller than INT_MAX(%d)", INT_MAX);
	else if (error != arrayMatchMexOk)
		return internalError();

	++index;

	error = parseB(context, prhs[index]);

	if (error == arrayMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Data type of A must be double.");
	else if (error == arrayMatchMexErrorNumberOfMatrixDimensions)
		return generateErrorMessage(error, "Number of dimensions of A must be 2.");
	else if (error == arrayMatchMexErrorSizeOfMatrixMismatch)
		return generateErrorMessage(error, "Dimensions of A and B mismatch.");
	else if (error != arrayMatchMexOk)
		return internalError();

	++index;
	error = parseMeasureMethod(context, prhs[index]);
	if (error == arrayMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Data type of MeasureMethod must be Char Array.");
	else if (error == arrayMatchMexErrorInvalidValue)
		return generateErrorMessage(error, "MeasureMethod must be 'mse'(Mean Square Error) or 'cc'(Correlation Coefficient).");
	else if (error != arrayMatchMexOk)
		return internalError();

	return generateErrorMessage(arrayMatchMexOk, "");
}