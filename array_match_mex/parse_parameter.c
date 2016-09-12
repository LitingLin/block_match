#include "common.h"

#include <string.h>

enum libMatchMexError parseMeasureMethod(struct ArrayMatchMexContext *context,
	const mxArray *pa)
{
	enum LibBlockMatchMexError error;
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

enum libMatchMexError parseB(struct ArrayMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return libMatchMexErrorTypeOfArgument;
	}

	size_t sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	if (context->numberOfArray != sequenceAMatrixDimensions[1])
	{
		return libMatchMexErrorSizeOfMatrixMismatch;
	}
	if (context->lengthOfArray != sequenceAMatrixDimensions[0])
	{
		return libMatchMexErrorSizeOfMatrixMismatch;
	}

	context->B = mxGetData(pa);

	return libMatchMexOk;
}

enum libMatchMexError parseA(struct ArrayMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return libMatchMexErrorTypeOfArgument;
	}
	
	size_t sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return libMatchMexErrorNumberOfMatrixDimension;
	}

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	if (sequenceAMatrixDimensions[1] > INT_MAX)
		return libMatchMexErrorOverFlow;
	context->numberOfArray = sequenceAMatrixDimensions[1];
	if (sequenceAMatrixDimensions[0] > INT_MAX)
		return libMatchMexErrorOverFlow;
	context->lengthOfArray = sequenceAMatrixDimensions[0];

	if (sequenceAMatrixDimensions[0] * sequenceAMatrixDimensions[1] > INT_MAX)
		return libMatchMexErrorOverFlow;

	context->A = mxGetData(pa);

	return libMatchMexOk;
}

enum libMatchMexError parseOutputArgument(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return libMatchMexErrorNumberOfArguments;

	return libMatchMexOk;
}

struct LibMatchMexErrorWithMessage internalError()
{
	return generateErrorMessage(libMatchMexErrorInternal, "Unknown internal error.");
}

struct LibMatchMexErrorWithMessage parseParameter(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	enum LibMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == libMatchMexErrorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments.");

	if (nrhs != 3)
		return generateErrorMessage(libMatchMexErrorNumberOfArguments, "Too few input arguments.");

	int index = 0;
	error = parseA(context, prhs[index]);
	if (error == libMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Data type of A must be double");
	else if (error == libMatchMexErrorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of dimensions of A must be 2");
	else if (error == libMatchMexErrorOverFlow)
		return generateErrorMessage(error, "Total size of A must smaller than INT_MAX(%d)", INT_MAX);
	else if (error != libMatchMexOk)
		return internalError();

	++index;

	error = parseB(context, prhs[index]);

	if (error == libMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Data type of A must be double.");
	else if (error == libMatchMexErrorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of dimensions of A must be 2.");
	else if (error == libMatchMexErrorSizeOfMatrixMismatch)
		return generateErrorMessage(error, "Dimensions of A and B mismatch.");
	else if (error != libMatchMexOk)
		return internalError();

	++index;
	error = parseMeasureMethod(context, prhs[index]);
	if (error == libMatchMexErrorTypeOfArgument)
		return generateErrorMessage(error, "Data type of MeasureMethod must be Char Array.");
	else if (error == libMatchMexErrorInvalidValue)
		return generateErrorMessage(error, "MeasureMethod must be 'mse'(Mean Square Error) or 'cc'(Correlation Coefficient).");
	else if (error != libMatchMexOk)
		return internalError();

	return generateErrorMessage(libMatchMexOk, "");
}