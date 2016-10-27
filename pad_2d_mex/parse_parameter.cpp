#include "common.h"

#include <string.h>

LibMatchMexError parsePadM(struct PaddingMexContext *context,
	const mxArray *pa)
{
	return parse2ElementIntegerParameter(pa, &context->pad_M_pre, &context->pad_M_post);
}

LibMatchMexError parsePadN(struct PaddingMexContext *context,
	const mxArray *pa)
{
	return parse2ElementIntegerParameter(pa, &context->pad_N_pre, &context->pad_N_post);
}

LibMatchMexError parseMethod(struct PaddingMexContext *context,
	const mxArray *pa)
{
	LibMatchMexError error;
	char buffer[10];
	error = getStringFromMxArray(pa, buffer, 10);
	if (error != LibMatchMexError::success)
		return error;

	if (strncmp(buffer, "zero", 10) == 0)
		context->method = PadMethod::zero;
	else if (strncmp(buffer, "circular", 10) == 0)
		context->method = PadMethod::circular;
	else if (strncmp(buffer, "replicate", 10) == 0)
		context->method = PadMethod::replicate;
	else if (strncmp(buffer, "symmetric", 10) == 0)
		context->method = PadMethod::symmetric;
	else
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

LibMatchMexError parseMatrix(struct PaddingMexContext *context,
	const mxArray *pa)
{
	return parse2DMatrixParameter(pa, reinterpret_cast<void**>(&context->originImage), &context->image_M, &context->image_N);
}

LibMatchMexError parseOutputArgument(struct PaddingMexContext *context, 
	int nlhs, mxArray *plhs[])
{
	if (nlhs > 1)
		return LibMatchMexError::errorNumberOfArguments;

	return LibMatchMexError::success;
}

struct LibMatchMexErrorWithMessage parseParameter(struct PaddingMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	LibMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error == LibMatchMexError::errorNumberOfArguments)
		return generateErrorMessage(error, "Too many output arguments.");

	if (nrhs != 4)
		return generateErrorMessage(LibMatchMexError::errorNumberOfArguments, "Too few input arguments.");

	int index = 0;
	error = parseMatrix(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of matrix must be double");
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of dimensions of matrix must be 2");
	else if (error == LibMatchMexError::errorOverFlow)
		return generateErrorMessage(error, "Total size of matrix must smaller than INT_MAX(%d)", INT_MAX);
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();

	++index;

	error = parsePadM(context, prhs[index]);

	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of PadM must be double.");
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of elements of PadM must be 1 or 2.");
	else if (error == LibMatchMexError::errorOverFlow)
		return generateErrorMessage(error, "Value of PadM must smaller than INT_MAX(%d)", INT_MAX);
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();

	++index;

	error = parsePadN(context, prhs[index]);

	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of PadN must be double.");
	else if (error == LibMatchMexError::errorNumberOfMatrixDimension)
		return generateErrorMessage(error, "Number of elements of PadN must be 1 or 2.");
	else if (error == LibMatchMexError::errorOverFlow)
		return generateErrorMessage(error, "Value of PadN must smaller than INT_MAX(%d)", INT_MAX);
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();

	++index;
	error = parseMethod(context, prhs[index]);
	if (error == LibMatchMexError::errorTypeOfArgument)
		return generateErrorMessage(error, "Data type of Method must be Char Array.");
	else if (error == LibMatchMexError::errorInvalidValue || error == LibMatchMexError::errorSizeOfArray)
		return generateErrorMessage(error, "Method must be 'zero', 'circular', 'replicate' or 'symmetric'.");
	else if (error != LibMatchMexError::success)
		return internalErrorMessage();

	if (context->method != PadMethod::zero)
	{
		if (error == LibMatchMexError::success)
		{
			if (context->pad_M_pre > context->image_M || context->pad_M_post > context->image_M)
				return generateErrorMessage(LibMatchMexError::errorInvalidValue, "Value of PadM cannot exceed the size of matrix");
		}

		if (error == LibMatchMexError::success)
		{
			if (context->pad_N_pre > context->image_N || context->pad_N_post > context->image_N)
				return generateErrorMessage(LibMatchMexError::errorInvalidValue, "Value of PadN cannot exceed the size of matrix");
		}
	}

	return generateErrorMessage(LibMatchMexError::success, "");
}