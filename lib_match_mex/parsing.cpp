#include "lib_match_mex_common.h"

LibMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId != mxCHAR_CLASS)
		return LibMatchMexError::errorTypeOfArgument;

	size_t numberOfElements = mxGetNumberOfElements(pa);

	if (numberOfElements >= bufferLength)
		return LibMatchMexError::errorSizeOfArray;

	mxGetNChars(pa, buffer, bufferLength);

	return LibMatchMexError::success;
}

// Update: won't fix
// TODO: int to size_t
LibMatchMexError parse2ElementIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB)
{
	LibMatchMexError error;
	if (mxGetNumberOfElements(pa) == 1)
	{
		error = getIntegerFromMxArray(pa, parameterA);
		if (error != LibMatchMexError::success)
			return error;

		*parameterB = *parameterA;
		return LibMatchMexError::success;
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		return getTwoIntegerFromMxArray(pa, parameterA, parameterB);
	}
	else
		return LibMatchMexError::errorNumberOfMatrixDimension;
}

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  errorTypeOfArgument
*  success
*/
LibMatchMexError parse2ElementPositiveIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB)
{
	LibMatchMexError error = parse2ElementIntegerParameter(pa, parameterA, parameterB);
	if (error != LibMatchMexError::success)
		return error;

	if (*parameterA <= 0 || *parameterB <= 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  success
*/
LibMatchMexError parse2ElementNonNegativeIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB)
{
	LibMatchMexError error = parse2ElementIntegerParameter(pa, parameterA, parameterB);
	if (error != LibMatchMexError::success)
		return error;

	if (*parameterA < 0 || *parameterB < 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}
/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  success
*/
LibMatchMexError parse4ElementIntegerParameter(const mxArray *pa,
	int *parameterA1, int *parameterA2,
		int *parameterB1, int *parameterB2)
{
	LibMatchMexError error;
	size_t numberOfElement = mxGetNumberOfElements(pa);
	if (numberOfElement == 1 || numberOfElement == 2)
	{
		error = parse2ElementIntegerParameter(pa, parameterA1, parameterB1);
		*parameterA2 = *parameterA1;
		*parameterB2 = *parameterB1;
		return error;
	}
	else if (numberOfElement == 4)
	{
		return getFourIntegerFromMxArray(pa, parameterA1, parameterA2, parameterB1, parameterB2);
	}
	else
		return LibMatchMexError::errorNumberOfMatrixDimension;
}
/*
* Return:
*  errorNumberOfMatrixDimension
*  errorOverFlow
*  errorInvalidValue
*  success
*/
LibMatchMexError parse4ElementNonNegativeIntegerParameter(const mxArray *pa,
	int *parameterA1, int *parameterA2,
	int *parameterB1, int *parameterB2)
{
	LibMatchMexError error = parse4ElementIntegerParameter(pa, parameterA1, parameterA2, parameterB1, parameterB2);
	if (error != LibMatchMexError::success)
		return error;
	if (*parameterA1 < 0 || *parameterA2 < 0 || *parameterB1 < 0 || *parameterB2 < 0)
		return LibMatchMexError::errorInvalidValue;

	return LibMatchMexError::success;
}

// Update: won't fix
// TODO: int to size_t
LibMatchMexError parse2DMatrixParameter(const mxArray *pa,
	void **pointer,
	int *size_M, int *size_N)
{
	if (!mxIsNumeric(pa))
	{
		return LibMatchMexError::errorTypeOfArgument;
	}

	if (mxGetNumberOfDimensions(pa) > INT_MAX)
		return LibMatchMexError::errorOverFlow;

	int numberOfDimensions = static_cast<int>(mxGetNumberOfDimensions(pa));
	if (numberOfDimensions != 2)
	{
		return LibMatchMexError::errorNumberOfMatrixDimension;
	}

	const size_t *dimensions = mxGetDimensions(pa);
	if (dimensions[0] > INT_MAX || dimensions[1] > INT_MAX)
		return LibMatchMexError::errorOverFlow;

	*size_M = static_cast<int>(dimensions[0]);
	*size_N = static_cast<int>(dimensions[1]);

	*pointer = mxGetData(pa);

	return LibMatchMexError::success;
}
