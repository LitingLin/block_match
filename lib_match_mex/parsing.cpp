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

// TODO: int to size_t
LibMatchMexError parse2ElementIntegerParameter(const mxArray *pa,
	int *parameterA, int *parameterB)
{
	int a, b;

	if (mxGetNumberOfElements(pa) == 1)
	{
		double tempValue = mxGetScalar(pa);
		if (tempValue > INT_MAX)
			return LibMatchMexError::errorOverFlow;
		
		a = b = static_cast<int>(tempValue);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = static_cast<const double*>(mxGetData(pa));
		if (pr[0] > INT_MAX || pr[1] > INT_MAX)
			return LibMatchMexError::errorOverFlow;

		a = static_cast<int>(pr[0]);
		b = static_cast<int>(pr[1]);
	}
	else
		return LibMatchMexError::errorSizeOfArray;
	
	*parameterA = a, *parameterB = b;

	return LibMatchMexError::success;
}

// TODO: int to size_t
LibMatchMexError parse2DMatrixParameter(const mxArray *pa,
	double **pointer,
	int *size_M, int *size_N)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return LibMatchMexError::errorTypeOfArgument;
	}

	int numberOfDimensions = mxGetNumberOfDimensions(pa);
	if (numberOfDimensions != 2)
	{
		return LibMatchMexError::errorNumberOfMatrixDimension;
	}

	const size_t *dimensions = mxGetDimensions(pa);
	if (dimensions[0] > INT_MAX || dimensions[1] > INT_MAX)
		return LibMatchMexError::errorOverFlow;

	*size_M = static_cast<int>(dimensions[0]);
	*size_N = static_cast<int>(dimensions[1]);

	*pointer = static_cast<double*>(mxGetData(pa));

	return LibMatchMexError::success;
}
