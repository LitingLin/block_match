#include "common.h"

enum LibBlockMatchMexError
{
	blockMatchMexOk,
	blockMatchMexErrorNumberOfOutputArguments,
	blockMatchMexErrorTypeOfArgument,
	blockMatchMexErrorNumberOfMatrixDimension,
	blockMatchMexErrorNumberOfMatrixDimensionMismatch,
	blockMatchMexErrorSizeOfMatrixDimension,

};

struct LibBlockMatchMexErrorWithMessage
{
	enum LibBlockMatchMexError error;
	char message[32];
};

struct LibBlockMatchMexContext
{
	enum Method method;
	size_t sequenceMatrixNumberOfDimensions;

	size_t sequenceAMatrixDimensions[4];
	double *sequenceAMatrixPointer;

	size_t sequenceBMatrixDimensions[4];
	double *sequenceBMatrixPointer;

	size_t blockWidth;
	size_t blockHeight;
	size_t searchRegionWidth;
	size_t searchRegionHeight;

	size_t sequenceBStrideWidth;
	size_t sequenceBStrideHeight;
};

enum LibBlockMatchMexError parseSequenceBStrideSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t sequenceBStrideWidth, sequenceBStrideHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		sequenceBStrideWidth = sequenceBStrideHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		sequenceBStrideHeight = pr[0];
		sequenceBStrideWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceBStrideWidth = sequenceBStrideWidth;
	context->sequenceBStrideHeight = sequenceBStrideHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSearchRegionSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t searchRegionWidth, searchRegionHeight;

	if (mxGetNumberOfElements(pa) == 1)
	{
		searchRegionWidth = searchRegionHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		searchRegionHeight = pr[0];
		searchRegionWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	if (context->sequenceAMatrixDimensions[0] - context->blockWidth < searchRegionWidth || context->sequenceAMatrixDimensions[1] - context->blockHeight < searchRegionHeight)
	{
		return blockMatchMexErrorSizeOfMatrixDimension;
	}
	context->searchRegionWidth = searchRegionWidth;
	context->searchRegionHeight = searchRegionHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseBlockSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t blockWidth, blockHeight;
	if (mxGetNumberOfElements(pa) == 1)
	{
		blockWidth = blockHeight = mxGetScalar(pa);
	}
	else if (mxGetNumberOfElements(pa) == 2)
	{
		const double *pr = mxGetData(pa);
		blockHeight = pr[0];
		blockWidth = pr[1];
	}
	else
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	if (context->sequenceAMatrixDimensions[0] < blockWidth || context->sequenceAMatrixDimensions[1] < blockHeight)
	{
		return blockMatchMexErrorSizeOfMatrixDimension;
	}
	
	context->blockWidth = blockWidth;
	context->blockHeight = blockHeight;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequence2(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return blockMatchMexErrorTypeOfArgument;
	}

	size_t sequenceBMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);

	if (sequenceBMatrixNumberOfDimensions != context->sequenceMatrixNumberOfDimensions)
	{
		return blockMatchMexErrorNumberOfMatrixDimensionMismatch;
	}

	const size_t *sequenceBMatrixDimensions = mxGetDimensions(pa);
	context->sequenceBMatrixDimensions[1] = sequenceBMatrixDimensions[0];
	context->sequenceBMatrixDimensions[0] = sequenceBMatrixDimensions[1];

	if (sequenceBMatrixDimensions[0] < context->sequenceAMatrixDimensions[0] || sequenceBMatrixDimensions[1] < context->sequenceAMatrixDimensions[1])
	{
		return blockMatchMexErrorSizeOfMatrixDimension;
	}

	context->sequenceBMatrixPointer = mxGetData(pa);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequenceA(struct LibBlockMatchMexContext *context, 
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return blockMatchMexErrorTypeOfArgument;
	}

	size_t sequenceAMatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequenceAMatrixNumberOfDimensions != 2)
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceMatrixNumberOfDimensions = sequenceAMatrixNumberOfDimensions;

	const size_t *sequenceAMatrixDimensions = mxGetDimensions(pa);
	context->sequenceAMatrixDimensions[1] = sequenceAMatrixDimensions[0];
	context->sequenceAMatrixDimensions[0] = sequenceAMatrixDimensions[1];

	context->sequenceAMatrixPointer = mxGetData(pa);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseOutputArgument(struct LibBlockMatchMexContext *context, 
	int nlhs, mxArray *plhs[])
{
	if (nlhs >= 2)
		return blockMatchMexErrorNumberOfOutputArguments;

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseParameter(struct LibBlockMatchMexContext *context,
	int nlhs, mxArray *plhs[], 
	int nrhs, const mxArray *prhs[])
{
	enum LibBlockMatchMexError error = parseOutputArgument(context, nlhs, plhs);
	if (error != blockMatchMexOk)
		return error;

	return blockMatchMexOk;
}