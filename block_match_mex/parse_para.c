#include "common.h"

enum LibBlockMatchMexError
{
	blockMatchMexOk,
	blockMatchMexErrorNumberOfOutputArguments,
	blockMatchMexErrorTypeOfArgument,
	blockMatchMexErrorNumberOfMatrixDimension,
	blockMatchMexErrorNumberOfMatrixDimensionMismatch,
};

struct LibBlockMatchMexContext
{
	enum Method method;
	size_t sequenceMatrixNumberOfDimensions;

	size_t sequence1MatrixDimensions[4];
	double *sequence1MatrixPointer;

	size_t sequence2MatrixDimensions[4];
	double *sequence2MatrixPointer;

	size_t blockWidth;
	size_t blockHeight;
	size_t neighbourWidth;
	size_t neighbourHeight;

	size_t sequence2StrideWidth;
	size_t sequence2StrideHeight;
};

enum LibBlockMatchMexError parseBlockSize(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	size_t blockWidth, blockHeight;
	if (mxGetNumberOfElements(prhs[index]) == 1)
	{
		blockWidth = blockHeight = mxGetScalar(prhs[index]);
	}
	else if (mxGetNumberOfElements(prhs[index]) == 2)
	{
		const double *pr = mxGetData(prhs[index]);
		blockHeight = pr[0];
		blockWidth = pr[1];
	}
	else
	{
		mexErrMsgTxt("check failed: dim of blockSize should be 1 or 2\n");
		return;
	}

	if (sequence1MatrixDimensions[0] < blockWidth || sequence1MatrixDimensions[1] < blockHeight)
	{
		mexErrMsgTxt("check failed: size of matrix(para1) smaller than blockSize\n");
		return;
	}

	if (sequence2MatrixDimensions[0] < blockWidth || sequence2MatrixDimensions[1] < blockHeight)
	{
		mexErrMsgTxt("check failed: size of matrix(para2) smaller than blockSize\n");
		return;
	}

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequence2(struct LibBlockMatchMexContext *context,
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return blockMatchMexErrorTypeOfArgument;
	}

	size_t sequence2MatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);

	if (sequence2MatrixNumberOfDimensions != context->sequenceMatrixNumberOfDimensions) {
		return blockMatchMexErrorNumberOfMatrixDimensionMismatch;
	}

	const size_t *sequence2MatrixDimension = mxGetDimensions(pa);
	context->sequence2MatrixDimensions[1] = sequence2MatrixDimension[0];
	context->sequence2MatrixDimensions[0] = sequence2MatrixDimension[1];

	context->sequence2MatrixPointer = mxGetData(pa);

	return blockMatchMexOk;
}

enum LibBlockMatchMexError parseSequence1(struct LibBlockMatchMexContext *context, 
	const mxArray *pa)
{
	if (mxGetClassID(pa) != mxDOUBLE_CLASS)
	{
		return blockMatchMexErrorTypeOfArgument;
	}

	size_t sequence1MatrixNumberOfDimensions = mxGetNumberOfDimensions(pa);
	if (sequence1MatrixNumberOfDimensions != 2)
	{
		return blockMatchMexErrorNumberOfMatrixDimension;
	}

	context->sequenceMatrixNumberOfDimensions = sequence1MatrixNumberOfDimensions;

	const size_t *sequenceDimension = mxGetDimensions(pa);
	context->sequence1MatrixDimensions[1] = sequenceDimension[0];
	context->sequence1MatrixDimensions[0] = sequenceDimension[1];

	context->sequence1MatrixPointer = mxGetData(pa);

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