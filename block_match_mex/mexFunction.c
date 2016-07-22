#include "common.h"

#include <memory.h>
#include <string.h>

#include "mxUtils.h"
#include "utils.h"


bool onLoaded = false;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	if (!onLoaded) {
		mexAtExit(atExit);
		onLoaded = true;
	}

	if (nlhs > 1)
	{
		mexErrMsgTxt("err: number of func return para\n");
		return;
	}

	const uint32_t numPara = 3;

	uint32_t index = 0;

	if (!(nrhs == numPara + 1 || nrhs == numPara + 2))
	{
		mexErrMsgTxt("err: number of func para\n"
			"input:\t(sequence, [sequence2], blockSize, neighborSize, measure, precision) or\n"
			"output:\t(block)\n");
		return;
	}

	if (mxGetClassID(prhs[index]) != mxDOUBLE_CLASS)
	{
		mexErrMsgTxt("check failed: type of matrix(para 1) must be double\n");
		return;
	}

	size_t sequence1MatrixNumberOfDimensions = mxGetNumberOfDimensions(prhs[index]);
	if (sequence1MatrixNumberOfDimensions != 2) {
		mexErrMsgTxt("check failed: dim of matrix(para 1) must be 2\n");
		return;
	}

	const size_t *sequenceDimension = mxGetDimensions(prhs[index]);
	size_t sequence1MatrixDimensions[4];
	sequence1MatrixDimensions[1] = sequenceDimension[0];
	sequence1MatrixDimensions[0] = sequenceDimension[1];

	const double *sequence1MatrixPointer = mxGetData(prhs[index]);

	const double *sequence2MatrixPointer;
	size_t sequence2MatrixNumberOfDimensions;
	size_t sequence2MatrixDimensions[4];
	if (nrhs == numPara + 2)
	{
		++index;
		sequence2MatrixNumberOfDimensions = mxGetNumberOfDimensions(prhs[index]);

		if (sequence2MatrixNumberOfDimensions != sequence1MatrixNumberOfDimensions) {
			mexErrMsgTxt("check failed: dim of matrix(para 1) != dim of matrix(para 2)\n");
			return;
		}
		const size_t *sequence2MatrixDimension = mxGetDimensions(prhs[index]);
		sequence2MatrixDimensions[1] = sequence2MatrixDimension[0];
		sequence2MatrixDimensions[0] = sequence2MatrixDimension[1];

		sequence2MatrixPointer = mxGetData(prhs[index]);
	}
	else
	{
		sequence2MatrixPointer = sequence1MatrixPointer;
		memcpy(sequence2MatrixDimensions, sequence1MatrixDimensions, sizeof(sequence2MatrixDimensions));
		sequence2MatrixNumberOfDimensions = sequence1MatrixNumberOfDimensions;
	}

	++index;

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

	++index;

	size_t neighbourWidth, neighbourHeight;
	if (mxGetNumberOfElements(prhs[index]) == 1)
	{
		neighbourWidth = neighbourHeight = mxGetScalar(prhs[index]);
	}
	else if (mxGetNumberOfElements(prhs[index]) == 2)
	{
		const double *pr = mxGetData(prhs[index]);
		neighbourHeight = pr[0];
		neighbourWidth = pr[1];
	}
	else
	{
		mexErrMsgTxt("check failed: dim of neighbourSize should be 1 or 2\n");
		return;
	}

	if (sequence1MatrixDimensions[0] < neighbourWidth || sequence1MatrixDimensions[1] < neighbourHeight)
	{
		mexErrMsgTxt("check failed: size of matrix(para1) smaller than neighbourSize\n");
		return;
	}

	if (sequence2MatrixDimensions[0] < neighbourWidth || sequence2MatrixDimensions[1] < neighbourHeight)
	{
		mexErrMsgTxt("check failed: size of matrix(para2) smaller than neighbourSize\n");
		return;
	}

	++index;


	enum Method method;
	char charBuffer[4];
	if (mxGetString(prhs[index], charBuffer, 4))
	{
		mexErrMsgTxt("check failed: meature type must be string\n");
		return;
	}
	if (strncmp(charBuffer, "mse", 4) == 0)
	{
		method = MSE;
	}
	else if (strncmp(charBuffer, "cc", 4) == 0)
	{
		method = CC;
	}
	else
	{
		mexErrMsgTxt("check failed: meature method must be 'mse' or 'cc'\n");
		return;
	}

	++index;

	size_t stride_M = 1, stride_N = 1;
	
	float *sequence1, *sequence2;
	size_t sequence1Size = sequence1MatrixDimensions[0] * sequence1MatrixDimensions[1];
	size_t sequence2Size = sequence2MatrixDimensions[0] * sequence2MatrixDimensions[1];
	sequence1 = malloc(sequence1Size * sizeof(float));
	if (!sequence1) {
		mexErrMsgTxt("malloc failed\n");
		return;
	}

	sequence2 = malloc(sequence2Size * sizeof(float));
	if (!sequence2)
	{
		free(sequence1);
		mexErrMsgTxt("malloc failed\n");
		return;
	}

	doubleToFloat(sequence1MatrixPointer, sequence1, sequence1Size);
	doubleToFloat(sequence2MatrixPointer, sequence2, sequence1Size);

	void *instance;
	if (!initialize(&instance, sequence1MatrixDimensions[0], sequence1MatrixDimensions[1], sequence2MatrixDimensions[0], sequence2MatrixDimensions[1],
		blockWidth, blockHeight, neighbourWidth, neighbourHeight, stride_M, stride_N))
	{
		if (!reset())
		{
			free(sequence1);
			free(sequence2);
			mexErrMsgTxt("reset failed");
			return;
		}
		if (!initialize(&instance, sequence1MatrixDimensions[0], sequence1MatrixDimensions[1], sequence2MatrixDimensions[0], sequence2MatrixDimensions[1],
			blockWidth, blockHeight, neighbourWidth, neighbourHeight, stride_M, stride_N))
		{
			free(sequence1);
			free(sequence2);
			mexErrMsgTxt("malloc failed\n");

			return;
		}
	}
	float *result;
	int result_dims[4];
	if (!process(instance, sequence1, sequence2, method, &result, result_dims))
	{
		free(sequence1);
		free(sequence2);
		finalize(instance);
		mexErrMsgTxt("unknown cuda error\n");
		return;
	}
	mxArray *block = mxCreateNumericArray(4, result_dims, mxDOUBLE_CLASS, mxREAL);
	void *block_p = mxGetData(block);
	floatToDouble(result, block_p, result_dims[0] * result_dims[1] * result_dims[2] * result_dims[3]);

	plhs[0] = block;

	finalize(instance);

	free(sequence1);
	free(sequence2);
}