#include "common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	PaddingMexContext context;
	LibMatchMexErrorWithMessage errorWithMessage = parseParameter(&context, nlhs, plhs, nrhs, prhs);

	if (errorWithMessage.error != LibMatchMexError::success)
	{
		mexErrMsgTxt(errorWithMessage.message);
		return;
	}

	int newM = context.image_M + context.pad_M_pre + context.pad_M_post, 
		newN = context.image_N + context.pad_N_pre + context.pad_N_post;

	plhs[0] = mxCreateDoubleMatrix(newM, newN, mxREAL);
	double *newImageMatrixPointer = mxGetPr(plhs[0]);
	switch (context.method)
	{
	case PadMethod::zero:
		zeroPadding(context.originImage, newImageMatrixPointer, context.image_M, context.image_N,
			context.pad_M_pre, context.pad_M_post, context.pad_N_pre, context.pad_N_post);
		break;
	case PadMethod::circular:
		circularPadding(context.originImage, newImageMatrixPointer, context.image_M, context.image_N,
			context.pad_M_pre, context.pad_M_post, context.pad_N_pre, context.pad_N_post);
		break;
	case PadMethod::replicate:
		replicatePadding(context.originImage, newImageMatrixPointer, context.image_M, context.image_N,
			context.pad_M_pre, context.pad_M_post, context.pad_N_pre, context.pad_N_post);
		break;
	case PadMethod::symmetric:
		symmetricPadding(context.originImage, newImageMatrixPointer, context.image_M, context.image_N,
			context.pad_M_pre, context.pad_M_post, context.pad_N_pre, context.pad_N_post);
		break;
	default: break;
	}
}
