#include "common.h"

bool isLoaded = false;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	if (!isLoaded) {
		mexAtExit(atExit);

		isLoaded = true;
	}

	struct ArrayMatchMexContext context;

	struct ArrayMatchMexErrorWithMessage errorWithMessage = parseParameter(&context,
		nlhs, plhs,
		nrhs, prhs);

	if (errorWithMessage.error != arrayMatchMexOk)
		mexErrMsgTxt(errorWithMessage.message);

	int lengthOfArray = context.lengthOfArray;
	int numberOfArray = context.numberOfArray;
	int totalSize = lengthOfArray * numberOfArray;
	float *A;
	float *B;
	A = malloc(totalSize * 2 * sizeof(float));
	if (A == NULL)
	{
		snprintf(errorWithMessage.message, ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH, "Malloc failed, need %d bytes.",
			conjectureMaximumMemoryAllocationSize(lengthOfArray, numberOfArray, false));
		mexErrMsgTxt(errorWithMessage.message);
	}
	B = A + totalSize;
	convertDoubleToFloat(context.A, A, totalSize);
	convertDoubleToFloat(context.B, B, totalSize);

	free(A);


}