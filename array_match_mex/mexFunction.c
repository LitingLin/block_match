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

	if (errorWithMessage.error != arrayMatchMexOk) {
		mexErrMsgTxt(errorWithMessage.message);
		return;
	}

	int lengthOfArray = context.lengthOfArray;
	int numberOfArray = context.numberOfArray;
	int totalSize = lengthOfArray * numberOfArray;
	float *A;
	float *B;
	A = malloc(totalSize * 2 * sizeof(float));
	if (A == NULL)
	{
		snprintf(errorWithMessage.message, ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH, 
			"Malloc failed, need %d bytes for normal memory allocation, %d bytes for page locked memory allocation.\n",
			getMaximumMemoryAllocationSize(lengthOfArray, numberOfArray) + arrayMatchGetMaximumMemoryAllocationSize(numberOfArray, lengthOfArray),
			arrayMatchGetMaximumPageLockedMemoryAllocationSize(numberOfArray, lengthOfArray));
		mexErrMsgTxt(errorWithMessage.message);
		return;
	}
	B = A + totalSize;
	convertDoubleToFloat(context.A, A, totalSize);
	convertDoubleToFloat(context.B, B, totalSize);

	char buffer[LIB_MATCH_MAX_MESSAGE_LENGTH + ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH];

	void *arrayMatchInstance;
	enum ErrorCode errorCode = arrayMatchInitialize(&arrayMatchInstance, numberOfArray, lengthOfArray);
	if (errorCode != LibMatchErrorOk)
	{		
		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		if (errorCode == LibMatchErrorMemoryAllocation || errorCode == LibMatchErrorPageLockedMemoryAllocation)
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n"
				"Memory allocation failed, need %d bytes for normal memory allocation, %d bytes for page locked memory allocation.\n",
				buffer, getMaximumMemoryAllocationSize(lengthOfArray, numberOfArray) + arrayMatchGetMaximumMemoryAllocationSize(numberOfArray, lengthOfArray),
				arrayMatchGetMaximumPageLockedMemoryAllocationSize(numberOfArray, lengthOfArray));
		else if (errorCode == LibMatchErrorGpuMemoryAllocation)
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n"
				"Gpu Memory allocation failed, need %d bytes.\n",
				buffer, arrayMatchGetMaximumGpuMemoryAllocationSize(numberOfArray, lengthOfArray));
		else
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n",
				buffer);
		mexErrMsgTxt(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH);
		return;
	}
	
	float *result;
	errorCode = arrayMatchExecute(arrayMatchInstance, A, B, context.method, &result);
	if (errorCode != LibMatchErrorOk)
	{
		arrayMatchFinalize(arrayMatchInstance);
		free(A);

		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		mexErrMsgTxt(buffer);
		return;
	}

	free(A);

	plhs[0] = mxCreateDoubleMatrix(numberOfArray, 1, mxREAL);
	double *mxResult = mxGetPr(plhs[0]);
	convertFloatToDouble(result, mxResult, numberOfArray);

	errorCode = arrayMatchFinalize(arrayMatchInstance);
	if (errorCode != LibMatchErrorOk)
	{
		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		mexErrMsgTxt(buffer);
		return;
	}
}