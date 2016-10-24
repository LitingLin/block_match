#include "common.h"

extern "C"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	libMatchMexInitalize();

	struct ArrayMatchMexContext context;

	struct LibMatchMexErrorWithMessage errorWithMessage = parseParameter(&context,
		nlhs, plhs,
		nrhs, prhs);

	if (errorWithMessage.error != LibMatchMexError::success) {
		mexErrMsgTxt(errorWithMessage.message);
		return;
	}

	int lengthOfArray = context.lengthOfArray;
	int numberOfArray = context.numberOfArray;
	int totalSize = lengthOfArray * numberOfArray;
	float *A;
	float *B;
	A = static_cast<float *>(malloc(totalSize * 2 * sizeof(float)));
	if (A == nullptr)
	{
		snprintf(errorWithMessage.message, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, 
			"Malloc failed, need %d bytes for normal memory allocation, %d bytes for page locked memory allocation.\n",
			getMaximumMemoryAllocationSize(lengthOfArray, numberOfArray) + arrayMatchGetMaximumMemoryAllocationSize(numberOfArray, lengthOfArray),
			arrayMatchGetMaximumPageLockedMemoryAllocationSize(numberOfArray, lengthOfArray));
		mexErrMsgTxt(errorWithMessage.message);
		return;
	}
	B = A + totalSize;
	convertArrayType(context.A, A, totalSize);
	convertArrayType(context.B, B, totalSize);

	char buffer[LIB_MATCH_MAX_MESSAGE_LENGTH + LIB_MATCH_MEX_MAX_MESSAGE_LENGTH];

	void *arrayMatchInstance;
	LibMatchErrorCode errorCode = arrayMatchInitialize(&arrayMatchInstance, numberOfArray, lengthOfArray);
	if (errorCode != LibMatchErrorCode::success)
	{		
		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		if (errorCode == LibMatchErrorCode::errorMemoryAllocation || errorCode == LibMatchErrorCode::errorPageLockedMemoryAllocation)
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n"
				"Memory allocation failed, need %d bytes for normal memory allocation, %d bytes for page locked memory allocation.\n",
				buffer, getMaximumMemoryAllocationSize(lengthOfArray, numberOfArray) + arrayMatchGetMaximumMemoryAllocationSize(numberOfArray, lengthOfArray),
				arrayMatchGetMaximumPageLockedMemoryAllocationSize(numberOfArray, lengthOfArray));
		else if (errorCode == LibMatchErrorCode::errorGpuMemoryAllocation)
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n"
				"Gpu Memory allocation failed, need %d bytes.\n",
				buffer, arrayMatchGetMaximumGpuMemoryAllocationSize(numberOfArray, lengthOfArray));
		else
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n",
				buffer);
		mexErrMsgTxt(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH);
		return;
	}
	
	float *result;
	errorCode = arrayMatchExecute(arrayMatchInstance, A, B, context.method, &result);
	if (errorCode != LibMatchErrorCode::success)
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
	convertArrayType(result, mxResult, numberOfArray);

	errorCode = arrayMatchFinalize(arrayMatchInstance);
	if (errorCode != LibMatchErrorCode::success)
	{
		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		mexErrMsgTxt(buffer);
		return;
	}
}