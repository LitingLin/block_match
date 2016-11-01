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
	int numberOfArrayA = context.numberOfArrayA;
	int numberOfArrayB = context.numberOfArrayB;
	int sizeOfA = lengthOfArray * numberOfArrayA;
	int sizeOfB = lengthOfArray * numberOfArrayB;
	float *A;
	float *B;
	A = static_cast<float *>(malloc((sizeOfA + sizeOfB) * sizeof(float)));
	if (A == nullptr)
	{
		snprintf(errorWithMessage.message, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, 
			"Malloc failed, need %zd bytes for normal memory allocation, %zd bytes for page locked memory allocation.\n",
			getMaximumMemoryAllocationSize(lengthOfArray, numberOfArrayA, numberOfArrayB) + arrayMatchGetMaximumMemoryAllocationSize(),
			arrayMatchGetMaximumPageLockedMemoryAllocationSize(numberOfArrayA, numberOfArrayA, lengthOfArray));
		mexErrMsgTxt(errorWithMessage.message);
		return;
	}
	B = A + sizeOfA;
	convertArrayType(context.A, A, sizeOfA);
	convertArrayType(context.B, B, sizeOfB);

	char buffer[LIB_MATCH_MAX_MESSAGE_LENGTH + LIB_MATCH_MEX_MAX_MESSAGE_LENGTH];

	void *arrayMatchInstance;
	LibMatchErrorCode errorCode = arrayMatchInitialize(&arrayMatchInstance, numberOfArrayA, numberOfArrayB, lengthOfArray);
	if (errorCode != LibMatchErrorCode::success)
	{		
		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		if (errorCode == LibMatchErrorCode::errorMemoryAllocation || errorCode == LibMatchErrorCode::errorPageLockedMemoryAllocation)
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n"
				"Memory allocation failed, need %zd bytes for normal memory allocation, %zd bytes for page locked memory allocation.\n",
				buffer, getMaximumMemoryAllocationSize(lengthOfArray, numberOfArrayA, numberOfArrayB) + arrayMatchGetMaximumMemoryAllocationSize(),
				arrayMatchGetMaximumPageLockedMemoryAllocationSize(numberOfArrayA,numberOfArrayB, lengthOfArray));
		else if (errorCode == LibMatchErrorCode::errorGpuMemoryAllocation)
			snprintf(buffer + LIB_MATCH_MAX_MESSAGE_LENGTH, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, "%s\n"
				"Gpu Memory allocation failed, need %zd bytes.\n",
				buffer, arrayMatchGetMaximumGpuMemoryAllocationSize(lengthOfArray));
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
	
	plhs[0] = mxCreateDoubleMatrix(numberOfArrayA * numberOfArrayB, 1, mxREAL);
	double *mxResult = mxGetPr(plhs[0]);
	convertArrayType(result, mxResult, numberOfArrayA * numberOfArrayB);

	errorCode = arrayMatchFinalize(arrayMatchInstance);
	if (errorCode != LibMatchErrorCode::success)
	{
		libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		mexErrMsgTxt(buffer);
		return;
	}
}