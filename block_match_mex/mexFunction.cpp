#include "common.h"
#include <string.h>

extern "C"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	libMatchMexInitalize();
	struct BlockMatchMexContext context;
	struct LibMatchMexErrorWithMessage errorMessage = parseParameter(&context, nlhs, plhs, nrhs, prhs);

	if (errorMessage.error != LibMatchMexError::success)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}
	errorMessage = validateParameter(&context);

	if (errorMessage.error != LibMatchMexError::success)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}

	int sequenceASize = context.sequenceAMatrixDimensions[0] * context.sequenceAMatrixDimensions[1];
	int sequenceBSize = context.sequenceBMatrixDimensions[0] * context.sequenceBMatrixDimensions[1];

	float *sequenceAPointer_converted, *sequenceBPointer_converted;
	sequenceAPointer_converted = static_cast<float*>(malloc(sequenceASize * sizeof(float)));
	if (!sequenceAPointer_converted) {
		mexErrMsgTxt("malloc failed\n");
		return;
	}

	sequenceBPointer_converted = static_cast<float*>(malloc(sequenceBSize * sizeof(float)));
	if (!sequenceBPointer_converted)
	{
		free(sequenceAPointer_converted);
		mexErrMsgTxt("malloc failed\n");
		return;
	}

	convertArrayFromDoubleToFloat(context.sequenceAMatrixPointer, sequenceAPointer_converted, sequenceASize);
	convertArrayFromDoubleToFloat(context.sequenceBMatrixPointer, sequenceBPointer_converted, sequenceBSize);

	char errorStringBuffer[LIB_MATCH_MAX_MESSAGE_LENGTH];

	void *instance;
	int matrixC_M, matrixC_N, matrixC_O;
	int matrixA_padded_M, matrixA_padded_N,
		matrixB_padded_M, matrixB_padded_N;
	if (!blockMatchAndSortingInitialize(&instance, context.searchType, context.method, PadMethod::symmetric,
		context.sequenceAMatrixDimensions[1], context.sequenceAMatrixDimensions[0], context.sequenceBMatrixDimensions[1], context.sequenceBMatrixDimensions[0],
		context.searchRegionWidth, context.searchRegionHeight,
		context.blockWidth, context.blockHeight,
		context.sequenceAStrideWidth, context.sequenceAStrideHeight,
		context.sequenceBStrideWidth, context.sequenceBStrideHeight,
		context.sequenceAPaddingWidth, context.sequenceAPaddingWidth,
		context.sequenceAPaddingHeight, context.sequenceAPaddingHeight,
		context.sequenceBPaddingWidth, context.sequenceBPaddingWidth,
		context.sequenceBPaddingHeight, context.sequenceBPaddingHeight,
		context.retain,
		&matrixC_M, &matrixC_N, &matrixC_O,
		&matrixA_padded_M, &matrixA_padded_N,
		&matrixB_padded_M, &matrixB_padded_N))
	{
		libMatchGetLastErrorString(errorStringBuffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		free(sequenceAPointer_converted);
		free(sequenceBPointer_converted);
		mexErrMsgTxt(errorStringBuffer);

		return;
	}
	float *matrixC = (float*)malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(float));
	if (!matrixC)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for matrixC");
		goto matrixC_mallocFailed;
	}
	float *matrixA_padded = (float*)malloc(matrixA_padded_M * matrixA_padded_N * sizeof(float));
	if (!matrixA_padded)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for matrixA_padded");
		goto matrixA_padded_mallocFailed;
	}
	float *matrixB_padded = (float*)malloc(matrixB_padded_M * matrixB_padded_N * sizeof(float));
	if (!matrixB_padded)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for matrixB_padded");
		goto matrixB_padded_mallocFailed;
	}
	int *index_x = (int*)malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(float));
	if (!index_x)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for index_x");
		goto index_x_mallocFailed;
	}
	int *index_y = (int*)malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(float));
	if (!index_y)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for index_y");
		goto index_y_mallocFailed;
	}
	if (!blockMatchExecute(instance, sequenceAPointer_converted, sequenceBPointer_converted, matrixC, matrixA_padded, matrixB_padded, index_x, index_y))
	{
		libMatchGetLastErrorString(errorStringBuffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		blockMatchFinalize(instance);
		goto runtime_error;
	}

	blockMatchFinalize(instance);

	free(sequenceAPointer_converted);
	free(sequenceBPointer_converted);

	if (!generate_result(&plhs[0], matrixC_N, matrixC_M, index_y, index_x, matrixC, matrixC_O))
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc result");

		free(index_y);
		free(index_x);
		free(matrixC);
		free(matrixB_padded);
		free(matrixA_padded);

		goto generateErrorMessageAndExit;
	}

	free(index_y);
	free(index_x);
	free(matrixC);

	if (nlhs > 1)
	{
		if (!generatePaddedMatrix(&plhs[1], matrixA_padded_N, matrixA_padded_N, matrixA_padded))
		{
			strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc matrixAPadded");
			free(matrixB_padded);
			free(matrixA_padded);

			goto generateErrorMessageAndExit;
		}
	}
	free(matrixA_padded);

	if (nlhs > 2)
	{
		if (!generatePaddedMatrix(&plhs[2], matrixB_padded_N, matrixB_padded_N, matrixB_padded))
		{
			strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc matrixBPadded");
			free(matrixB_padded);

			goto generateErrorMessageAndExit;
		}
	}

	free(matrixB_padded);
	return;


runtime_error:
	free(index_y);

index_y_mallocFailed:
	free(index_x);

index_x_mallocFailed:
	free(matrixB_padded);
matrixB_padded_mallocFailed:
	free(matrixA_padded);
matrixA_padded_mallocFailed:
	free(matrixC);
matrixC_mallocFailed:

	free(sequenceAPointer_converted);
	free(sequenceBPointer_converted);
generateErrorMessageAndExit:
	mexErrMsgTxt(errorStringBuffer);
}
