#include "common.h"
#include "utils.h"
#include <string.h>


template <typename IntermidateType, typename ResultType>
void process(BlockMatchMexContext *context,int nlhs, mxArray *plhs[])
{
	int sequenceASize = context->sequenceAMatrixDimensions[0] * context->sequenceAMatrixDimensions[1];
	int sequenceBSize = context->sequenceBMatrixDimensions[0] * context->sequenceBMatrixDimensions[1];

	IntermidateType *sequenceAPointer_converted, *sequenceBPointer_converted;
	sequenceAPointer_converted = static_cast<IntermidateType*>(malloc(sequenceASize * sizeof(IntermidateType)));

	if (!sequenceAPointer_converted) {
		mexErrMsgTxt("Memory allocation failed: in malloc for sequenceAPointer_converted.");
		return;
	}

	sequenceBPointer_converted = static_cast<IntermidateType*>(malloc(sequenceBSize * sizeof(IntermidateType)));

	if (!sequenceBPointer_converted)
	{
		free(sequenceAPointer_converted);
		mexErrMsgTxt("Memory allocation failed: in malloc for sequenceBPointer_converted.");
		return;
	}

	convertArrayType(context->sourceAType, context->intermediateType, context->sequenceAMatrixPointer, sequenceAPointer_converted, sequenceASize);
	convertArrayType(context->sourceBType, context->intermediateType, context->sequenceBMatrixPointer, sequenceBPointer_converted, sequenceBSize);

	char errorStringBuffer[LIB_MATCH_MAX_MESSAGE_LENGTH];

	void *instance;
	int matrixC_M, matrixC_N, matrixC_O;
	int matrixA_padded_M, matrixA_padded_N,
		matrixB_padded_M, matrixB_padded_N;
	if (!blockMatchInitialize<IntermidateType>(&instance, context->searchType, context->method, context->padMethodA, context->padMethodB, context->sequenceABorderType,
		context->sort,
		context->sequenceAMatrixDimensions[1], context->sequenceAMatrixDimensions[0], context->sequenceBMatrixDimensions[1], context->sequenceBMatrixDimensions[0],
		context->searchRegionWidth, context->searchRegionHeight,
		context->blockWidth, context->blockHeight,
		context->sequenceAStrideWidth, context->sequenceAStrideHeight,
		context->sequenceBStrideWidth, context->sequenceBStrideHeight,
		context->sequenceAPaddingWidthPre, context->sequenceAPaddingWidthPost,
		context->sequenceAPaddingHeightPre, context->sequenceAPaddingHeightPost,
		context->sequenceBPaddingWidthPre, context->sequenceBPaddingWidthPost,
		context->sequenceBPaddingHeightPre, context->sequenceBPaddingHeightPost,
		context->retain,
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
	IntermidateType *matrixC = static_cast<IntermidateType*>(malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(IntermidateType)));
	if (!matrixC)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for matrixC");
		goto matrixC_mallocFailed;
	}
	IntermidateType *matrixA_padded = static_cast<IntermidateType*>(malloc(matrixA_padded_M * matrixA_padded_N * sizeof(IntermidateType)));
	if (!matrixA_padded)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for matrixA_padded");
		goto matrixA_padded_mallocFailed;
	}
	IntermidateType *matrixB_padded = static_cast<IntermidateType*>(malloc(matrixB_padded_M * matrixB_padded_N * sizeof(IntermidateType)));
	if (!matrixB_padded)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for matrixB_padded");
		goto matrixB_padded_mallocFailed;
	}
	int *index_x = static_cast<int*>(malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(IntermidateType)));
	if (!index_x)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for index_x");
		goto index_x_mallocFailed;
	}
	int *index_y = static_cast<int*>(malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(IntermidateType)));
	if (!index_y)
	{
		strcpy_s(errorStringBuffer, "Memory allocation failed: in malloc for index_y");
		goto index_y_mallocFailed;
	}
	if (!blockMatchExecute(instance, sequenceAPointer_converted, sequenceBPointer_converted, matrixC, matrixA_padded, matrixB_padded, index_x, index_y))
	{
		libMatchGetLastErrorString(errorStringBuffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
		blockMatchFinalize<IntermidateType>(instance);
		goto runtime_error;
	}

	blockMatchFinalize<IntermidateType>(instance);

	free(sequenceAPointer_converted);
	free(sequenceBPointer_converted);

	if (!generate_result<IntermidateType, ResultType>(&plhs[0], matrixC_N, matrixC_M, index_y, index_x, matrixC, matrixC_O))
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
		if (!generatePaddedMatrix<IntermidateType,ResultType>(&plhs[1], matrixA_padded_N, matrixA_padded_M, matrixA_padded))
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
		if (!generatePaddedMatrix<IntermidateType, ResultType>(&plhs[2], matrixB_padded_N, matrixB_padded_M, matrixB_padded))
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
	
	/*errorMessage = validateParameter(&context);

	if (errorMessage.error != LibMatchMexError::success)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}*/

	if (context.intermediateType == typeid(float) && context.resultType == typeid(double))
		process<float, double>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(float) && context.resultType == typeid(float))
		process<float, float>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(double) && context.resultType == typeid(float))
		process<double, float>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(double) && context.resultType == typeid(double))
		process<double, double>(&context, nlhs, plhs);
	else
		mexErrMsgTxt("Processing data type can only be float or double");
}
