#include "common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	libMatchMexInitalize();
	struct LibBlockMatchMexContext context;
	struct LibMatchMexErrorWithMessage errorMessage = parseParameter(&context, nlhs, plhs, nrhs, prhs);

	if (errorMessage.error != libMatchMexOk)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}
	errorMessage = validateParameter(&context);

	if (errorMessage.error != libMatchMexOk)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}

	int sequenceASize = context.sequenceAMatrixDimensions[0] * context.sequenceAMatrixDimensions[1];
	int sequenceBSize = context.sequenceBMatrixDimensions[0] * context.sequenceBMatrixDimensions[1];

	float *sequenceAPointer_converted, *sequenceBPointer_converted;
	sequenceAPointer_converted = malloc(sequenceASize * sizeof(float));
	if (!sequenceAPointer_converted) {
		mexErrMsgTxt("malloc failed\n");
		return;
	}

	sequenceBPointer_converted = malloc(sequenceBSize * sizeof(float));
	if (!sequenceBPointer_converted)
	{
		free(sequenceAPointer_converted);
		mexErrMsgTxt("malloc failed\n");
		return;
	}

	convertArrayFromDoubleToFloat(context.sequenceAMatrixPointer, sequenceAPointer_converted, sequenceASize);
	convertArrayFromDoubleToFloat(context.sequenceBMatrixPointer, sequenceBPointer_converted, sequenceBSize);

	void *instance;
	if (!initialize(&instance, context.sequenceAMatrixDimensions[1], context.sequenceAMatrixDimensions[0], context.sequenceBMatrixDimensions[1], context.sequenceBMatrixDimensions[0],
		context.searchRegionWidth, context.searchRegionHeight,
		context.blockWidth, context.blockHeight, context.sequenceAStrideWidth, context.sequenceAStrideHeight, context.sequenceBStrideWidth, context.sequenceBStrideHeight,
		context.sequenceAPaddingWidth, context.sequenceAPaddingHeight, context.sequenceBPaddingWidth, context.sequenceBPaddingHeight,
		context.retain))
	{
		free(sequenceAPointer_converted);
		free(sequenceBPointer_converted);
		mexErrMsgTxt("malloc failed\n");

		return;
	}
	float *result;
	int result_dims[4];
	int *index_x, *index_y;
	if (!execute(instance, sequenceAPointer_converted, sequenceBPointer_converted, context.method, &index_y, &index_x, &result, result_dims))
	{
		free(sequenceAPointer_converted);
		free(sequenceBPointer_converted);
		finalize(instance);
		mexErrMsgTxt("unknown cuda error\n");
		return;
	}

	if (!generate_result(&plhs[0], result_dims[1], result_dims[0], index_x, index_y, result, result_dims[2]))
	{
		finalize(instance);

		free(sequenceAPointer_converted);
		free(sequenceBPointer_converted);

		mexErrMsgTxt("malloc failed");
		return;
	}

	finalize(instance);

	free(sequenceAPointer_converted);
	free(sequenceBPointer_converted);
}