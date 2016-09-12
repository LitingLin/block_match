#pragma once

#include <lib_match_mex_common.h>

struct LibBlockMatchMexContext
{
	enum Method method;
	int sequenceMatrixNumberOfDimensions;

	int sequenceAMatrixDimensions[4];
	double *sequenceAMatrixPointer;

	int sequenceBMatrixDimensions[4];
	double *sequenceBMatrixPointer;

	int blockWidth;
	int blockHeight;
	int searchRegionWidth;
	int searchRegionHeight;

	int sequenceAStrideWidth;
	int sequenceAStrideHeight;
	int sequenceBStrideWidth;
	int sequenceBStrideHeight;

	int sequenceAPaddingWidth;
	int sequenceAPaddingHeight;
	int sequenceBPaddingWidth;
	int sequenceBPaddingHeight;

	bool sort;
	int retain;
};

struct LibMatchMexErrorWithMessage parseParameter(struct LibBlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

struct LibMatchMexErrorWithMessage validateParameter(struct LibBlockMatchMexContext *context);

bool generate_result(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y, const float *value, const int size);
