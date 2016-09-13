#pragma once

#include <lib_match_mex_common.h>

struct ArrayMatchMexContext
{
	LibMatchMeasureMethod method;
	int numberOfArray;
	int lengthOfArray;

	double *A;
	double *B;
};

struct LibMatchMexErrorWithMessage parseParameter(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

size_t getMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArray);