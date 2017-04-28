#pragma once

#include <lib_match_mex_common.h>

struct ArrayMatchMexContext
{
	MeasureMethod method;
	int numberOfArrayA; 
	int numberOfArrayB;
	int lengthOfArray;

	double *A;
	double *B;

	bool debug;
};

struct LibMatchMexErrorWithMessage parseParameter(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

size_t getMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArrayA, int numberOfArrayB);