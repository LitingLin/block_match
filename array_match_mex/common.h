#pragma once

#include <lib_match_mex_common.h>

struct ArrayMatchMexContext
{
	MeasureMethod method;
	int numberOfArrayA; 
	int numberOfArrayB;
	int lengthOfArray;

	void *A;
	void *B;

	std::type_index sourceAType = typeid(nullptr);
	std::type_index sourceBType = typeid(nullptr);
	std::type_index intermediateType = typeid(nullptr);
	std::type_index resultType = typeid(nullptr);
	std::type_index indexDataType = typeid(nullptr);

	bool sort;
	int retain;

	bool threshold;
	double thresholdValue;
	double thresholdReplacementValue;

	unsigned numberOfThreads;
	int indexOfDevice;
};

struct LibMatchMexErrorWithMessage parseParameter(struct ArrayMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

size_t getMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArrayA, int numberOfArrayB);