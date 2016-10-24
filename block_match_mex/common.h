#pragma once

#include <lib_match_mex_common.h>
#include <typeindex>

struct BlockMatchMexContext
{
	LibMatchMeasureMethod method;
	SearchType searchType;
	PadMethod padMethodA;
	PadMethod padMethodB;
	
	std::type_index sourceType = typeid(nullptr);
	std::type_index intermediateType = typeid(nullptr);
	std::type_index resultType = typeid(nullptr);

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

struct LibMatchMexErrorWithMessage parseParameter(struct BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

struct LibMatchMexErrorWithMessage validateParameter(struct BlockMatchMexContext *context);

template <typename IntermidateType, typename ResultType>
bool generate_result(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y,
	const IntermidateType *value, const int size);

template <typename ResultType>
bool generatePaddedMatrix(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const ResultType *data);