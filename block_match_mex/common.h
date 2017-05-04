#pragma once

#include <lib_match_mex_common.h>
#include <typeindex>

struct BlockMatchMexContext
{
	MeasureMethod method;
	SearchType searchType;
	PadMethod padMethodA;
	PadMethod padMethodB;
	BorderType sequenceABorderType;
	SearchFrom searchFrom;

	std::type_index sourceAType = typeid(nullptr);
	std::type_index sourceBType = typeid(nullptr);
	std::type_index intermediateType = typeid(nullptr);
	std::type_index resultType = typeid(nullptr);
	std::type_index indexDataType = typeid(nullptr);

	int sequenceMatrixNumberOfDimensions;

	int sequenceAMatrixDimensions[4];
	void *sequenceAMatrixPointer;

	int sequenceBMatrixDimensions[4];
	void *sequenceBMatrixPointer;

	int block_M;
	int block_N;
	int searchRegion_M;
	int searchRegion_N;

	int sequenceAStride_M;
	int sequenceAStride_N;
	int sequenceBStride_M;
	int sequenceBStride_N;

	int sequenceAPadding_M_Pre;
	int sequenceAPadding_M_Post;
	int sequenceAPadding_N_Pre;
	int sequenceAPadding_N_Post;
	int sequenceBPadding_M_Pre;
	int sequenceBPadding_M_Post;
	int sequenceBPadding_N_Pre;
	int sequenceBPadding_N_Post;

	bool sort;
	int retain;
	bool threshold;
	double thresholdValue;
	int sparse;
};

struct LibMatchMexErrorWithMessage parseParameter(struct BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);
