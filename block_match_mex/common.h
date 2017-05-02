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
};

struct LibMatchMexErrorWithMessage parseParameter(struct BlockMatchMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);

struct LibMatchMexErrorWithMessage validateParameter(struct BlockMatchMexContext *context);

template <typename IntermidateType, typename ResultType>
void generate_result(mxArray **pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y,
	const IntermidateType *value, const int size);

template <typename IntermidateType, typename ResultType>
void generatePaddedMatrix(mxArray **pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const IntermidateType *data);