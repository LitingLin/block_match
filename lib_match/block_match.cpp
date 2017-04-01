
#include "lib_match.h"

template <typename T>
BlockMatch<T>::BlockMatch()
	: instance(nullptr)
{
}

template <typename T>
BlockMatch<T>::~BlockMatch()
{
	if (instance)
		release();
}

template <typename T>
void BlockMatch<T>::initialize(
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	SearchFrom searchFrom,
	bool sort,
	int matrixA_M, int matrixA_N, int matrixB_M, int matrixB_N,
	int searchRegion_M, int searchRegion_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int matrixAPadding_M_pre, int matrixAPadding_M_post,
	int matrixAPadding_N_pre, int matrixAPadding_N_post,
	int matrixBPadding_M_pre, int matrixBPadding_M_post,
	int matrixBPadding_N_pre, int matrixBPadding_N_post,
	int numberOfIndexRetain)
{
	instance = blockMatchInitialize<T>(
		searchType,
		measureMethod,
		padMethodA, padMethodB,
		sequenceABorderType,
		searchFrom,
		sort,
		matrixA_M, matrixA_N, matrixB_M, matrixB_N,
		searchRegion_M, searchRegion_N,
		block_M, block_N,
		strideA_M, strideA_N,
		strideB_M, strideB_N,
		matrixAPadding_M_pre, matrixAPadding_M_post,
		matrixAPadding_N_pre, matrixAPadding_N_post,
		matrixBPadding_M_pre, matrixBPadding_M_post,
		matrixBPadding_N_pre, matrixBPadding_N_post,
		numberOfIndexRetain);
}
