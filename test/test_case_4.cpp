#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_case_4)
{
	void *instance;
	float inputMatrix[10 * 10] = {
		30,49,68,34,89,62,98,82,43,53,
		32,58,40,68,33,86,71,82,83,42,
		42,24,37,14,70,81,50,72, 8,66,
		51,46,99,72,20,58,47,15,13,63,
		9,96, 4,11, 3,18, 6,66,17,29,
		26,55,89,65,74,24,68,52,39,43,
		80,52,91,49,50,89, 4,97,83, 2,
		3,23,80,78,48, 3, 7,65,80,98,
		93,49,10,72,90,49,52,80, 6,17,
		73,62,26,90,61,17,10,45,40,11 };

	int matM = 10, matN = 10, blockM = 2, blockN = 2, strideM = 2, strideN = 2, searchRegionM = 0, searchRegionN = 0, numberOfResultRetain = 0,
		matrixPaddingMPre = 2, matrixPaddingMPost = 2, matrixPaddingNPre = 2, matrixPaddingNPost = 2;

	int matrixC_M, matrixC_N, matrixC_O,
		matrixA_padded_M, matrixA_padded_N, matrixB_padded_M, matrixB_padded_N;
	BOOST_TEST(blockMatchAndSortingInitialize(&instance, SearchType::global, LibMatchMeasureMethod::mse, PadMethod::symmetric,
		matM, matN, matM, matN, searchRegionM, searchRegionN, blockM, blockN, strideM, strideN, strideM, strideN,
		matrixPaddingMPre, matrixPaddingMPost, matrixPaddingNPre, matrixPaddingNPost, matrixPaddingMPre, matrixPaddingMPost, matrixPaddingNPre, matrixPaddingNPost,
		numberOfResultRetain,
		&matrixC_M, &matrixC_N, &matrixC_O,
		&matrixA_padded_M, &matrixA_padded_N,
		&matrixB_padded_M, &matrixB_padded_N), getLastErrorString());

	float *matrixC = (float*)malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(float));
	float *matrixAPadded = (float*)malloc(matrixA_padded_M * matrixA_padded_N * sizeof(float));
	float *matrixBPadded = (float*)malloc(matrixB_padded_M * matrixB_padded_N * sizeof(float));
	int *indexX = (int*)malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(int));
	int *indexY = (int*)malloc(matrixC_M * matrixC_N * matrixC_O * sizeof(int));

	BOOST_TEST(blockMatchExecute(instance, inputMatrix, inputMatrix, matrixC, matrixAPadded, matrixBPadded, indexX, indexY), getLastErrorString());
	blockMatchFinalize(instance);
	free(matrixC);
	free(matrixAPadded);
	free(matrixBPadded);
	free(indexX);
	free(indexY);
}
