#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_case_5)
{
	void *instance;
	float inputMatrix[25 * 20] = { 
		9,7,5,8,4,2,2,9,8,6,7,4,1,1,1,1,2,10,1,3,5,8,3,2,7,
		10,1,4,3,9,8,10,7,4,3,4,10,3,9,8,7,4,10,6,3,1,5,3,8,7,
		2,9,8,6,6,4,1,4,3,8,9,5,9,10,6,1,9,1,9,7,6,1,7,5,7,
		10,10,8,7,6,6,8,6,5,2,6,2,1,8,5,1,9,8,7,3,5,3,9,2,10,
		7,7,2,9,10,2,9,5,1,7,4,10,10,1,10,6,1,3,2,9,7,2,4,4,3,
		1,8,5,10,3,7,9,1,2,2,10,10,8,3,7,1,4,5,4,10,7,3,8,7,8,
		3,8,5,6,8,3,1,3,10,4,9,5,5,4,7,9,6,6,5,8,7,5,7,2,3,
		6,4,7,2,8,7,4,2,10,7,6,2,6,7,9,9,5,10,10,4,1,6,1,8,2,
		10,7,8,2,4,7,3,2,6,8,7,3,3,2,9,8,7,5,2,6,1,5,7,3,7,
		10,2,8,3,6,8,9,3,1,1,6,5,5,8,6,2,7,10,9,2,4,9,4,10,5,
		2,8,3,9,1,5,5,5,3,10,3,6,10,2,2,7,3,4,7,10,6,6,10,3,5,
		10,1,7,3,1,1,10,1,4,8,4,3,6,7,3,6,5,8,4,9,7,10,1,8,7,
		10,3,7,9,6,3,2,10,9,5,5,7,6,5,9,10,1,7,2,9,5,7,5,2,8,
		5,1,2,3,8,10,3,10,1,5,3,8,3,8,1,7,10,6,5,3,9,10,5,3,4,
		9,1,2,10,10,2,2,5,1,5,9,3,5,8,5,9,2,7,5,6,8,3,5,1,7,
		2,9,5,4,2,9,2,5,2,4,2,2,7,10,2,5,2,7,2,1,10,7,8,6,5,
		5,7,10,2,6,6,9,4,7,6,3,3,7,9,10,5,4,2,6,5,6,3,4,7,9,
		10,4,4,3,5,10,6,10,8,6,2,4,4,4,8,9,2,2,3,4,4,7,8,6,9,
		8,10,6,7,1,1,6,4,7,9,3,5,4,7,6,1,5,10,4,2,2,7,5,5,3,
		10,1,3,5,4,5,2,2,5,8,5,6,10,2,5,2,4,2,6,2,7,1,1,7,7
	};

	int matM = 25, matN = 20, blockM = 2, blockN = 2, strideM = 3, strideN = 1, searchRegionM = 0, searchRegionN = 0, numberOfResultRetain = 10,
		matrixPaddingMPre = 0, matrixPaddingMPost = 0, matrixPaddingNPre = 0, matrixPaddingNPost = 0;

	int matrixC_M, matrixC_N, matrixC_O,
		matrixA_padded_M, matrixA_padded_N, matrixB_padded_M, matrixB_padded_N;
	BOOST_TEST(blockMatchInitialize<float>(&instance, SearchType::global, LibMatchMeasureMethod::mse, PadMethod::symmetric, PadMethod::symmetric,BorderType::normal, SearchFrom::topLeft, false,
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
	blockMatchFinalize<float>(instance);
	free(matrixC);
	free(matrixAPadded);
	free(matrixBPadded);
	free(indexX);
	free(indexY);
}