#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_case_10)
{
	MemoryMappedIO memoryMappedIO(L"test_case_10.bin");
	const void *fileptr = memoryMappedIO.getPtr();
	const int outDim0 = 7, outDim1 = 7, outDim2 = 49;
	const int outSize = outDim0 * outDim1 * outDim2;
	double groundtruth[outSize];
	memcpy(groundtruth, fileptr, outSize * sizeof(double));
	double out[outSize];
	MemoryMappedIO inputFile(L"test_case_10_in.bin");
	const void *inputFilePtr = inputFile.getPtr();
	const int inDim0 = 16, inDim1 = 16;
	const int blockSize = 4;
	const int stepSize = 2;

	uint8_t input[inDim0 * inDim1];
	memcpy(input, inputFilePtr, inDim0 * inDim1 * sizeof(uint8_t));

	BlockMatch<double> blockMatch(typeid(uint8_t), typeid(uint8_t), typeid(double), typeid(uint8_t),
		SearchType::global, MeasureMethod::mse, PadMethod::zero, PadMethod::zero,
		BorderType::normal, false,
		inDim0, inDim1, inDim0, inDim1, 1,
		0, 0, 0, 0, blockSize, blockSize,
		stepSize, stepSize, stepSize, stepSize,
		1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, true);
	blockMatch.initialize();
	int estimateOutDim0, estimateOutDim1, estimateOutDim2;
	blockMatch.get_matrixC_dimensions(&estimateOutDim0, &estimateOutDim1, &estimateOutDim2);
	BOOST_CHECK_EQUAL(outDim0, estimateOutDim0);
	BOOST_CHECK_EQUAL(outDim1, estimateOutDim1);
	BOOST_CHECK_EQUAL(outDim2, estimateOutDim2);

	uint8_t index_m[outSize];
	uint8_t index_n[outSize];
	MemoryMappedIO indexFile(L"test_case_10_ind.bin");
	const void *indexFilePtr = indexFile.getPtr();

	blockMatch.execute(input, input, out, nullptr, nullptr, index_m, index_n);

	checkFloatPointEqual(out, groundtruth, outSize);
	uint8_t groundtruth_ind[outSize * 2];
	memcpy(groundtruth_ind, indexFilePtr, outSize * 2);
	checkIndexEqual(index_n, index_m, groundtruth_ind, outDim2, outDim0 * outDim1);
}
