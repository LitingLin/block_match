#include "test_common.h"

int getLengthByIteration(int matSize, int padSize, int blockSize, int strideSize)
{
	int count = 0;
	for (int i = -padSize; i < matSize + padSize; i += strideSize)
	{
		if (i + blockSize - 1 >= matSize + padSize)
			break;
		++count;
	}
	return count;
}

BOOST_AUTO_TEST_CASE(test_get_length)
{
	int matSize, padSize, blockSize, strideSize;
	int v1, v2;

	matSize = 3; padSize = 1; blockSize = 1; strideSize = 2;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
	matSize = 6; padSize = 2; blockSize = 4; strideSize = 2;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
	matSize = 9; padSize = 51; blockSize = 6; strideSize = 8;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
	matSize = 4; padSize = 0; blockSize = 2; strideSize = 4;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
	matSize = 256; padSize = 0; blockSize = 8; strideSize = 3;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
	matSize = 25; padSize = 0; blockSize = 2; strideSize = 3;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
	matSize = 20, padSize = 0, blockSize = 2, strideSize = 1;
	v1 = getLength(matSize, padSize, blockSize, strideSize);
	v2 = getLengthByIteration(matSize, padSize, blockSize, strideSize);
	BOOST_CHECK_EQUAL(v1, v2);
}