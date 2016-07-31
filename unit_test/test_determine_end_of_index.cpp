#include "test_common.h"

bool validateEndOfIndex(int matSize, int padSize, int blockSize, int endIndex)
{
	int i = 0;
	for (;i<padSize+matSize;++i)
	{
		if (i+blockSize>=padSize+matSize)
			break;

		if (i >= endIndex)
			return false;
	}

	if (i + 1 != endIndex)
		return false;

	return true;
}

BOOST_AUTO_TEST_CASE(test_determine_end_of_index)
{
	int matSize, padSize, blockSize;
	int v1;

	matSize = 3, padSize = 6, blockSize = 3;
	v1 = determineEndOfIndex(matSize, padSize, blockSize);
	BOOST_CHECK(validateEndOfIndex(matSize, padSize, blockSize, v1));
	matSize = 5, padSize = 2, blockSize = 5;
	v1 = determineEndOfIndex(matSize, padSize, blockSize);
	BOOST_CHECK(validateEndOfIndex(matSize, padSize, blockSize, v1));
	matSize = 20, padSize = 0, blockSize = 2;
	v1 = determineEndOfIndex(matSize, padSize, blockSize);
	BOOST_CHECK(validateEndOfIndex(matSize, padSize, blockSize, v1));
}