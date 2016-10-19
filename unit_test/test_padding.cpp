#include "test_common.h"

bool validate_pad_result(float *to_validate, float *truth_array, int src_m, int src_n, int padding_m, int padding_n, int index_m, int index_n, int block_m, int block_n)
{
	int truth_index_m = index_m + padding_m;
	int truth_index_n = index_n + padding_n;
	for (int i = 0; i < block_m; i++)
		for (int j = 0; j < block_n; j++)
			if (to_validate[i*block_n + j] != truth_array[(truth_index_m + i)*src_n + truth_index_n + j])
				return false;
	return true;
}

BOOST_AUTO_TEST_CASE(test_padding)
{
	float to_pad[] = {
		1,2,3,
		4,5,6,
		7,8,9 };
	float padded[] = {
		9,8,7,7,8,9,9,8,7,
		6,5,4,4,5,6,6,5,4,
		3,2,1,1,2,3,3,2,1,
		3,2,1,1,2,3,3,2,1,
		6,5,4,4,5,6,6,5,4,
		9,8,7,7,8,9,9,8,7,
		9,8,7,7,8,9,9,8,7,
		6,5,4,4,5,6,6,5,4,
		3,2,1,1,2,3,3,2,1 };

	float buf[32];

	int mat_m = 3, mat_n = 3;
	int test_index_m = 1, test_index_n = 1, test_block_m = 2, test_block_n = 2;

	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 9, 9, 3, 3, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = -2;
	test_index_n = -1;
	test_block_m = 1;
	test_block_n = 3;

	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 9, 9, 3, 3, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = 5;
	test_index_n = 4;
	test_block_m = 1;
	test_block_n = 2;
	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 9, 9, 3, 3, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = 2;
	test_index_n = 0;
	test_block_m = 3;
	test_block_n = 1;
	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 9, 9, 3, 3, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = 1;
	test_index_n = -2;
	test_block_m = 2;
	test_block_n = 1;
	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 9, 9, 3, 3, test_index_m, test_index_n, test_block_m, test_block_n));

	float to_pad_2[] = {
		1,2,
		3,4,
		5,6
	};
	float padded_2[] = {
		4,3,3,4,4,3,
		2,1,1,2,2,1,
		2,1,1,2,2,1,
		4,3,3,4,4,3,
		6,5,5,6,6,5,
		6,5,5,6,6,5,
		4,3,3,4,4,3
	};

	memcpy(to_pad, to_pad_2, 2 * 3 * sizeof(float));
	memcpy(padded, padded_2, 6 * 7 * sizeof(float));

	mat_m = 3, mat_n = 2;
	test_index_m = 1, test_index_n = 1, test_block_m = 2, test_block_n = 2;

	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 7, 6, 2, 2, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = -2;
	test_index_n = -1;
	test_block_m = 1;
	test_block_n = 3;

	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 7, 6, 2, 2, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = -1;
	test_index_n = 0;
	test_block_m = 3;
	test_block_n = 1;
	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 7, 6, 2, 2, test_index_m, test_index_n, test_block_m, test_block_n));
	test_index_m = 1;
	test_index_n = -2;
	test_block_m = 2;
	test_block_n = 1;
	copyBlockWithSymmetricPadding(buf, to_pad, mat_m, mat_n, test_index_m, test_index_n, test_block_m, test_block_n);

	BOOST_CHECK(validate_pad_result(buf, padded, 7, 6, 2, 2, test_index_m, test_index_n, test_block_m, test_block_n));
}

void fillWithSequence(float *ptr, size_t size)
{
	for (size_t i=0;i<size;++i)
	{
		ptr[i] = i;
	}
}

void fillWithSequence(int *ptr, size_t size)
{
	for (size_t i = 0; i<size; ++i)
	{
		ptr[i] = i;
	}
}