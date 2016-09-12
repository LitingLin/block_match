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

BOOST_AUTO_TEST_CASE(PaddingZero)
{
	int mat_M = 3,
		mat_N = 3,
		patch_M = 2,
		patch_N = 2;
	float *matrix = new float[mat_M * mat_N];
	fillWithSequence(matrix, 9); 
	int M_left, M_right, N_left, N_right;
	
	const float ground_truth1[] = {
		0, 1, 2, 0,
		3, 4, 5, 0,
		6, 7, 8, 0,
		9, 10, 11, 0
	};
	
	determinePadSizeAccordingToPatchSize(mat_M, mat_N, patch_M, patch_N, &M_left, &M_right, &N_left, &N_right);
	BOOST_CHECK_EQUAL(M_left, )
	ASSERT_EQ(matrix.width(), 4);
	ASSERT_EQ(matrix.height(), 4);
	for (size_t i = 0; i != 4 * 4; i++)
		ASSERT_EQ(p_const_data[i], ground_truth1[i]);

	matrix.resize(2, 4, 5);
	fillDataWithI << <getNumBlocks(2 * 4 * 5), numBlockThreads >> >(matrix.mutable_gpu_data(), 2 * 4 * 5);
	matrix.paddingAccordingToPatchSize();
	p_const_data = matrix.cpu_data();
	const float ground_truth2[] = {
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		0, 0, 0, 0,

		20, 21, 22, 23,
		24, 25, 26, 27,
		28, 29, 30, 31,
		32, 33, 34, 35,
		36, 37, 38, 39,
		0, 0, 0, 0
	};
	ASSERT_EQ(matrix.width(), 4);
	ASSERT_EQ(matrix.height(), 6);
	for (size_t i = 0; i != 2 * 4 * 6; i++)
		ASSERT_EQ(p_const_data[i], ground_truth2[i]);
}

BOOST_AUTO_TEST_CASE(PaddingCircular)
{
	Matrix3D<float> matrix(1, 3, 4);
	fillDataWithI << <getNumBlocks(3 * 4), numBlockThreads >> >(matrix.mutable_gpu_data(), 3 * 4);
	matrix.setPatchSize(2, 2);
	matrix.paddingAccordingToPatchSize(Matrix3D<float>::PaddingMethod::Circular);
	const float *p_const_data = matrix.cpu_data();
	const float ground_truth[] = {
		0, 1, 2, 0,
		3, 4, 5, 3,
		6, 7, 8, 6,
		9, 10, 11, 9
	};
	ASSERT_EQ(matrix.width(), 4);
	ASSERT_EQ(matrix.height(), 4);
	for (size_t i = 0; i != 4 * 4; i++)
		ASSERT_EQ(p_const_data[i], ground_truth[i]);
}

BOOST_AUTO_TEST_CASE(PaddingSymmetric)
{
	Matrix3D<float> matrix(1, 3, 4);
	fillDataWithI << <getNumBlocks(3 * 4), numBlockThreads >> >(matrix.mutable_gpu_data(), 3 * 4);
	matrix.setPatchSize(2, 2);
	matrix.paddingAccordingToPatchSize(Matrix3D<float>::PaddingMethod::Symmetric);
	const float *p_const_data = matrix.cpu_data();
	const float ground_truth1[] = {
		0, 1, 2, 2,
		3, 4, 5, 5,
		6, 7, 8, 8,
		9, 10, 11, 11
	};
	ASSERT_EQ(matrix.width(), 4);
	ASSERT_EQ(matrix.height(), 4);
	for (size_t i = 0; i != 4 * 4; i++)
		ASSERT_EQ(p_const_data[i], ground_truth1[i]);

	matrix.resize(2, 4, 5);
	fillDataWithI << <getNumBlocks(2 * 4 * 5), numBlockThreads >> >(matrix.mutable_gpu_data(), 2 * 4 * 5);
	matrix.setPatchSize(3, 3);
	matrix.paddingAccordingToPatchSize(Matrix3D<float>::PaddingMethod::Symmetric);
	const float ground_truth2[] = {
		0,0,1,2,3,3,
		4,4,5,6,7,7,
		8,8,9,10,11,11,
		12,12,13,14,15,15,
		16,16,17,18,19,19,
		16,16,17,18,19,19,

		20,20,21,22,23,23,
		24,24,25,26,27,27,
		28,28,29,30,31,31,
		32,32,33,34,35,35,
		36,36,37,38,39,39,
		36,36,37,38,39,39
	};
	p_const_data = matrix.cpu_data();
	ASSERT_EQ(matrix.width(), 6);
	ASSERT_EQ(matrix.width(), 6);
	for (size_t i = 0; i != 2 * 6 * 6; i++)
		ASSERT_EQ(p_const_data[i], ground_truth2[i]);
}

BOOST_AUTO_TEST_CASE(PaddingReplicate)
{
	Matrix3D<float> matrix(1, 3, 4);
	fillDataWithI << <getNumBlocks(3 * 4), numBlockThreads >> >(matrix.mutable_gpu_data(), 3 * 4);
	matrix.setPatchSize(2, 2);
	matrix.paddingAccordingToPatchSize(Matrix3D<float>::PaddingMethod::Replicate);
	const float *p_const_data = matrix.cpu_data();
	const float ground_truth[] = {
		0, 1, 2, 2,
		3, 4, 5, 5,
		6, 7, 8, 8,
		9, 10, 11, 11
	};
	ASSERT_EQ(matrix.width(), 4);
	ASSERT_EQ(matrix.height(), 4);
	for (size_t i = 0; i != 4 * 4; i++)
		ASSERT_EQ(p_const_data[i], ground_truth[i]);

	matrix.resize(1, 3, 2);
	const float data2[] = {
		1, 2, 3,
		4, 5, 6
	};
	memcpy(matrix.mutable_cpu_data(), data2, 3 * 2 * sizeof(float));
	matrix.padding(2, 4, Matrix3D<float>::PaddingMethod::Replicate);
	const float ground_truth2[] = {
		1,   1,   2,   3,   3,
		1,   1,   2,   3,   3,
		1,   1,   2,   3,   3,
		4,   4,   5,   6,   6,
		4,   4,   5,   6,   6,
		4,   4,   5,   6,   6
	};
	ASSERT_EQ(matrix.width(), 5);
	ASSERT_EQ(matrix.height(), 6);

	p_const_data = matrix.cpu_data();
	for (size_t i = 0; i != 5 * 6; i++)
		ASSERT_EQ(p_const_data[i], ground_truth2[i]);
}