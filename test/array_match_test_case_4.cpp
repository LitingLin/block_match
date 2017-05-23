#include "test_common.h"

#include <algorithm>

BOOST_AUTO_TEST_CASE(array_match_test_case_4) {
	const int numberOfArrayA = 2048, numberOfArrayB = 15;
	const int size = 5;
	int *A = (int*)malloc(sizeof(int) * numberOfArrayA * size);
	int T1[] = { 1,2,3,4,5 };
	int T2[] = { 2,3,4,5,6 };
	int T3[] = { 3,4,5,6,7 };
	int T4[] = { 4,5,6,7,8 };
	int T5[] = { 5,6,7,8,9 };
	memcpy(A, T1, sizeof(int) * size);
	memcpy(A + size, T2, sizeof(int) * size);
	memcpy(A + size * 2, T3, sizeof(int) * size);
	memcpy(A + size * 3, T4, sizeof(int) * size);
	memcpy(A + size * 4, T5, sizeof(int) * size);
	for (int i = 5; i < numberOfArrayA; ++i)
	{
		memcpy(A + size*i, T5, sizeof(int) * size);
	}
	double B[] = {
		2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 5, 6, 7, 8,
		9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7,
		8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2,
		3, 4, 5,
	};
	static_assert(numberOfArrayB*size * sizeof(double) == sizeof(B), "");
	int retain = 10;
	ArrayMatch<double> match(typeid(int), typeid(double), typeid(float), typeid(uint8_t),
		MeasureMethod::mse, true, numberOfArrayA, numberOfArrayB, size, retain,
		false, 0, 0, true);
	match.initialize();
	float *C = (float*)malloc(sizeof(float) * numberOfArrayA * numberOfArrayB);
	uint8_t *index = (uint8_t*)malloc(sizeof(uint8_t) * numberOfArrayA * numberOfArrayB);
	match.execute(A, B, C, index);

	free(A);

	BOOST_CHECK_EQUAL(C[0], 0.f);
	BOOST_CHECK_EQUAL(index[0], 13 + 1);
	BOOST_CHECK_EQUAL(C[1], 0.f);
	BOOST_CHECK_EQUAL(index[1], 14 + 1);
	BOOST_CHECK_EQUAL(C[2], 1.f);
	BOOST_CHECK_EQUAL(index[2], 0 + 1);
	BOOST_CHECK_EQUAL(C[3], 4.f);
	BOOST_CHECK_EQUAL(index[3], 1 + 1);
	BOOST_CHECK_EQUAL(C[4], 9.f);
	BOOST_CHECK_EQUAL(index[4], 2 + 1);
	BOOST_CHECK_EQUAL(C[5], 16.f);
	BOOST_CHECK(inRange(index[5], uint8_t(3 + 1), uint8_t(14 + 1)));
	BOOST_CHECK_EQUAL(C[retain], 0.f);
	BOOST_CHECK_EQUAL(index[retain], 0 + 1);
	BOOST_CHECK_EQUAL(C[retain + 1], 1.f);
	int a[3] = { 1 + 1,13 + 1,14 + 1 };
	BOOST_CHECK(inRange(index[retain + 1], a));

	free(C);
}