#include "test_common.h"

BOOST_AUTO_TEST_CASE(array_match_test_case_2) {
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
	for (int i = 5; i<numberOfArrayA; ++i)
	{
		memcpy(A + size*i, T5, sizeof(int) * size);
	}
	double B[] = {
		2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 5, 6, 7, 8,
		9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7,
		8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2,
		3, 4, 5
	};
	static_assert(numberOfArrayB*size * sizeof(double) == sizeof(B), "");
	ArrayMatch<double> match(typeid(int), typeid(double), typeid(float), typeid(nullptr),
		MeasureMethod::mse, false, numberOfArrayA, numberOfArrayB, size, 0);
	match.initialize();
	float *C = (float*)malloc(sizeof(float) * numberOfArrayA * numberOfArrayB);

	match.execute(A, B, C);

	free(A);

	BOOST_CHECK_EQUAL(C[0], 1.f);
	BOOST_CHECK_EQUAL(C[1], 4.f);
	BOOST_CHECK_EQUAL(C[numberOfArrayB], 0.f);
	BOOST_CHECK_EQUAL(C[numberOfArrayA * numberOfArrayB - 1], 16.f);

	free(C);
}