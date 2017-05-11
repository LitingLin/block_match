#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_copy_block) {
	float A[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	const size_t size = sizeof(A) / sizeof(*A);
	double B[size];
	copyBlock(B, A, 4, 3, 1, 1, 2, 2);
	BOOST_CHECK_EQUAL(B[0], 5);
	BOOST_CHECK_EQUAL(B[1], 6);
	BOOST_CHECK_EQUAL(B[2], 8);
	BOOST_CHECK_EQUAL(B[3], 9);
}