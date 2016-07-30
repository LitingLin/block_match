#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_mse_cpu)
{
	float a[] = { 3,2,4,1,5 };
	float b[] = { 4,2,6,9,4 };
	float result;
	block_match_mse_cpu(a, b, 1, 1, sizeof(a) / sizeof(*a), &result);
	BOOST_CHECK(result == 14);
}