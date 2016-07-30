#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_standardize_cpu)
{
	float a[] = { 3,4,6,2,4,7,1,5,345,4,6,8 };
	int a_length = sizeof(a) / sizeof(*a);
	standardize_cpu(a, a_length);
	isNormal(a, a_length);
}