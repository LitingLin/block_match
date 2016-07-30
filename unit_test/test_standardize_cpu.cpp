#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_standardize_cpu)
{
	float a[] = { 3,4,6,2,4,7,1,5,345,4,6,8 };
	int a_length = sizeof(a) / sizeof(*a);
	standardize_cpu(a, a_length);
	float mean = 0;
	for (int i=0;i<a_length;++i)
	{
		mean += a[i];
	}
	mean /= a_length;
	BOOST_CHECK(mean < 0.000001 && mean > -0.000001);

	float sd = 0;

	for (int i=0;i<a_length;++i)
	{
		sd += a[i]*a[i];
	}
	sd /= a_length;
	sd=sqrt(sd);
	BOOST_CHECK(sd > 0.999999&&sd < 1.0000001);
}