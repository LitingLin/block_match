#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_standardize_cpu_float)
{
	float a[] = { 1.7339,3.9094,8.3138,8.0336,0.6047,3.9926,5.2688,
		4.1680,6.5686,6.2797,2.9198,4.3165,0.1549,9.8406,1.6717,1.0622,3.7241,1.9812,4.8969,3.3949 };
	float c[] = { -0.9063,-0.0875,1.5703,1.4648,-1.3313,-0.0562,0.4242,0.0099,0.9134,0.8047,-0.4599,0.0658,-1.5006,
		2.1449,-0.9297,-1.1591,-0.1572,-0.8132,0.2842,-0.2811 };
	int a_length = sizeof(a) / sizeof(*a);
	standardize_cpu(a, a_length);
	isNormal(a, a_length);
	for (int i = 0; i<a_length; ++i)
	{
		BOOST_CHECK_SMALL(a[i] - c[i], 0.0001f);
	}
}

BOOST_AUTO_TEST_CASE(test_standardize_cpu_double)
{
	double a[] = { 1.7339,3.9094,8.3138,8.0336,0.6047,3.9926,5.2688,
		4.1680,6.5686,6.2797,2.9198,4.3165,0.1549,9.8406,1.6717,1.0622,3.7241,1.9812,4.8969,3.3949 };
	double c[] = { -0.9063,-0.0875,1.5703,1.4648,-1.3313,-0.0562,0.4242,0.0099,0.9134,0.8047,-0.4599,0.0658,-1.5006,
		2.1449,-0.9297,-1.1591,-0.1572,-0.8132,0.2842,-0.2811 };
	int a_length = sizeof(a) / sizeof(*a);
	standardize_cpu(a, a_length);
	isNormal(a, a_length);
	for (int i = 0; i<a_length; ++i)
	{
		BOOST_CHECK_SMALL(a[i] - c[i], 0.0001);
	}
}