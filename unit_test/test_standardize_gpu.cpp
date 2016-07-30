#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_standardize_gpu)
{
	float a[] = { 1.7339,3.9094,8.3138,8.0336,0.6047,3.9926,5.2688,
		4.1680,6.5686,6.2797,2.9198,4.3165,0.1549,9.8406,1.6717,1.0622,3.7241,1.9812,4.8969,3.3949 };
	float c[]={-0.9063,-0.0875,1.5703,1.4648,-1.3313,-0.0562,0.4242,0.0099,0.9134,0.8047,-0.4599,0.0658,-1.5006,
		2.1449,-0.9297,-1.1591,-0.1572,-0.8132,0.2842,-0.2811};
	float *dev_a;
	const int a_length = sizeof(a) / sizeof(*a);
	const int c_length = sizeof(c) / sizeof(*c);
	static_assert(a_length == c_length, "mismatch");

	cudaError_t cuda_error = cudaMalloc(&dev_a, sizeof(a));
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	cuda_error = cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	cuda_error = standardize(dev_a, 1, a_length, 1,cudaStreamDefault);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	cuda_error = cudaMemcpy(a, dev_a, sizeof(a), cudaMemcpyDeviceToHost);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);

	isNormal(a, a_length);

	for (int i=0;i<a_length;++i)
	{
		BOOST_CHECK_SMALL(a[i] - c[i], 0.0001f);
	}
	cuda_error = cudaFree(dev_a);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
}