#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_cc_gpu)
{
	float a[] = { 1.7339,3.9094,8.3138,8.0336,0.6047,3.9926,5.2688,
		4.1680,6.5686,6.2797,2.9198,4.3165,0.1549,9.8406,1.6717,1.0622,3.7241,1.9812,4.8969,3.3949 };
	float b[] = { 9.5163,9.2033,0.5268,7.3786,2.6912,4.2284,5.4787,9.4274,4.1774,9.8305,3.0145,7.0110,6.6634,
		5.3913,6.9811,6.6653,1.7813,1.2801,9.9908,1.7112 };
	float *dev_a, *dev_b, *dev_c;

	int a_length = sizeof(a) / sizeof(*a);
	int b_length = sizeof(b) / sizeof(*b);
	BOOST_CHECK_EQUAL(a_length, b_length);

	cudaError_t cuda_error = cudaMalloc(&dev_a, sizeof(a) * 2 + 1 * sizeof(*a));
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	dev_b = dev_a + a_length;
	dev_c = dev_b + b_length;
	cuda_error = cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	cuda_error = cudaMemcpy(dev_b, b, sizeof(b), cudaMemcpyHostToDevice);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	cuda_error = block_match_cc(dev_a, dev_b, 1, 1, 1, a_length, dev_c, 1, 1, cudaStreamDefault);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	float c;
	cuda_error = cudaMemcpy(&c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	BOOST_CHECK(c > -1.8452e-04 && c<1.8454e-04); // c ~= 1.8453e-04
	cuda_error = cudaFree(dev_a);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
}