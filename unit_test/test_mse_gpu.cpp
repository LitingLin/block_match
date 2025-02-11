#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_mse_gpu_float)
{
	float a[] = { 1.7339f,3.9094f,8.3138f,8.0336f,0.6047f,3.9926f,5.2688f,
		4.1680f,6.5686f,6.2797f,2.9198f,4.3165f,0.1549f,9.8406f,1.6717f,1.0622f,3.7241f,1.9812f,4.8969f,3.3949f };
	float b[] = { 9.5163f,9.2033f,0.5268f,7.3786f,2.6912f,4.2284f,5.4787f,9.4274f,4.1774f,9.8305f,3.0145f,7.0110f,6.6634f,
		5.3913f,6.9811f,6.6653f,1.7813f,1.2801f,9.9908f,1.7112f };
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
	cuda_error = lib_match_mse(dev_a, dev_b, 1, 1, a_length, dev_c, 1, 1, cudaStreamDefault);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	float c;
	cuda_error = cudaMemcpy(&c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	BOOST_CHECK_SMALL(c - 18.1079f, singleFloatingPointErrorTolerance);
	cuda_error = cudaFree(dev_a);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
}

BOOST_AUTO_TEST_CASE(test_mse_gpu_double)
{
	double a[] = { 1.7339,3.9094,8.3138,8.0336,0.6047,3.9926,5.2688,
		4.1680,6.5686,6.2797,2.9198,4.3165,0.1549,9.8406,1.6717,1.0622,3.7241,1.9812,4.8969,3.3949 };
	double b[] = { 9.5163,9.2033,0.5268,7.3786,2.6912,4.2284,5.4787,9.4274,4.1774,9.8305,3.0145,7.0110,6.6634,
		5.3913,6.9811,6.6653,1.7813,1.2801,9.9908,1.7112 };
	double *dev_a, *dev_b, *dev_c;

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
	cuda_error = lib_match_mse(dev_a, dev_b, 1, 1, a_length, dev_c, 1, 1, cudaStreamDefault);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	double c;
	cuda_error = cudaMemcpy(&c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	BOOST_CHECK_SMALL(c - 18.1079, doubleFloatingPointErrorTolerance);
	cuda_error = cudaFree(dev_a);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
}