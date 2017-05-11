#include "test_common.h"

// TODO fix fail
// But it seems like the precision problem of float
BOOST_AUTO_TEST_CASE(test_cc_gpu_float)
{
	float a[] = {
		108, 215, 3,   136, 218, 154, 162, 90,  63,  222, 57,  179, 119, 60,  215,
		72,  141, 166, 12,  240, 43,  93,  71,  173, 5,   20,  25,  226, 22,  6,
		156, 168, 12,  47,  188, 93,  70,  191, 172, 74,  69,  180, 48,  130, 179,
		229, 3,   98,  80,  204, 32,  37,  231, 174, 93,  203, 60,  97,  206, 2,
		71,  232, 230, 25,  24,  28,  249, 129, 237, 93,  6,   215, 217, 168, 151,
		197, 201, 251, 97,  34,  29,  252, 25,  155, 213, 15,  52,  154, 175, 186,
		97,  227, 38,  189, 141, 92,  69,  224, 183, 65,  116, 10,  52,  142, 10,
		224, 136, 254, 129, 14,  197, 67,  60,  28,  110, 96,  142, 35,  56,  234,
		198, 114, 199, 221, 97,  48,  158, 210, 111, 131, 186, 37,  128, 60,  50,
		110, 104, 192, 149, 240, 79,  176, 124, 175, 68,  143, 151, 245, 221, 114,
		197, 141, 56,  156, 31,  158, 11,  186, 86,  198, 242, 195, 122, 171, 191,
		90,  230, 8,   174, 36,  137, 161, 12,  40,  136, 178, 252, 15,  248, 197,
		154, 157, 50,  103, 33,  66,  57,  104, 133, 231, 84,  107, 182, 218, 52,
		218, 118, 23,  247, 52,  82,  190, 8,   174, 219, 50,  10,  49,  131, 183,
		149, 89,  42,  174, 225, 65,  164, 143, 21,  150, 145, 235, 33,  112, 136,
		176, 158, 175, 212, 131, 39,  232, 246, 132, 175, 223, 66,  193, 248, 178,
		202, 180, 234, 64,  106, 109, 46,  30,  216, 182, 26,  125, 256, 51,  150,
		104 };
	float b[] = {
		61,  105, 249, 207, 167, 57,  100, 141, 154, 112, 213, 138, 75,  58,  130,
		249, 118, 47,  78,  251, 196, 111, 40,  88,  94,  34,  53,  134, 185, 121,
		243, 26,  225, 50,  76,  198, 16,  20,  82,  65,  11,  206, 31,  211, 177,
		102, 158, 188, 183, 48,  26,  207, 137, 36,  67,  215, 39,  51,  148, 201,
		155, 40,  72,  67,  229, 1,   96,  30,  11,  64,  56,  68,  251, 104, 156,
		98,  20,  149, 164, 245, 199, 184, 232, 148, 107, 142, 53,  196, 95,  140,
		29,  122, 40,  229, 131, 207, 242, 143, 205, 228, 106, 26,  89,  190, 110,
		204, 3,   8,   149, 95,  214, 128, 250, 103, 140, 182, 151, 131, 242, 166,
		55,  212, 93,  201, 52,  10,  91,  247, 58,  45,  172, 4,   194, 47,  193,
		177, 61,  150, 68,  118, 159, 154, 79,  159, 129, 142, 132, 156, 15,  77,
		211, 101, 78,  122, 108, 93,  67,  60,  23,  52,  8,   14,  211, 222, 117,
		57,  19,  111, 189, 5,   139, 6,   46,  4,   112, 17,  182, 155, 206, 83,
		237, 57,  43,  101, 119, 131, 243, 246, 120, 130, 215, 48,  67,  153, 122,
		213, 15,  99,  28,  243, 196, 135, 193, 119, 3,   146, 104, 25,  141, 92,
		76,  97,  145, 192, 163, 207, 109, 98,  138, 225, 13,  91,  103, 92,  106,
		157, 198, 100, 131, 10,  150, 58,  78,  256, 239, 69,  39,  38,  35,  23,
		185, 221, 145, 110, 205, 175, 55,  94,  214, 233, 173, 253, 120, 168, 209,
		151 };
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
	cuda_error = block_match_cc(dev_a, dev_b, 1, 1, a_length, dev_c, 1, 1,
		cudaStreamDefault);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	float c;
	cuda_error = cudaMemcpy(&c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	BOOST_CHECK_SMALL(c - 0.0014f, singleFloatingPointErrorTolerance);
	cuda_error = cudaFree(dev_a);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
}

BOOST_AUTO_TEST_CASE(test_cc_gpu_double)
{
	double a[] = {
		108, 215, 3,   136, 218, 154, 162, 90,  63,  222, 57,  179, 119, 60,  215,
		72,  141, 166, 12,  240, 43,  93,  71,  173, 5,   20,  25,  226, 22,  6,
		156, 168, 12,  47,  188, 93,  70,  191, 172, 74,  69,  180, 48,  130, 179,
		229, 3,   98,  80,  204, 32,  37,  231, 174, 93,  203, 60,  97,  206, 2,
		71,  232, 230, 25,  24,  28,  249, 129, 237, 93,  6,   215, 217, 168, 151,
		197, 201, 251, 97,  34,  29,  252, 25,  155, 213, 15,  52,  154, 175, 186,
		97,  227, 38,  189, 141, 92,  69,  224, 183, 65,  116, 10,  52,  142, 10,
		224, 136, 254, 129, 14,  197, 67,  60,  28,  110, 96,  142, 35,  56,  234,
		198, 114, 199, 221, 97,  48,  158, 210, 111, 131, 186, 37,  128, 60,  50,
		110, 104, 192, 149, 240, 79,  176, 124, 175, 68,  143, 151, 245, 221, 114,
		197, 141, 56,  156, 31,  158, 11,  186, 86,  198, 242, 195, 122, 171, 191,
		90,  230, 8,   174, 36,  137, 161, 12,  40,  136, 178, 252, 15,  248, 197,
		154, 157, 50,  103, 33,  66,  57,  104, 133, 231, 84,  107, 182, 218, 52,
		218, 118, 23,  247, 52,  82,  190, 8,   174, 219, 50,  10,  49,  131, 183,
		149, 89,  42,  174, 225, 65,  164, 143, 21,  150, 145, 235, 33,  112, 136,
		176, 158, 175, 212, 131, 39,  232, 246, 132, 175, 223, 66,  193, 248, 178,
		202, 180, 234, 64,  106, 109, 46,  30,  216, 182, 26,  125, 256, 51,  150,
		104 };
	double b[] = {
		61,  105, 249, 207, 167, 57,  100, 141, 154, 112, 213, 138, 75,  58,  130,
		249, 118, 47,  78,  251, 196, 111, 40,  88,  94,  34,  53,  134, 185, 121,
		243, 26,  225, 50,  76,  198, 16,  20,  82,  65,  11,  206, 31,  211, 177,
		102, 158, 188, 183, 48,  26,  207, 137, 36,  67,  215, 39,  51,  148, 201,
		155, 40,  72,  67,  229, 1,   96,  30,  11,  64,  56,  68,  251, 104, 156,
		98,  20,  149, 164, 245, 199, 184, 232, 148, 107, 142, 53,  196, 95,  140,
		29,  122, 40,  229, 131, 207, 242, 143, 205, 228, 106, 26,  89,  190, 110,
		204, 3,   8,   149, 95,  214, 128, 250, 103, 140, 182, 151, 131, 242, 166,
		55,  212, 93,  201, 52,  10,  91,  247, 58,  45,  172, 4,   194, 47,  193,
		177, 61,  150, 68,  118, 159, 154, 79,  159, 129, 142, 132, 156, 15,  77,
		211, 101, 78,  122, 108, 93,  67,  60,  23,  52,  8,   14,  211, 222, 117,
		57,  19,  111, 189, 5,   139, 6,   46,  4,   112, 17,  182, 155, 206, 83,
		237, 57,  43,  101, 119, 131, 243, 246, 120, 130, 215, 48,  67,  153, 122,
		213, 15,  99,  28,  243, 196, 135, 193, 119, 3,   146, 104, 25,  141, 92,
		76,  97,  145, 192, 163, 207, 109, 98,  138, 225, 13,  91,  103, 92,  106,
		157, 198, 100, 131, 10,  150, 58,  78,  256, 239, 69,  39,  38,  35,  23,
		185, 221, 145, 110, 205, 175, 55,  94,  214, 233, 173, 253, 120, 168, 209,
		151 };
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
	cuda_error = block_match_cc(dev_a, dev_b, 1, 1, a_length, dev_c, 1, 1,
		cudaStreamDefault);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	double c;
	cuda_error = cudaMemcpy(&c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
	BOOST_CHECK_SMALL(c - 0.0014, doubleFloatingPointErrorTolerance);
	cuda_error = cudaFree(dev_a);
	BOOST_CHECK_EQUAL(cuda_error, cudaSuccess);
}