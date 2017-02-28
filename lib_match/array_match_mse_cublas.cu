#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

__global__ void
array_match_mse_kernel(const float *block_A, const float *block_B, int blockSize, float *result)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const float *c_block_A = block_A + tid * blockSize;
	const float *c_block_B = block_B + tid * blockSize;

	float temp = 0;
	for (int i = 0; i<blockSize; ++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}
	temp /= blockSize;
	result[tid] = temp;
}

__global__ void
array_match_mse_kernel(const float *block_A, const float *block_B, int blockSize, float *result, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	const float *c_block_A = block_A + tid * blockSize;
	const float *c_block_B = block_B + tid * blockSize;

	float temp = 0;
	for (int i = 0; i<blockSize; ++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}
	temp /= blockSize;
	result[tid] = temp;
}

cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads)
{
}


cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads)
{
	
}