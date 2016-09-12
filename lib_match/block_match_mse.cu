#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void
array_match_mse_kernel(const float *block_A, const float *block_B, int blockSize, float *result)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const float *c_block_A = block_A + tid * blockSize;
	const float *c_block_B = block_B + tid * blockSize;

	float temp = 0;
	for (int i = 0;i<blockSize;++i)
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

__global__ void
block_match_mse_async_kernel(const float *blocks_A, const float *blocks_B, int block_B_groupSize, int blockSize, float *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	const int block_A_groupSize = 1;
	int blockGroupSize = block_A_groupSize * block_B_groupSize;

	int groupIndex = tid / blockGroupSize;

	int inGroupOffset = tid % blockGroupSize;

	int block_A_index = inGroupOffset / block_B_groupSize;

	int block_B_index = inGroupOffset % block_B_groupSize;

	const float *c_block_A = blocks_A + groupIndex * block_A_groupSize * blockSize + block_A_index * blockSize;
	const float *c_block_B = blocks_B + groupIndex * block_B_groupSize * blockSize + block_B_index * blockSize;

	float temp = 0;
	for (int i = 0; i<blockSize; ++i)
	{
		float cc = c_block_A[i] - c_block_B[i];
		temp += cc*cc;
	}

	temp /= blockSize;

	resultsBuffer[tid] = temp;
}

__global__ void
block_match_mse_async_kernel(const float *blocks_A, const float *blocks_B, int block_B_groupSize, int blockSize, float *resultsBuffer, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	const int block_A_groupSize = 1;
	int blockGroupSize = block_A_groupSize * block_B_groupSize;

	int groupIndex = tid / blockGroupSize;

	int inGroupOffset = tid % blockGroupSize;

	int block_A_index = inGroupOffset / block_B_groupSize;

	int block_B_index = inGroupOffset % block_B_groupSize;

	const float *c_block_A = blocks_A + groupIndex * block_A_groupSize * blockSize + block_A_index * blockSize;
	const float *c_block_B = blocks_B + groupIndex * block_B_groupSize * blockSize + block_B_index * blockSize;

	float temp = 0;
	for (int i = 0; i<blockSize; ++i)
	{
		float cc = c_block_A[i] - c_block_B[i];
		temp += cc*cc;
	}

	temp /= blockSize;

	resultsBuffer[tid] = temp;
}

cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads)
{
	array_match_mse_kernel << <numberOfProcessors, numberOfThreads >> > (A, B, lengthOfArray, C);
	return cudaGetLastError();
}

cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads)
{
	array_match_mse_kernel << <numberOfProcessors, numberOfThreads >> > (A, B, lengthOfArray, C, numberOfArray);
	return cudaGetLastError();
}

cudaError_t block_match_mse(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	block_match_mse_async_kernel << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result);
	return cudaGetLastError();
}

cudaError_t block_match_mse_check_border(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	block_match_mse_async_kernel << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result, numBlocks_B);
	return cudaGetLastError();
}