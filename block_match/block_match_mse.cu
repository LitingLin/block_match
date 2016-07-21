#include "block_match_internal.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void
block_match_mse_kernel(const float *block_A, const float *block_B, size_t blockSize, float *result)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const float *c_block_A = block_A + blockIdx.x * blockSize;
	const float *c_block_B = block_B + threadIdx.x * blockSize;

	float temp = 0;
	for (size_t i = 0;i<blockSize;++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}
	temp /= blockSize;
	result[tid] = temp;
}

__global__ void
block_match_mse_async_kernel(const float *blocks_A, const float *blocks_B, size_t block_B_groupSize, size_t blockSize, float *resultsBuffer)
{
	const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	const size_t block_A_groupSize = 1;
	size_t blockGroupSize = block_A_groupSize * block_B_groupSize;

	size_t groupIndex = tid / blockGroupSize;

	size_t inGroupOffset = tid % blockGroupSize;

	size_t block_A_index = inGroupOffset / block_B_groupSize;

	size_t block_B_index = inGroupOffset % block_B_groupSize;

	const float *c_block_A = blocks_A + groupIndex * block_A_groupSize * blockSize + block_A_index * blockSize;
	const float *c_block_B = blocks_B + groupIndex * block_B_groupSize * blockSize + block_B_index * blockSize;

	float temp = 0;
	for (size_t i = 0; i<blockSize; ++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}

	temp /= blockSize;

	resultsBuffer[tid] = temp;
}

__global__ void
block_match_mse_async_kernel(const float *blocks_A, const float *blocks_B, size_t block_B_groupSize, size_t blockSize, float *resultsBuffer, size_t n)
{
	const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	const size_t block_A_groupSize = 1;
	size_t blockGroupSize = block_A_groupSize * block_B_groupSize;

	size_t groupIndex = tid / blockGroupSize;

	size_t inGroupOffset = tid % blockGroupSize;

	size_t block_A_index = inGroupOffset / block_B_groupSize;

	size_t block_B_index = inGroupOffset % block_B_groupSize;

	const float *c_block_A = blocks_A + groupIndex * block_A_groupSize * blockSize + block_A_index * blockSize;
	const float *c_block_B = blocks_B + groupIndex * block_B_groupSize * blockSize + block_B_index * blockSize;

	float temp = 0;
	for (size_t i = 0; i<blockSize; ++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}

	temp /= blockSize;

	resultsBuffer[tid] = temp;
}

cudaError_t block_match_mse(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	block_match_mse_async_kernel << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result);
	return cudaGetLastError();
}

cudaError_t block_match_mse_ch(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream)
{
	block_match_mse_async_kernel << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result, numTasks);
	return cudaGetLastError();
}