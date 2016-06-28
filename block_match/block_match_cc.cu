#include "block_match.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

__global__ void
standardize_block(float *data, size_t blockSize)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	float *c_data = data + tid * blockSize;

	float mean = 0;
	for (size_t i = 0; i < blockSize; ++i)
	{
		mean += c_data[i];
	}
	mean /= blockSize;

	float sd = 0;
	for (size_t i = 0; i < blockSize; ++i)
	{
		float d = c_data[i] -= mean;
		sd += d*d;
	}

	sd /= blockSize;
	sd = sqrtf(sd);

	for (size_t i = 0; i < blockSize; ++i)
	{
		c_data[i] /= sd;
	}
}

__global__ void
standardize_block(float *data, size_t blockSize, size_t n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	float *c_data = data + tid * blockSize;

	float mean = 0;
	for (size_t i = 0; i < blockSize; ++i)
	{
		mean += c_data[i];
	}
	mean /= blockSize;

	float sd = 0;
	for (size_t i = 0; i < blockSize; ++i)
	{
		float d = c_data[i] -= mean;
		sd += d*d;
	}

	sd /= blockSize;
	sd = sqrtf(sd);

	for (size_t i = 0; i < blockSize; ++i)
	{
		c_data[i] /= sd;
	}
}

__global__ void
vector_multiply_add(float *block_A, float *block_B, size_t blockSize, float *result)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const float *c_block_A = block_A + blockIdx.x * blockSize;
	const float *c_block_B = block_B + threadIdx.x * blockSize;

	float temp = 0;
	for (size_t i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	result[tid] = temp;
}

__global__ void
vector_multiply_add_async(const float *blocks_A, const float *blocks_B, size_t block_B_groupSize, size_t blockSize, float *resultsBuffer)
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
	for (size_t i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	resultsBuffer[tid] = temp;
}

__global__ void
vector_multiply_add_async(const float *blocks_A, const float *blocks_B, size_t block_B_groupSize, size_t blockSize, float *resultsBuffer, size_t n)
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
	for (size_t i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	resultsBuffer[tid] = temp;
}

cudaError_t block_match_cc(float * block_A, float * block_B, size_t numBlock_A, size_t numBlock_B, size_t blockSize, float * result, cudaStream_t stream)
{
	standardize_block << <1, numBlock_A, 0, stream >> > (block_A, blockSize);
	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		return cuda_error;
	standardize_block << <1, numBlock_B, 0, stream >> > (block_B, blockSize);
	cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		return cuda_error;
	cuda_error = cudaStreamSynchronize(stream);
	if (cuda_error != cudaSuccess)
		return cuda_error;
	vector_multiply_add << <numBlock_A, numBlock_B, 0, stream >> > (block_A, block_B, blockSize, result);
	cuda_error = cudaGetLastError();

	return cuda_error;
}

cudaError_t block_match_cc_async(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream)
{
	standardize_block << <(numBlocks_A + numThreads - 1) / numThreads, numThreads, 0, stream >> > (blocks_A, blockSize, numBlocks_A);

	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		return cuda_error;

	standardize_block << <(numBlocks_B + numThreads - 1) / numThreads, numThreads, 0, stream >> > (blocks_B, blockSize, numBlocks_B);

	cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		return cuda_error;
	
	vector_multiply_add_async << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result, numTasks);
	cuda_error = cudaGetLastError();

	return cuda_error;
}

cudaError_t block_match_cc_async(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_blockSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	standardize_block << <(numBlocks_A + numThreads - 1) / numThreads, numThreads, 0, stream >> > (blocks_A, blockSize, numBlocks_A);

	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		return cuda_error;

	standardize_block << <(numBlocks_B + numThreads - 1) / numThreads, numThreads, 0, stream >> > (blocks_B, blockSize, numBlocks_B);

	cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
		return cuda_error;
	
	vector_multiply_add_async << <numProcessors, numThreads,0,stream >> > (blocks_A, blocks_B, block_B_blockSize, blockSize, result);
	cuda_error = cudaGetLastError();

	return cuda_error;
}
