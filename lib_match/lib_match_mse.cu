#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "template_instantiate_helper.h"

template <typename Type>
__device__ inline Type
mean_square_error(const Type *A, const Type *B, const int size)
{
	Type t = 0;
	for (int i = 0; i<size; ++i)
	{
		Type cc = A[i] - B[i];
		t += cc*cc;
	}

	return t / size;
}

template <typename Type>
__global__ void
array_match_mse_kernel(const Type *A, const Type *B, const int size, Type *C,
	const int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	const Type *c_A = A + tid * size;
	const Type *c_B = B + tid * size;

	C[tid] = mean_square_error(c_A, c_B, size);
}

template <typename Type>
__device__ inline void
block_match_mse_kernel_helper(const Type *blocks_A, const Type *blocks_B, const int numberOfBlockBPerBlockA, const int blockSize,
	Type *resultsBuffer, const int tid)
{
	int groupIndex = tid / numberOfBlockBPerBlockA;

	int inGroupOffset = tid % numberOfBlockBPerBlockA;

	const Type *c_block_A = blocks_A + groupIndex * blockSize;
	const Type *c_block_B = blocks_B + groupIndex * numberOfBlockBPerBlockA * blockSize + inGroupOffset * blockSize;

	resultsBuffer[tid] = mean_square_error(c_block_A, c_block_B, blockSize);
}

template <typename Type>
__global__ void
block_match_mse_kernel(const Type *blocks_A, const Type *blocks_B, const int numberOfBlockBPerBlockA, const int blockSize, Type *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	block_match_mse_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
__global__ void
block_match_mse_kernel(const Type *blocks_A, const Type *blocks_B, const int numberOfBlockBPerBlockA, const int blockSize, Type *resultsBuffer,
	const int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	block_match_mse_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
cudaError_t block_match_mse(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	block_match_mse_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result);
	return cudaGetLastError();
}

template <typename Type>
cudaError_t block_match_mse_check_border(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	block_match_mse_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result, numberOfBlockBPerBlockA * numBlocks_A);
	return cudaGetLastError();
}

template <typename Type>
cudaError_t array_match_mse_check_border(const Type *A, const Type *B, const int numberOfArray,
	const int size, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	array_match_mse_kernel << <numProcessors, numThreads, 0, stream >> >
		(A, B, size, result, numberOfArray);
	return cudaGetLastError();
}

#define EXP(type) \
template \
cudaError_t block_match_mse(const type *, const type *, const int, \
	const int, const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP

#define EXP(type) \
template \
cudaError_t block_match_mse_check_border(const type *, const type *, const int, \
	const int, const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP

#define EXP(type) \
template \
cudaError_t array_match_mse_check_border(const type *, const type *, const int, \
	const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP