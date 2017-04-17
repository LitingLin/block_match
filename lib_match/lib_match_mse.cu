#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename Type>
__device__ inline void
lib_match_mse_kernel_helper(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, 
	Type *resultsBuffer, int tid)
{
	int groupIndex = tid / numberOfBlockBPerBlockA;

	int inGroupOffset = tid % numberOfBlockBPerBlockA;

	const Type *c_block_A = blocks_A + groupIndex * blockSize;
	const Type *c_block_B = blocks_B + groupIndex * numberOfBlockBPerBlockA * blockSize + inGroupOffset * blockSize;

	Type temp = 0;
	for (int i = 0; i<blockSize; ++i)
	{
		Type cc = c_block_A[i] - c_block_B[i];
		temp += cc*cc;
	}

	temp /= blockSize;

	resultsBuffer[tid] = temp;
}


template <typename Type>
__global__ void
lib_match_mse_kernel(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, Type *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	lib_match_mse_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
__global__ void
lib_match_mse_kernel(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, Type *resultsBuffer,
	int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	lib_match_mse_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
cudaError_t lib_match_mse(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	lib_match_mse_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result);
	return cudaGetLastError();
}

template <typename Type>
cudaError_t lib_match_mse_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	lib_match_mse_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result, numberOfBlockBPerBlockA * numBlocks_A);
	return cudaGetLastError();
}

template
cudaError_t lib_match_mse(float *, float *, int,
	int, int, float *, int, int, cudaStream_t);
template
cudaError_t lib_match_mse(double *, double *, int,
	int, int, double *, int, int, cudaStream_t);
template
cudaError_t lib_match_mse_check_border(float *, float *, int,
	int, int, float *, int, int, cudaStream_t);
template
cudaError_t lib_match_mse_check_border(double *, double *, int,
	int, int, double *, int, int, cudaStream_t);