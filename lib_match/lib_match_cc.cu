#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename Type>
__device__ inline Type
accumulate(const Type *begin, const Type *end)
{
	Type sum = static_cast<Type>(0);
	for (const Type *iter = begin; iter != end; ++iter)
	{
		sum += *iter;
	}
	return sum;
}

template <typename Type>
__device__ inline void
lib_match_cc_kernel_helper(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize,
	Type *resultsBuffer, int tid)
{
	int groupIndex = tid / numberOfBlockBPerBlockA;

	int inGroupOffset = tid % numberOfBlockBPerBlockA;

	const Type *c_block_A = blocks_A + groupIndex * blockSize;
	const Type *c_block_B = blocks_B + groupIndex * numberOfBlockBPerBlockA * blockSize + inGroupOffset * blockSize;
		
	Type X = 0, Y = 0, Z = 0;
	Type A_mean = accumulate(c_block_A, c_block_A + blockSize) / static_cast<Type>(blockSize);
	Type B_mean = accumulate(c_block_B, c_block_B + blockSize) / static_cast<Type>(blockSize);

	for (int i = 0; i < blockSize; ++i)
	{
		Type M = c_block_A[i] - A_mean;
		Type N = c_block_B[i] - B_mean;
		X += M * N;
		Y += M * M;
		Z += N * N;
	}
	
	resultsBuffer[tid] = X / sqrt(Y*Z);
}

template <typename Type>
__global__ void
lib_match_cc_kernel(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, Type *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	lib_match_cc_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
__global__ void
lib_match_cc_kernel(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, Type *resultsBuffer, 
	int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;
		
	lib_match_cc_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
cudaError_t lib_match_cc(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	lib_match_cc_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result);

	return cudaGetLastError();
}

template <typename Type>
cudaError_t lib_match_cc_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A, 
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	lib_match_cc_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result, numberOfBlockBPerBlockA*numBlocks_A);

	return cudaGetLastError();
}

template
cudaError_t lib_match_cc(float *, float *, int,
	int, int, float *, int, int, cudaStream_t);
template
cudaError_t lib_match_cc(double *, double *, int,
	int, int, double *, int, int, cudaStream_t);
template
cudaError_t lib_match_cc_check_border(float *, float *, int,
	int, int, float *, int, int, cudaStream_t);
template
cudaError_t lib_match_cc_check_border(double *, double *, int,
	int, int, double *, int, int, cudaStream_t);