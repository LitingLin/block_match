#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "template_instantiate_helper.h"

template <typename Type>
__device__ inline Type
accumulate(const Type *begin, const int size)
{
	Type sum = static_cast<Type>(0);
	for (int i = 0; i != size; ++i)
	{
		sum += begin[i];
	}
	return sum;
}

template <typename Type>
__device__ inline Type
correlation_coefficient(const Type *A, const Type *B, const int size)
{
	Type X = 0, Y = 0, Z = 0;
	
	Type A_mean = accumulate(A, size) / static_cast<Type>(size);
	Type B_mean = accumulate(B, size) / static_cast<Type>(size);

	for (int i = 0; i < size; ++i)
	{
		Type M = A[i] - A_mean;
		Type N = B[i] - B_mean;
		X += M * N;
		Y += M * M;
		Z += N * N;
	}

	return X / sqrt(Y*Z);
}

template <typename Type>
__global__ void
lib_match_cc_global_kernel(const Type *A, const Type *B, const int numberOfBPerA, const int size, Type *C,
	const int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;
		
	int groupIndex = tid / numberOfBPerA;

	int inGroupOffset = tid % numberOfBPerA;

	const Type *c_A = A + groupIndex * size;
	const Type *c_B = B + inGroupOffset * size;

	C[tid] = correlation_coefficient(c_A, c_B, size);
}

template <typename Type>
__device__ inline void
lib_match_cc_kernel_helper(const Type *blocks_A, const Type *blocks_B, const int numberOfBlockBPerBlockA, const int blockSize,
	Type *resultsBuffer, const int tid)
{
	int groupIndex = tid / numberOfBlockBPerBlockA;
	
	const Type *c_block_A = blocks_A + groupIndex * blockSize;
	const Type *c_block_B = blocks_B + tid * blockSize;

	resultsBuffer[tid] = correlation_coefficient(c_block_A, c_block_B, blockSize);
}

template <typename Type>
__global__ void
lib_match_cc_kernel_offset(const Type *blocks_A, const Type *blocks_B, Type *resultsBuffer,const int blockSize, const int *offsetA,const int numberOfB)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= numberOfB)
		return;
			
	const Type *c_block_A = blocks_A + offsetA[tid] * blockSize;
	const Type *c_block_B = blocks_B + tid * blockSize;

	resultsBuffer[tid] = correlation_coefficient(c_block_A, c_block_B, blockSize);
}

template <typename Type>
__global__ void
lib_match_cc_kernel(const Type *blocks_A, const Type *blocks_B, const int numberOfBlockBPerBlockA, const int blockSize, Type *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	lib_match_cc_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
__global__ void
lib_match_cc_kernel(const Type *blocks_A, const Type *blocks_B, const int numberOfBlockBPerBlockA, const int blockSize, Type *resultsBuffer,
	const int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;
		
	lib_match_cc_kernel_helper(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, resultsBuffer, tid);
}

template <typename Type>
cudaError_t lib_match_cc(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	lib_match_cc_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result);

	return cudaGetLastError();
}

template <typename Type>
cudaError_t lib_match_cc_check_border(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	lib_match_cc_kernel <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result, numberOfBlockBPerBlockA*numBlocks_A);

	return cudaGetLastError();
}

template <typename Type>
cudaError_t lib_match_cc_global(const Type *A, const Type *B, const int numberOfA,
	const int numberOfBPerA, const int size, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	lib_match_cc_global_kernel << <numProcessors, numThreads, 0, stream >> >
		(A, B, numberOfBPerA, size, result, numberOfA * numberOfBPerA);

	return cudaGetLastError();
}

template <typename Type>
cudaError_t lib_match_cc_offset(const Type *blocks_A, const Type *blocks_B, const int *offsetA,
	const int numberOfB, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream)
{
	lib_match_cc_kernel_offset <<<numProcessors, numThreads, 0, stream >>> 
		(blocks_A, blocks_B,result, blockSize, offsetA, numberOfB);
	return cudaGetLastError();
}

#define EXP(type) \
template \
cudaError_t lib_match_cc(const type *, const type *, const int, \
	const int, const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP

#define EXP(type) \
template \
cudaError_t lib_match_cc_check_border(const type *, const type *, const int, \
	const int, const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP

#define EXP(type) \
template \
cudaError_t lib_match_cc_global(const type *, const type *, const int, \
	const int, const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP

#define EXP(type) \
template \
cudaError_t lib_match_cc_offset(const type *, const type *, const int*, \
	const int, const int, type *, const int, const int, const cudaStream_t)
InstantiateTemplateFloating(EXP);
#undef EXP
