#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename Type>
__global__ void
array_match_mse_kernel(const Type *block_A, const Type *block_B, int blockSize, Type *result)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const Type *c_block_A = block_A + tid * blockSize;
	const Type *c_block_B = block_B + tid * blockSize;

	Type temp = 0;
	for (int i = 0;i<blockSize;++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}
	temp /= blockSize;
	result[tid] = temp;
}

template <typename Type>
__global__ void
array_match_mse_kernel(const Type *block_A, const Type *block_B, int blockSize, Type *result, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;
	
	const Type *c_block_A = block_A + tid * blockSize;
	const Type *c_block_B = block_B + tid * blockSize;

	Type temp = 0;
	for (int i = 0; i<blockSize; ++i)
	{
		temp += (c_block_A[i] - c_block_B[i]) * (c_block_A[i] - c_block_B[i]);
	}
	temp /= blockSize;
	result[tid] = temp;
}

template <typename Type>
__global__ void
block_match_mse_async_kernel(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, Type *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

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
block_match_mse_async_kernel(const Type *blocks_A, const Type *blocks_B, int numberOfBlockBPerBlockA, int blockSize, Type *resultsBuffer, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;
		
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
cudaError_t arrayMatchMse(Type *A, Type *B, Type *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads)
{
	array_match_mse_kernel << <numberOfProcessors, numberOfThreads >> > (A, B, lengthOfArray, C);
	return cudaGetLastError();
}

template <typename Type>
cudaError_t arrayMatchMse(Type *A, Type *B, Type *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads)
{
	array_match_mse_kernel << <numberOfProcessors, numberOfThreads >> > (A, B, lengthOfArray, C, numberOfArray);
	return cudaGetLastError();
}

template <typename Type>
cudaError_t block_match_mse(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	block_match_mse_async_kernel << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result);
	return cudaGetLastError();
}

template <typename Type>
cudaError_t block_match_mse_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int numberOfBlockBPerBlockA, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	block_match_mse_async_kernel << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, numberOfBlockBPerBlockA, blockSize, result, 
		numberOfBlockBPerBlockA * numBlocks_A);
	return cudaGetLastError();
}

template
cudaError_t block_match_mse(float *, float *, int,
	int, int, float *, int, int, cudaStream_t);
template
cudaError_t block_match_mse(double *, double *, int,
	int, int, double *, int, int, cudaStream_t);
template
cudaError_t block_match_mse_check_border(float *, float *, int,
	int, int, float *, int, int, cudaStream_t);
template
cudaError_t block_match_mse_check_border(double *, double *, int,
	int, int, double *, int, int, cudaStream_t);
template
cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads);
template
cudaError_t arrayMatchMse(double *A, double *B, double *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads);
template
cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);
template
cudaError_t arrayMatchMse(double *A, double *B, double *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);