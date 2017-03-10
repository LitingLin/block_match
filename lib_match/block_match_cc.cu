#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename Type>
__global__ void
standardize_block_kernel(Type *data, int blockSize)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	Type *c_data = data + tid * blockSize;

	Type mean = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		mean += c_data[i];
	}
	mean /= blockSize;

	Type sd = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		Type d = c_data[i] -= mean;
		sd += d*d;
	}

	sd /= blockSize;
	sd = sqrt(sd);

	for (int i = 0; i < blockSize; ++i)
	{
		c_data[i] /= sd;
	}
}

template <typename Type>
__global__ void
standardize_block_kernel(Type *data, int blockSize, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	Type *c_data = data + tid * blockSize;

	Type mean = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		mean += c_data[i];
	}
	mean /= blockSize;

	Type sd = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		Type d = c_data[i] -= mean;
		sd += d*d;
	}

	sd /= blockSize;
	sd = sqrt(sd);

	for (int i = 0; i < blockSize; ++i)
	{
		c_data[i] /= sd;
	}
}

template <typename Type>
__global__ void
vector_multiply_add(Type *block_A, Type *block_B, int blockSize, Type *result)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const Type *c_block_A = block_A + blockIdx.x * blockSize;
	const Type *c_block_B = block_B + threadIdx.x * blockSize;

	Type temp = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	result[tid] = temp;
}

template <typename Type>
__global__ void
array_vector_multiply_add(Type *block_A, Type *block_B, int blockSize, Type *result, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	const Type *c_block_A = block_A + tid * blockSize;
	const Type *c_block_B = block_B + tid * blockSize;

	Type temp = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	result[tid] = temp;
}


template <typename Type>
__global__ void
vector_multiply_add(Type *block_A, Type *block_B, int blockSize, Type *result, int n)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= n)
		return;

	const Type *c_block_A = block_A + blockIdx.x * blockSize;
	const Type *c_block_B = block_B + threadIdx.x * blockSize;

	Type temp = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	result[tid] = temp;
}

template <typename Type>
__global__ void
vector_multiply_add(const Type *blocks_A, const Type *blocks_B, int block_B_groupSize, int blockSize, Type *resultsBuffer)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;

	const int block_A_groupSize = 1;

	int blockGroupSize = block_A_groupSize * block_B_groupSize;

	int groupIndex = tid / blockGroupSize;

	int inGroupOffset = tid % blockGroupSize;

	int block_A_index = inGroupOffset / block_B_groupSize;

	int block_B_index = inGroupOffset % block_B_groupSize;

	const Type *c_block_A = blocks_A + groupIndex * block_A_groupSize * blockSize + block_A_index * blockSize;
	const Type *c_block_B = blocks_B + groupIndex * block_B_groupSize * blockSize + block_B_index * blockSize;

	Type temp = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	resultsBuffer[tid] = temp;
}

template <typename Type>
__global__ void
vector_multiply_add(const Type *blocks_A, const Type *blocks_B, int block_B_groupSize, int blockSize, Type *resultsBuffer, int n)
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

	const Type *c_block_A = blocks_A + groupIndex * block_A_groupSize * blockSize + block_A_index * blockSize;
	const Type *c_block_B = blocks_B + groupIndex * block_B_groupSize * blockSize + block_B_index * blockSize;

	Type temp = 0;
	for (int i = 0; i < blockSize; ++i)
	{
		temp += c_block_A[i] * c_block_B[i];
	}

	resultsBuffer[tid] = temp;
}

template <typename Type>
cudaError_t standardize(Type *sequence, int numberOfBlocks, int size, int numThreads, cudaStream_t stream)
{
	standardize_block_kernel << <(numberOfBlocks + numThreads - 1) / numThreads, numThreads, 0, stream >> > (sequence, size, numberOfBlocks);
	cudaError_t cuda_error = cudaGetLastError();
	return cuda_error;
}

template <typename Type>
cudaError_t standardize(Type *sequence, int numberOfBlocks, int size, int numThreads)
{
	standardize_block_kernel << <(numberOfBlocks + numThreads - 1) / numThreads, numThreads>> > (sequence, size, numberOfBlocks);
	cudaError_t cuda_error = cudaGetLastError();
	return cuda_error;
}

template <typename Type>
cudaError_t arrayMatchCc(Type *A, Type *B, Type *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads)
{
	cudaError_t cudaError = standardize<Type>(A, numberOfArray, lengthOfArray, numberOfThreads);

	if (cudaError != cudaSuccess)
		return cudaError;

	cudaError = standardize<Type>(B, numberOfArray, lengthOfArray, numberOfThreads);

	if (cudaError != cudaSuccess)
		return cudaError;

	array_vector_multiply_add << <numberOfProcessors, numberOfThreads >> > (A, B, lengthOfArray, C, numberOfArray);
	cudaError = cudaGetLastError();

	return cudaError;
}

template <typename Type>
cudaError_t block_match_cc(Type *blocks_A, Type *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	cudaError_t cuda_error = standardize(blocks_A, numBlocks_A, blockSize, numThreads, stream);

	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = standardize(blocks_B, numBlocks_B, blockSize, numThreads, stream);

	if (cuda_error != cudaSuccess)
		return cuda_error;

	vector_multiply_add << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result);
	cuda_error = cudaGetLastError();

	return cuda_error;
}

template <typename Type>
cudaError_t block_match_cc_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream)
{
	cudaError_t cuda_error = standardize(blocks_A, numBlocks_A, blockSize, numThreads, stream);

	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = standardize(blocks_B, numBlocks_B, blockSize, numThreads, stream);

	if (cuda_error != cudaSuccess)
		return cuda_error;

	vector_multiply_add << <numProcessors, numThreads, 0, stream >> > (blocks_A, blocks_B, block_B_groupSize, blockSize, result, numBlocks_B);
	cuda_error = cudaGetLastError();

	return cuda_error;
}

template
cudaError_t standardize(float *sequence, int numberOfBlocks, int size, int numThreads);
template
cudaError_t standardize(double *sequence, int numberOfBlocks, int size, int numThreads);
template
cudaError_t standardize(float *sequence, int numberOfBlocks, int size, int numThreads, cudaStream_t stream);
template
cudaError_t standardize(double *sequence, int numberOfBlocks, int size, int numThreads, cudaStream_t stream);
template
cudaError_t block_match_cc(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
template
cudaError_t block_match_cc(double *blocks_A, double *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, double *result, int numProcessors, int numThreads, cudaStream_t stream);
template
cudaError_t block_match_cc_check_border(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
template
cudaError_t block_match_cc_check_border(double *blocks_A, double *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, double *result, int numProcessors, int numThreads, cudaStream_t stream);
template
cudaError_t arrayMatchCc(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);
template
cudaError_t arrayMatchCc(double *A, double *B, double *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);