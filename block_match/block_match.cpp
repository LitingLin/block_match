#include "block_match.h"

#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <functional>
#include <cuda_runtime.h>

#include "thread_pool.h"
#include "block_match_internal.h"
#include "block_match.cuh"

const uint32_t numSubmitThread = 2;
thread_pool *pool = nullptr;

enum Type
{
	FULL,
	COMBILE,
};

struct Context
{
	Type type;
	size_t matA_M;
	size_t matA_N;
	size_t matB_M;
	size_t matB_N;
	size_t block_M;
	size_t block_N;
	float *buffer_A;
	float *buffer_B;
	float *result_buffer;
	float *device_buffer_A;
	float *device_buffer_B;
	float *device_result_buffer;
	size_t result_dim0;
	size_t result_dim1;
	size_t result_dim2;
	size_t result_dim3;
};



bool initialize_TypeA(void **_instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N)
{
	struct Context * instance = (struct Context *)malloc(sizeof(struct Context));
	if (!instance)
		return false;

	instance->type = FULL;

	instance->matA_M = matA_M;
	instance->matA_N = matA_N;
	instance->matB_M = matB_M;
	instance->matB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	cudaError_t cuda_error = cudaMallocHost(&instance->buffer_A, (matA_M - block_M)*(matA_N - block_N)*block_M*block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}
	cuda_error = cudaMallocHost(&instance->buffer_B, (matB_M - block_M)*(matB_N - block_N)*block_M*block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer, (matA_M - block_M)*(matA_N - block_N)*(matB_M - block_M)*(matB_N - block_N) * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A, (matA_M - block_M)*(matA_N - block_N)*block_M*block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B, (matB_M - block_M)*(matB_N - block_N)*block_M*block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer, (matA_M - block_M)*(matA_N - block_N)*(matB_M - block_M)*(matB_N - block_N) * sizeof(float));

	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_B;
	}
	*_instance = instance;

	return true;

release_device_buffer_B:

	cudaFree(instance->device_buffer_B);
release_device_buffer_A:

	cudaFree(instance->device_buffer_A);
release_result_buffer:

	cudaFreeHost(instance->result_buffer);
release_buffer_B:

	cudaFreeHost(instance->buffer_B);
release_buffer_A:

	cudaFreeHost(instance->buffer_A);
release_instance:

	free(instance);
	return false;

}

void copyPatch(float *blockBuf, float *mat, size_t mat_M, size_t mat_N, size_t ind_x, size_t ind_y, size_t block_M, size_t block_N)
{
	float *c_blockBuf = blockBuf;
	float *c_mat = mat + ind_x * mat_N + ind_y;
	for (size_t i = 0; i < block_M; ++i)
	{
		memcpy(c_blockBuf, c_mat, block_N * sizeof(float));
		c_blockBuf += block_N;
		c_mat += mat_N;
	}
}

void copyPatchWithSymmetricPaddding(float *blockBuf, float *mat, size_t mat_M, size_t mat_N, size_t ind_x, size_t minus_ind_x, size_t ind_y, size_t minus_ind_y, size_t block_M, size_t block_N)
{
#ifdef _DEBUG
	if ((ind_x > 0 && minus_ind_x > 0) || (ind_y > 0 && minus_ind_y > 0))
		abort();
#endif

	float *c_blockBuf = blockBuf;
	float *c_mat = mat + ind_x * mat_N + ind_y;
	size_t ind_x_end, ind_x_end_after;
	size_t ind_y_end, ind_y_end_after;

	if (ind_x + block_M - minus_ind_x >= mat_M) {
		ind_x_end = mat_M;
		ind_x_end_after = ind_x + block_M - minus_ind_x - mat_M;
	}
	else
	{
		ind_x_end = ind_x + block_M - minus_ind_x;
		ind_x_end_after = 0;
	}
	if (ind_y + block_N - minus_ind_y >= mat_N)
	{
		ind_y_end = mat_N;
		ind_y_end_after = ind_y + block_N - minus_ind_y - mat_N;
	}
	else
	{
		ind_y_end = ind_y + block_N - minus_ind_y;
		ind_y_end_after = 0;
	}

	size_t offset;

	if (minus_ind_x)
	{
		float *t_mat = c_mat + mat_N * (minus_ind_x - 1);
		for (size_t i = 0; i < minus_ind_x; ++i)
		{
			if (minus_ind_y)
			{
				float *t_t_mat = t_mat + minus_ind_y - 1;
				for (size_t j = 0; j < minus_ind_y; ++j)
				{
					*c_blockBuf = *t_t_mat;
					++c_blockBuf;
					--t_t_mat;
				}
			}
			float *t_t_mat = t_mat;

			offset = ind_y_end - ind_y;
			memcpy(c_blockBuf, t_t_mat, offset * sizeof(float));

			c_blockBuf += offset;

			if (ind_y_end_after)
			{
				t_t_mat += offset - 1;

				for (size_t j = 0; j < ind_y_end_after; ++j)
				{
					*c_blockBuf = *t_t_mat;

					++c_blockBuf;
					--t_t_mat;
				}
			}

			t_mat -= mat_N;
		}
	}

	float *t_mat = c_mat;
	for (size_t i = ind_x; i < ind_x_end; ++i)
	{
		if (minus_ind_y)
		{
			float *t_t_mat = t_mat + minus_ind_y - 1;
			for (size_t j = 0; j < minus_ind_y; ++j)
			{
				*c_blockBuf = *t_t_mat;
				++c_blockBuf;
				--t_t_mat;
			}
		}
		float *t_t_mat = t_mat;

		offset = ind_y_end - ind_y;
		memcpy(c_blockBuf, t_t_mat, offset * sizeof(float));

		c_blockBuf += offset;

		if (ind_y_end_after)
		{
			t_t_mat += offset - 1;

			for (size_t j = 0; j < ind_y_end_after; ++j)
			{
				*c_blockBuf = *t_t_mat;

				++c_blockBuf;
				--t_t_mat;
			}
		}

		t_mat += mat_N;
	}

	if (ind_x_end_after)
	{
		t_mat -= mat_N;

		for (size_t i = 0; i < ind_x_end_after; ++i)
		{
			if (minus_ind_y)
			{
				float *t_t_mat = t_mat + minus_ind_y - 1;
				for (size_t j = 0; j < minus_ind_y; ++j)
				{
					*c_blockBuf = *t_t_mat;
					++c_blockBuf;
					--t_t_mat;
				}
			}
			float *t_t_mat = t_mat;

			offset = ind_y_end - ind_y;
			memcpy(c_blockBuf, t_t_mat, offset * sizeof(float));

			c_blockBuf += offset;

			if (ind_y_end_after)
			{
				t_t_mat += offset - 1;

				for (size_t j = 0; j < ind_y_end_after; ++j)
				{
					*c_blockBuf = *t_t_mat;

					++c_blockBuf;
					--t_t_mat;
				}
			}

			t_mat -= mat_N;
		}
	}
}

void determineIndexPreMat(int64_t index, int64_t mat_length, int64_t block_length, int64_t &index_pre_begin, int64_t &index_pre_end)
{
	if (index >= 0) {
		index_pre_begin = 0;
		index_pre_end = 0;
		return;
	}

	index_pre_begin = -index;
	if (index + (int64_t)block_length < 0) {
		index_pre_end = -(index + (int64_t)block_length);
	}
	else {
		index_pre_end = 0;
	}
}

void determineIndexInMat(int64_t index, int64_t mat_length, int64_t block_length, int64_t &index_begin, int64_t &index_end)
{
	if (index >= mat_length)
	{
		index_begin = 0;
		index_end = 0;
		return;
	}

	if (index < 0)
		index_begin = 0;
	else
		index_begin = index;

	if (index + block_length < 0)
	{
		index_end = 0;
	}
	else if (index + block_length < mat_length)
	{
		index_end = index + block_length;
	}
	else
	{
		index_end = mat_length;
	}
}

void determinIndexPostMat(int64_t index, int64_t mat_length, int64_t block_length, int64_t &index_post_begin, int64_t &index_post_end)
{
	if (index < mat_length)
		index_post_begin = 0;
	else
		index_post_begin = index - mat_length;

	if (index + block_length < mat_length)
		index_post_end = 0;
	else
		index_post_end = index + block_length - mat_length;
}

void determineIndex(int64_t index, int64_t mat_length, int64_t block_length, int64_t &index_pre_begin, int64_t &index_pre_end, int64_t &index_begin, int64_t &index_end, int64_t &index_post_begin, int64_t &index_post_end)
{
	determineIndexPreMat(index, mat_length, block_length, index_pre_begin, index_pre_end);
	determineIndexInMat(index, mat_length, block_length, index_begin, index_end);
	determinIndexPostMat(index, mat_length, block_length, index_post_begin, index_post_end);
}

void copyPatchWithSymmetricPaddding(float *buf, float *src, int64_t mat_M, int64_t mat_N, int64_t index_x, int64_t index_y, int64_t block_M, int64_t block_N)
{
	if (index_x >= 0 && index_y >= 0 && index_x + block_M < mat_M && index_y + block_N < mat_N)
	{
		copyPatch(buf, src, mat_M, mat_N, index_x, index_y, block_M, block_N);
		return;
	}

	int64_t x_index_pre_begin, x_index_pre_end, x_index_begin, x_index_end, x_index_post_begin, x_index_post_end;
	int64_t y_index_pre_begin, y_index_pre_end, y_index_begin, y_index_end, y_index_post_begin, y_index_post_end;

	determineIndex(index_x, mat_M, block_M, x_index_pre_begin, x_index_pre_end, x_index_begin, x_index_end, x_index_post_begin, x_index_post_end);
	determineIndex(index_y, mat_N, block_N, y_index_pre_begin, y_index_pre_end, y_index_begin, y_index_end, y_index_post_begin, y_index_post_end);

	for (int64_t i = x_index_pre_begin; i>x_index_pre_end; --i)
	{
		float *c_mat = src + (i - 1) * mat_N;
		for (int64_t j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			float *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(float));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_N - 1;
		for (int64_t j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			float *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}

	for (int64_t i = x_index_begin; i<x_index_end; ++i)
	{
		float *c_mat = src + i * mat_N;
		for (int64_t j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			float *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(float));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_N - 1;
		for (int64_t j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			float *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}

	for (int64_t i = x_index_post_begin; i<x_index_post_end; ++i)
	{
		float *c_mat = src + (mat_M - i - 1) * mat_N;
		for (int64_t j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			float *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(float));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_N - 1;
		for (int64_t j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			float *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}
}

struct Context_Async
{
	Type type;

	size_t matA_M;
	size_t matA_N;
	size_t matB_M;
	size_t matB_N;
	size_t block_M;
	size_t block_N;

	size_t neighbour_M;
	size_t neighbour_N;
	size_t stride_M;
	size_t stride_N;

	int numDeviceMultiProcessor;
	int numProcessorThread;

	float *buffer_A;
	float *buffer_B;
	float *result_buffer;
	float *device_buffer_A;
	float *device_buffer_B;
	float *device_result_buffer;

	size_t result_dim0;
	size_t result_dim1;
	size_t result_dim2;
	size_t result_dim3;

	cudaStream_t stream[numSubmitThread];
	thread_pool *pool;
};


bool initialize_async_submit(void **_instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N, size_t neighbour_M, size_t neighbour_N, size_t stride_M, size_t stride_N)
{
	struct Context_Async * instance = (struct Context_Async *)malloc(sizeof(struct Context_Async));
	if (!instance)
		return false;

	instance->type = COMBILE;

	instance->matA_M = matA_M;
	instance->matA_N = matA_N;
	instance->matB_M = matB_M;
	instance->matB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->neighbour_M = neighbour_M;
	instance->neighbour_N = neighbour_N;

	instance->stride_M = stride_M;
	instance->stride_N = stride_N;

	size_t result_dim0 = matA_M - block_M + 1;
	size_t result_dim1 = matA_N - block_N + 1;
	size_t result_dim2 = (neighbour_M + stride_M - 1) / stride_M;
	size_t result_dim3 = (neighbour_N + stride_N - 1) / stride_N;
	instance->result_dim0 = result_dim0;
	instance->result_dim1 = result_dim1;
	instance->result_dim2 = result_dim2;
	instance->result_dim3 = result_dim3;

	int numDeviceMultiProcessor;
	const int numProcessorThread = 512;

	cudaError_t cuda_error = cudaDeviceGetAttribute(&numDeviceMultiProcessor, cudaDevAttrMultiProcessorCount, 0);
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	instance->numDeviceMultiProcessor = numDeviceMultiProcessor;
	instance->numProcessorThread = numProcessorThread;

	instance->pool = pool;

	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
			return false;
	}

	cuda_error = cudaMallocHost(&instance->buffer_A, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->buffer_B, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer, result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * sizeof(float));

	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_B;
	}
	*_instance = instance;

	return true;

release_device_buffer_B:

	cudaFree(instance->device_buffer_B);
release_device_buffer_A:

	cudaFree(instance->device_buffer_A);
release_result_buffer:

	cudaFreeHost(instance->result_buffer);
release_buffer_B:

	cudaFreeHost(instance->buffer_B);
release_buffer_A:

	cudaFreeHost(instance->buffer_A);
release_instance:

	free(instance);
	return false;
}
bool initialize_async_submit_bak(void **_instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N, size_t neighbour_M, size_t neighbour_N, size_t stride_M, size_t stride_N)
{
	struct Context_Async * instance = (struct Context_Async *)malloc(sizeof(struct Context_Async));
	if (!instance)
		return false;

	instance->type = COMBILE;

	instance->matA_M = matA_M;
	instance->matA_N = matA_N;
	instance->matB_M = matB_M;
	instance->matB_N = matB_N;

	instance->block_M = block_M;
	instance->block_N = block_N;

	instance->neighbour_M = neighbour_M;
	instance->neighbour_N = neighbour_N;

	instance->stride_M = stride_M;
	instance->stride_N = stride_N;

	size_t result_dim0 = matA_M - block_M - neighbour_M + 2;
	size_t result_dim1 = matA_N - block_N - neighbour_N + 2;
	size_t result_dim2 = (neighbour_M + stride_M - 1) / stride_M;
	size_t result_dim3 = (neighbour_N + stride_N - 1) / stride_N;
	instance->result_dim0 = result_dim0;
	instance->result_dim1 = result_dim1;
	instance->result_dim2 = result_dim2;
	instance->result_dim3 = result_dim3;

	int numDeviceMultiProcessor;
	const int numProcessorThread = 512;

	cudaError_t cuda_error = cudaDeviceGetAttribute(&numDeviceMultiProcessor, cudaDevAttrMultiProcessorCount, 0);
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	instance->numDeviceMultiProcessor = numDeviceMultiProcessor;
	instance->numProcessorThread = numProcessorThread;

	instance->pool = pool;

	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		cuda_error = cudaStreamCreate(&instance->stream[i]);
		if (cuda_error != cudaSuccess)
			return false;
	}

	cuda_error = cudaMallocHost(&instance->buffer_A, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_instance;
	}

	cuda_error = cudaMallocHost(&instance->buffer_B, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_A;
	}

	cuda_error = cudaMallocHost(&instance->result_buffer, result_dim0 * result_dim1 * result_dim2 * result_dim3 * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_buffer_B;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_A, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_result_buffer;
	}

	cuda_error = cudaMalloc(&instance->device_buffer_B, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * block_M * block_N * sizeof(float));
	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_A;
	}
	cuda_error = cudaMalloc(&instance->device_result_buffer, numSubmitThread * numDeviceMultiProcessor * numProcessorThread * sizeof(float));

	if (cuda_error != cudaSuccess)
	{
		goto release_device_buffer_B;
	}
	*_instance = instance;

	return true;

release_device_buffer_B:

	cudaFree(instance->device_buffer_B);
release_device_buffer_A:

	cudaFree(instance->device_buffer_A);
release_result_buffer:

	cudaFreeHost(instance->result_buffer);
release_buffer_B:

	cudaFreeHost(instance->buffer_B);
release_buffer_A:

	cudaFreeHost(instance->buffer_A);
release_instance:

	free(instance);
	return false;
}
bool processWorker_bak(float *matA, float *matB, float *result,
	float *bufferA, size_t matA_M, size_t matA_N, size_t index_A_M_begin, size_t index_A_M_end, size_t index_A_N_begin, size_t index_A_N_end,
	float *bufferB, size_t matB_M, size_t matB_N,
	size_t block_M, size_t block_N,
	size_t neighbour_M, size_t neighbour_N,
	size_t stride_M, size_t stride_N,
	float *device_bufferA, float *device_bufferB, float *device_bufferC,
	cudaStream_t stream, int numDeviceMultiProcessor, int numThreadsPerProcessor, Method method)
{
	cudaError_t(*processFunction)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
	cudaError_t(*processFunction_borderCheck)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_blockSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream);
	if (method == MSE)
	{
		processFunction = block_match_mse_async;
		processFunction_borderCheck = block_match_mse_async;
	}
	else
	{
		processFunction = block_match_cc_async;
		processFunction_borderCheck = block_match_cc_async;
	}

	size_t blockSize = block_M * block_N;
	size_t neighbourSize = neighbour_M / stride_M * neighbour_N / stride_N;

	float *c_bufferA = bufferA;
	float *c_bufferB = bufferB;
	float *c_result = result;

	size_t neighbour_M_m = neighbour_M / 2;
	size_t neighbour_N_m = neighbour_N / 2;

	int blocksPerProcessor = 0;
	int filledProcessor = 0;
	size_t numTasks = 0;
	size_t numBlocks_A = 0, numBlocks_B = 0;

	for (size_t ind_A_M = index_A_M_begin; ind_A_M < index_A_M_end; ++ind_A_M)
	{
		for (size_t ind_A_N = index_A_N_begin; ind_A_N < index_A_N_end; ++ind_A_N)
		{
			copyPatch(c_bufferA, matA, matA_M, matA_N, ind_A_M, ind_A_N, block_M, block_N);

			for (size_t ind_neighbour_M = 0; ind_neighbour_M < neighbour_M; ind_neighbour_M += stride_M)
			{
				size_t index_x = ind_A_M + ind_neighbour_M - neighbour_M_m;

				for (size_t ind_neighbour_N = 0; ind_neighbour_N < neighbour_N; ind_neighbour_N += stride_N)
				{
					size_t index_y = ind_A_N + ind_neighbour_N - neighbour_N_m;

					copyPatch(c_bufferB, matB, matB_M, matB_N, index_x, index_y, block_M, block_N);

					c_bufferB += blockSize;
				}
			}

			numBlocks_B += neighbourSize;
			numBlocks_A += 1;

			numTasks += neighbourSize;

			blocksPerProcessor += neighbourSize;
			c_bufferA += blockSize;

			if (blocksPerProcessor + neighbourSize > numThreadsPerProcessor)
			{
				filledProcessor++;

				if (filledProcessor == numDeviceMultiProcessor)
				{
					cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					if (processFunction(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, filledProcessor, blocksPerProcessor, stream) != cudaSuccess)
						return false;

					cuda_error = cudaStreamSynchronize(stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
					if (cuda_error != cudaSuccess)
						return false;

					c_result += numTasks;
					c_bufferA = bufferA;
					c_bufferB = bufferB;

					numBlocks_A = 0;
					numBlocks_B = 0;
					numTasks = 0;
					filledProcessor = 0;
				}
				blocksPerProcessor = 0;
			}
		}
	}

	if (numTasks)
	{
		cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		if (!processFunction_borderCheck(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, (numTasks + numThreadsPerProcessor - 1) / numThreadsPerProcessor, numThreadsPerProcessor, numTasks, stream) == cudaSuccess)
			return false;

		cuda_error = cudaStreamSynchronize(stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
		if (cuda_error != cudaSuccess)
			return false;
	}

	return true;
}

bool processWorker(float *matA, float *matB, float *result,
	float *bufferA, size_t matA_M, size_t matA_N, size_t index_A_M_begin, size_t index_A_M_end, size_t index_A_N_begin, size_t index_A_N_end,
	float *bufferB, size_t matB_M, size_t matB_N,
	size_t block_M, size_t block_N,
	size_t neighbour_M, size_t neighbour_N,
	size_t stride_M, size_t stride_N,
	float *device_bufferA, float *device_bufferB, float *device_bufferC,
	cudaStream_t stream, int numDeviceMultiProcessor, int numThreadsPerProcessor, Method method)
{
	cudaError_t(*processFunction)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
	cudaError_t(*processFunction_borderCheck)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_blockSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream);
	if (method == MSE)
	{
		processFunction = block_match_mse_async;
		processFunction_borderCheck = block_match_mse_async;
	}
	else
	{
		processFunction = block_match_cc_async;
		processFunction_borderCheck = block_match_cc_async;
	}

	size_t blockSize = block_M * block_N;
	size_t neighbourSize = neighbour_M / stride_M * neighbour_N / stride_N;

	float *c_bufferA = bufferA;
	float *c_bufferB = bufferB;
	float *c_result = result;

	size_t neighbour_M_m = neighbour_M / 2;
	size_t neighbour_N_m = neighbour_N / 2;

	int blocksPerProcessor = 0;
	int filledProcessor = 0;
	size_t numTasks = 0;
	size_t numBlocks_A = 0, numBlocks_B = 0;

	for (size_t ind_A_M = index_A_M_begin; ind_A_M < index_A_M_end; ++ind_A_M)
	{
		for (size_t ind_A_N = index_A_N_begin; ind_A_N < index_A_N_end; ++ind_A_N)
		{
			copyPatch(c_bufferA, matA, matA_M, matA_N, ind_A_M, ind_A_N, block_M, block_N);

			for (size_t ind_neighbour_M = 0; ind_neighbour_M < neighbour_M; ind_neighbour_M += stride_M)
			{
				int64_t index_x = int64_t(ind_A_M + ind_neighbour_M) - neighbour_M_m;

				for (size_t ind_neighbour_N = 0; ind_neighbour_N < neighbour_N; ind_neighbour_N += stride_N)
				{
					int64_t index_y = int64_t(ind_A_N + ind_neighbour_N) - neighbour_N_m;

					copyPatchWithSymmetricPaddding(c_bufferB, matB, matB_M, matB_N, index_x, index_y, block_M, block_N);

					c_bufferB += blockSize;
				}
			}

			numBlocks_B += neighbourSize;
			numBlocks_A += 1;

			numTasks += neighbourSize;

			blocksPerProcessor += neighbourSize;
			c_bufferA += blockSize;

			if (blocksPerProcessor + neighbourSize > numThreadsPerProcessor)
			{
				filledProcessor++;

				if (filledProcessor == numDeviceMultiProcessor)
				{
					cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					if (processFunction(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, filledProcessor, blocksPerProcessor, stream) != cudaSuccess)
						return false;

					cuda_error = cudaStreamSynchronize(stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
					if (cuda_error != cudaSuccess)
						return false;

					c_result += numTasks;
					c_bufferA = bufferA;
					c_bufferB = bufferB;

					numBlocks_A = 0;
					numBlocks_B = 0;
					numTasks = 0;
					filledProcessor = 0;
				}
				blocksPerProcessor = 0;
			}
		}
	}

	if (numTasks)
	{
		cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		if (processFunction_borderCheck(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, (numTasks + numThreadsPerProcessor - 1) / numThreadsPerProcessor, numThreadsPerProcessor, numTasks, stream) != cudaSuccess)
			return false;

		cuda_error = cudaStreamSynchronize(stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
		if (cuda_error != cudaSuccess)
			return false;
	}

	return true;
}

bool processWorker_sort(float *matA, float *matB, float *result,
	float *bufferA, size_t matA_M, size_t matA_N, size_t index_A_M_begin, size_t index_A_M_end, size_t index_A_N_begin, size_t index_A_N_end,
	float *bufferB, size_t matB_M, size_t matB_N,
	size_t block_M, size_t block_N,
	size_t neighbour_M, size_t neighbour_N,
	size_t stride_M, size_t stride_N,
	float *device_bufferA, float *device_bufferB, float *device_bufferC,
	cudaStream_t stream, int numDeviceMultiProcessor, int numThreadsPerProcessor, Method method)
{
	cudaError_t(*processFunction)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
	cudaError_t(*processFunction_borderCheck)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_blockSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream);
	if (method == MSE)
	{
		processFunction = block_match_mse_async;
		processFunction_borderCheck = block_match_mse_async;
	}
	else
	{
		processFunction = block_match_cc_async;
		processFunction_borderCheck = block_match_cc_async;
	}

	size_t blockSize = block_M * block_N;
	size_t neighbourSize = neighbour_M / stride_M * neighbour_N / stride_N;

	float *c_bufferA = bufferA;
	float *c_bufferB = bufferB;
	float *c_result = result;

	size_t neighbour_M_m = neighbour_M / 2;
	size_t neighbour_N_m = neighbour_N / 2;

	int blocksPerProcessor = 0;
	int filledProcessor = 0;
	size_t numTasks = 0;
	size_t numBlocks_A = 0, numBlocks_B = 0;

	for (size_t ind_A_M = index_A_M_begin; ind_A_M < index_A_M_end; ++ind_A_M)
	{
		for (size_t ind_A_N = index_A_N_begin; ind_A_N < index_A_N_end; ++ind_A_N)
		{
			copyPatch(c_bufferA, matA, matA_M, matA_N, ind_A_M, ind_A_N, block_M, block_N);

			for (size_t ind_neighbour_M = 0; ind_neighbour_M < neighbour_M; ind_neighbour_M += stride_M)
			{
				size_t index_x = ind_A_M + ind_neighbour_M - neighbour_M_m;

				for (size_t ind_neighbour_N = 0; ind_neighbour_N < neighbour_N; ind_neighbour_N += stride_N)
				{
					size_t index_y = ind_A_N + ind_neighbour_N - neighbour_N_m;

					copyPatch(c_bufferB, matB, matB_M, matB_N, index_x, index_y, block_M, block_N);

					c_bufferB += blockSize;
				}
			}

			numBlocks_B += neighbourSize;
			numBlocks_A += 1;

			numTasks += neighbourSize;

			blocksPerProcessor += neighbourSize;
			c_bufferA += blockSize;

			if (blocksPerProcessor + neighbourSize > numThreadsPerProcessor)
			{
				filledProcessor++;

				if (filledProcessor == numDeviceMultiProcessor)
				{
					cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					if (processFunction(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, filledProcessor, blocksPerProcessor, stream) != cudaSuccess)
						return false;

					cuda_error = cudaStreamSynchronize(stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
					if (cuda_error != cudaSuccess)
						return false;

					c_result += numTasks;
					c_bufferA = bufferA;
					c_bufferB = bufferB;

					numBlocks_A = 0;
					numBlocks_B = 0;
					numTasks = 0;
					filledProcessor = 0;
				}
				blocksPerProcessor = 0;
			}
		}
	}

	if (numTasks)
	{
		cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		if (!processFunction_borderCheck(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, (numTasks + numThreadsPerProcessor - 1) / numThreadsPerProcessor, numThreadsPerProcessor, numTasks, stream) == cudaSuccess)
			return false;

		cuda_error = cudaStreamSynchronize(stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(c_result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
		if (cuda_error != cudaSuccess)
			return false;
	}

	return true;
}

bool processWorker_paddingB(float *matA, float *matB, float *result,
	float *bufferA, size_t matA_M, size_t matA_N, size_t begin_ind, size_t end_ind, size_t ind_A_N_end,
	float *bufferB, size_t matB_M, size_t matB_N,
	size_t block_M, size_t block_N,
	size_t neighbour_M, size_t neighbour_N,
	float *device_bufferA, float *device_bufferB, float *device_bufferC,
	cudaStream_t stream, int numDeviceMultiProcessor, int numThreadsPerProcessor, Method method)
{
	cudaError_t(*processFunction)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
	cudaError_t(*processFunction_borderCheck)(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_blockSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream);
	if (method == MSE)
	{
		processFunction = block_match_mse_async;
		processFunction_borderCheck = block_match_mse_async;
	}
	else
	{
		processFunction = block_match_cc_async;
		processFunction_borderCheck = block_match_cc_async;
	}

	size_t blockSize = block_M * block_N;
	size_t neighbourSize = neighbour_M * neighbour_N;

	float *c_bufferA = bufferA;
	float *c_bufferB = bufferB;

	size_t neighbour_M_m = neighbour_M / 2;
	size_t neighbour_N_m = neighbour_N / 2;

	int blocksPerProcessor = 0;
	int filledProcessor = 0;
	size_t numTasks = 0;
	size_t numBlocks_A = 0, numBlocks_B = 0;

	for (size_t ind_A_M = begin_ind; ind_A_M < end_ind; ++ind_A_M)
	{
		for (size_t ind_A_N = 0; ind_A_N < ind_A_N_end; ++ind_A_N)
		{
			copyPatch(c_bufferA, matA, matA_M, matA_N, ind_A_M, ind_A_N, block_M, block_N);

			for (size_t ind_neighbour_M = 0; ind_neighbour_M < neighbour_M; ++ind_neighbour_M)
			{
				size_t minus_ind_x, ind_x;
				if (ind_A_M + ind_neighbour_M < neighbour_M_m)
				{
					minus_ind_x = neighbour_M_m - ind_A_M - ind_neighbour_M;
					ind_x = 0;
				}
				else
				{
					minus_ind_x = 0;
					ind_x = ind_A_M + ind_neighbour_M - neighbour_M_m;
				}
				for (size_t ind_neighbour_N = 0; ind_neighbour_N < neighbour_N; ++ind_neighbour_N)
				{
					size_t minus_ind_y, ind_y;
					if (ind_A_N + ind_neighbour_N < neighbour_N_m)
					{
						minus_ind_y = neighbour_N_m - ind_A_N - ind_neighbour_N;
						ind_y = 0;
					}
					else
					{
						minus_ind_y = 0;
						ind_y = ind_A_N + ind_neighbour_N - neighbour_N_m;
					}

					copyPatchWithSymmetricPaddding(c_bufferB, matB, matB_M, matB_N, ind_x, minus_ind_x, ind_y, minus_ind_y, block_M, block_N);

					c_bufferB += blockSize;
				}
			}

			numBlocks_B += neighbourSize;
			numBlocks_A += 1;

			numTasks += neighbourSize;

			blocksPerProcessor += neighbourSize;
			c_bufferA += blockSize;

			if (blocksPerProcessor + neighbourSize > numThreadsPerProcessor)
			{
				filledProcessor++;

				if (filledProcessor == numDeviceMultiProcessor)
				{
					cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
					if (cuda_error != cudaSuccess)
						return false;

					if (processFunction(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, filledProcessor, blocksPerProcessor, stream) != cudaSuccess)
						return false;

					cuda_error = cudaStreamSynchronize(stream);
					if (cuda_error != cudaSuccess)
						return false;

					cuda_error = cudaMemcpyAsync(result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
					if (cuda_error != cudaSuccess)
						return false;

					result += numTasks;
					c_bufferA = bufferA;
					c_bufferB = bufferB;

					numBlocks_A = 0;
					numBlocks_B = 0;
					numTasks = 0;
					filledProcessor = 0;
				}
				blocksPerProcessor = 0;
			}

		}
	}

	if (numTasks)
	{
		cudaError_t cuda_error = cudaMemcpyAsync(device_bufferA, bufferA, numBlocks_A * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(device_bufferB, bufferB, numBlocks_B * blockSize * sizeof(float), cudaMemcpyHostToDevice, stream);
		if (cuda_error != cudaSuccess)
			return false;

		if (!processFunction_borderCheck(device_bufferA, device_bufferB, numBlocks_A, numBlocks_B, neighbourSize, blockSize, device_bufferC, (numTasks + numThreadsPerProcessor - 1) / numThreadsPerProcessor, numThreadsPerProcessor, numTasks, stream) == cudaSuccess)
			return false;

		cuda_error = cudaStreamSynchronize(stream);
		if (cuda_error != cudaSuccess)
			return false;

		cuda_error = cudaMemcpyAsync(result, device_bufferC, numTasks * sizeof(float), cudaMemcpyDeviceToHost, stream);
		if (cuda_error != cudaSuccess)
			return false;
	}

	return true;
}

unsigned processWorkerProxy(void *para)
{
	auto para_tuple =
		(std::tuple<float *, float *, float *,
			float *, size_t, size_t, size_t, size_t, size_t, size_t,
			float *, size_t, size_t,
			size_t, size_t,
			size_t, size_t,
			size_t, size_t,
			float *, float *, float *,
			cudaStream_t, int, int, Method > *)para;
	return processWorker(std::get<0>(*para_tuple), std::get<1>(*para_tuple), std::get<2>(*para_tuple), std::get<3>(*para_tuple), std::get<4>(*para_tuple), std::get<5>(*para_tuple), std::get<6>(*para_tuple), std::get<7>(*para_tuple), std::get<8>(*para_tuple), std::get<9>(*para_tuple), std::get<10>(*para_tuple), std::get<11>(*para_tuple), std::get<12>(*para_tuple), std::get<13>(*para_tuple), std::get<14>(*para_tuple), std::get<15>(*para_tuple), std::get<16>(*para_tuple), std::get<17>(*para_tuple), std::get<18>(*para_tuple), std::get<19>(*para_tuple), std::get<20>(*para_tuple), std::get<21>(*para_tuple), std::get<22>(*para_tuple), std::get<23>(*para_tuple), std::get<24>(*para_tuple), std::get<25>(*para_tuple))
		? 0 : 1;
}


bool process_async_submit(void *_instance, float *matA, float *matB, enum Method method)
{
	struct Context_Async *instance = (struct Context_Async *)_instance;
	thread_pool &pool = *instance->pool;

	void *task_handle[numSubmitThread];

	float *result_buffer = instance->result_buffer,
		*bufferA = instance->buffer_A,
		*bufferB = instance->buffer_B,
		*device_bufferA = instance->device_buffer_A,
		*device_bufferB = instance->device_buffer_B,
		*device_bufferC = instance->device_result_buffer;

	size_t matA_M = instance->matA_M,
		matA_N = instance->matA_N,
		matB_M = instance->matB_M,
		matB_N = instance->matB_N,
		block_M = instance->block_M,
		block_N = instance->block_N,
		neighbour_M = instance->neighbour_M,
		neighbour_N = instance->neighbour_N,
		stride_M = instance->stride_M,
		stride_N = instance->stride_N,
		result_dim0 = instance->result_dim0,
		result_dim1 = instance->result_dim1,
		result_dim2 = instance->result_dim2,
		result_dim3 = instance->result_dim3;

	int numDeviceMultiProcessor = instance->numDeviceMultiProcessor,
		numThreadsPerProcessor = instance->numProcessorThread;
	size_t ind_A_N_begin = 0;
	size_t ind_A_M_end = result_dim0;
	size_t ind_A_N_end = result_dim1;
	
	std::tuple<float *, float *, float *,
		float *, size_t, size_t, size_t, size_t, size_t, size_t,
		float *, size_t, size_t,
		size_t, size_t,
		size_t, size_t,
		size_t, size_t,
		float *, float *, float *,
		cudaStream_t, int, int, Method > para_tuple[numSubmitThread];
	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		size_t c_index_A_M_begin = (ind_A_M_end) / numSubmitThread * i;
		size_t c_index_A_M_end;
		if (i + 1 != numSubmitThread)
		{
			c_index_A_M_end = (ind_A_M_end) / numSubmitThread * (i + 1);
		}
		else
		{
			c_index_A_M_end = ind_A_M_end;
		}

		float *c_buffer_A = bufferA + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_buffer_B = bufferB + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_buffer_result = result_buffer + (c_index_A_M_begin) * result_dim1 * result_dim2 * result_dim3;
		float *c_device_buffer_A = device_bufferA + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_device_buffer_B = device_bufferB + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_device_buffer_C = device_bufferC + i * numDeviceMultiProcessor * numThreadsPerProcessor;

		cudaStream_t stream = instance->stream[i];

		para_tuple[i] =
			std::make_tuple(matA, matB, c_buffer_result,
				c_buffer_A, matA_M, matA_N, c_index_A_M_begin, c_index_A_M_end, ind_A_N_begin, ind_A_N_end,
				c_buffer_B, matB_M, matB_N,
				block_M, block_N,
				neighbour_M, neighbour_N,
				stride_M, stride_N,
				c_device_buffer_A, c_device_buffer_B, c_device_buffer_C,
				stream, numDeviceMultiProcessor, numThreadsPerProcessor, method);

		task_handle[i] = thread_pool_launcher(pool, processWorker, para_tuple[i]);
	}
	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		pool.join(task_handle[i]);
	}
	bool isFailed = false;
	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		if (pool.get_rc(task_handle[i]) != 0)
			isFailed = true;
		pool.release(task_handle[i]);
	}

	return !isFailed;
}

bool process_async_submit_bak(void *_instance, float *matA, float *matB, enum Method method)
{
	struct Context_Async *instance = (struct Context_Async *)_instance;
	thread_pool &pool = *instance->pool;

	void *task_handle[numSubmitThread];

	float *result_buffer = instance->result_buffer,
		*bufferA = instance->buffer_A,
		*bufferB = instance->buffer_B,
		*device_bufferA = instance->device_buffer_A,
		*device_bufferB = instance->device_buffer_B,
		*device_bufferC = instance->device_result_buffer;

	size_t matA_M = instance->matA_M,
		matA_N = instance->matA_N,
		matB_M = instance->matB_M,
		matB_N = instance->matB_N,
		block_M = instance->block_M,
		block_N = instance->block_N,
		neighbour_M = instance->neighbour_M,
		neighbour_N = instance->neighbour_N,
		stride_M = instance->stride_M,
		stride_N = instance->stride_N,
		result_dim1 = instance->result_dim1,
		result_dim2 = instance->result_dim2,
		result_dim3 = instance->result_dim3;

	int numDeviceMultiProcessor = instance->numDeviceMultiProcessor,
		numThreadsPerProcessor = instance->numProcessorThread;

	size_t ind_A_M_end = matA_M - block_M + 1;
	size_t ind_A_N_end = matA_N - block_N + 1;

	size_t ind_A_M_begin = neighbour_M / 2;
	size_t ind_A_N_begin = neighbour_N / 2;
	ind_A_M_end -= neighbour_M - ind_A_M_begin - 1;
	ind_A_N_end -= neighbour_N - ind_A_N_begin - 1;
	
	std::tuple<float *, float *, float *,
		float *, size_t, size_t, size_t, size_t, size_t, size_t,
		float *, size_t, size_t,
		size_t, size_t,
		size_t, size_t,
		size_t, size_t,
		float *, float *, float *,
		cudaStream_t, int, int, Method > para_tuple[numSubmitThread];
	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		size_t c_index_A_M_begin = (ind_A_M_end - ind_A_M_begin) / numSubmitThread * i + ind_A_M_begin;
		size_t c_index_A_M_end;
		if (i + 1 != numSubmitThread)
		{
			c_index_A_M_end = (ind_A_M_end - ind_A_M_begin) / numSubmitThread * (i + 1) + ind_A_M_begin;
		}
		else
		{
			c_index_A_M_end = ind_A_M_end;
		}

		float *c_buffer_A = bufferA + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_buffer_B = bufferB + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_buffer_result = result_buffer + (c_index_A_M_begin - ind_A_M_begin) * result_dim1 * result_dim2 * result_dim3;
		float *c_device_buffer_A = device_bufferA + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_device_buffer_B = device_bufferB + i * numDeviceMultiProcessor * numThreadsPerProcessor * block_M * block_N;
		float *c_device_buffer_C = device_bufferC + i * numDeviceMultiProcessor * numThreadsPerProcessor;

		cudaStream_t stream = instance->stream[i];

		para_tuple[i] =
			std::make_tuple(matA, matB, c_buffer_result,
				c_buffer_A, matA_M, matA_N, c_index_A_M_begin, c_index_A_M_end, ind_A_N_begin, ind_A_N_end,
				c_buffer_B, matB_M, matB_N,
				block_M, block_N,
				neighbour_M, neighbour_N,
				stride_M, stride_N,
				c_device_buffer_A, c_device_buffer_B, c_device_buffer_C,
				stream, numDeviceMultiProcessor, numThreadsPerProcessor, method);

		task_handle[i] = pool.submit(processWorkerProxy, &para_tuple[i]);
	}
	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		pool.join(task_handle[i]);
	}
	bool isFailed = false;
	for (uint32_t i = 0; i < numSubmitThread; ++i)
	{
		if (pool.get_rc(task_handle[i]) != 0)
			isFailed = true;
		pool.release(task_handle[i]);
	}

	return !isFailed;
}

bool process_TypeA(void *_instance, float *matA, float *matB, enum Method method)
{
	struct Context *instance = (struct Context *)_instance;

	size_t matA_M = instance->matA_M;
	size_t matA_N = instance->matA_N;
	size_t matB_M = instance->matB_M;
	size_t matB_N = instance->matB_N;
	size_t block_M = instance->block_M;
	size_t block_N = instance->block_N;

	float *buffer_A = instance->buffer_A;
	float *buffer_B = instance->buffer_B;
	float *c_buffer_A = buffer_A;
	float *c_buffer_B = buffer_B;

	float *result_buffer = instance->result_buffer;

	float *device_buffer_A = instance->device_buffer_A;
	float *device_buffer_B = instance->device_buffer_B;
	float *device_result_buffer = instance->device_result_buffer;

	size_t ind_A_M_end = matA_M - block_M;
	size_t ind_A_N_end = matA_N - block_N;
	size_t ind_B_M_end = matB_M - block_M;
	size_t ind_B_N_end = matB_N - block_N;

	size_t blockSize = block_M * block_N;

	for (size_t ind_A_M = 0; ind_A_M < ind_A_M_end; ++ind_A_M)
	{
		for (size_t ind_A_N = 0; ind_A_N < ind_A_N_end; ++ind_A_N)
		{
			copyPatch(c_buffer_A, matA, matA_M, matA_N, ind_A_M, ind_A_N, block_M, block_N);

			c_buffer_A += blockSize;
		}
	}

	for (size_t ind_B_M = 0; ind_B_M < ind_B_M_end; ++ind_B_M)
	{
		for (size_t ind_B_N = 0; ind_B_N < ind_B_N_end; ++ind_B_N)
		{
			copyPatch(c_buffer_B, matB, matB_M, matB_N, ind_B_M, ind_B_N, block_M, block_N);

			c_buffer_B += blockSize;
		}
	}

	cudaError_t cuda_error = cudaMemcpyAsync(device_buffer_A, buffer_A, ind_A_M_end * ind_A_N_end * blockSize * sizeof(float), cudaMemcpyHostToDevice, cudaStreamDefault);
	if (cuda_error != cudaSuccess)
		return false;

	cuda_error = cudaMemcpyAsync(device_buffer_B, buffer_B, ind_B_M_end * ind_B_N_end * blockSize * sizeof(float), cudaMemcpyHostToDevice, cudaStreamDefault);
	if (cuda_error != cudaSuccess)
		return false;

	cuda_error = cudaStreamSynchronize(cudaStreamDefault);
	if (cuda_error != cudaSuccess)
		return false;

	if (method == MSE)
		cuda_error = block_match_mse(device_buffer_A, device_buffer_B, ind_A_M_end * ind_A_N_end, ind_B_M_end * ind_B_N_end, blockSize, device_result_buffer, cudaStreamDefault);
	else if (method == CC)
		cuda_error = block_match_cc(device_buffer_A, device_buffer_B, ind_A_M_end * ind_A_N_end, ind_B_M_end * ind_B_N_end, blockSize, device_result_buffer, cudaStreamDefault);
	else
		abort();

	if (cuda_error != cudaSuccess)
		return false;

	cuda_error = cudaMemcpyAsync(result_buffer, device_result_buffer, ind_A_M_end * ind_A_N_end * ind_B_M_end * ind_B_N_end * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamDefault);
	if (cuda_error != cudaSuccess)
		return false;

	instance->result_dim0 = ind_A_M_end;
	instance->result_dim1 = ind_A_N_end;
	instance->result_dim2 = ind_B_M_end;
	instance->result_dim3 = ind_B_N_end;

	return true;
}

extern "C"
bool initialize(void **_instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N, size_t neighbour_M, size_t neighbour_N, size_t stride_M, size_t stride_N)
{
	if (neighbour_M == 0)
	{
		return initialize_TypeA(_instance, matA_M, matA_N, matB_M, matB_N, block_M, block_N);
	}
	else
	{
		return initialize_async_submit(_instance, matA_M, matA_N, matB_M, matB_N, block_M, block_N, neighbour_M, neighbour_N, stride_M, stride_N);
	}
}

extern "C"
bool process(void *_instance, float *matA, float *matB, enum Method method)
{
	Type type = *(Type *)_instance;
	if (type == FULL)
		return process_TypeA(_instance, matA, matB, method);
	else
		return process_async_submit(_instance, matA, matB, method);
}

extern "C"
void getResult(void *_instance, float **result, size_t *result_dim0, size_t *result_dim1, size_t *result_dim2, size_t *result_dim3)
{
	Type type = *(Type*)_instance;
	if (type == FULL) {
		struct Context *instance = (struct Context *)_instance;

		cudaDeviceSynchronize();

		*result = instance->result_buffer;
		*result_dim0 = instance->result_dim0;
		*result_dim1 = instance->result_dim1;
		*result_dim2 = instance->result_dim2;
		*result_dim3 = instance->result_dim3;
	}
	else if (type == COMBILE)
	{
		struct Context_Async *instance = (struct Context_Async*)_instance;

		cudaDeviceSynchronize();

		*result = instance->result_buffer;
		*result_dim0 = instance->result_dim0;
		*result_dim1 = instance->result_dim1;
		*result_dim2 = instance->result_dim2;
		*result_dim3 = instance->result_dim3;
	}
	else
		abort();
}

extern "C"
void finalize(void *_instance)
{
	Type type = *(Type*)_instance;
	if (type == FULL) {
		struct Context *instance = (struct Context *)_instance;

		cudaFreeHost(instance->buffer_A);
		cudaFreeHost(instance->buffer_B);
		cudaFreeHost(instance->result_buffer);
		cudaFree(instance->device_buffer_A);
		cudaFree(instance->device_buffer_B);
		cudaFree(instance->device_result_buffer);

		free(instance);
	}
	else if (type == COMBILE)
	{
		struct Context_Async *instance = (struct Context_Async *)_instance;

		for (uint32_t i = 0; i < numSubmitThread; ++i) {
			cudaStreamDestroy(instance->stream[i]);
		}

		cudaFreeHost(instance->buffer_A);
		cudaFreeHost(instance->buffer_B);
		cudaFreeHost(instance->result_buffer);
		cudaFree(instance->device_buffer_A);
		cudaFree(instance->device_buffer_B);
		cudaFree(instance->device_result_buffer);

		free(instance);
	}
}

bool reset()
{
	cudaError_t cuda_error = cudaDeviceReset();
	return cuda_error == cudaSuccess;
}

extern "C"
void onLoad(void)
{
	pool = new thread_pool(numSubmitThread);
}

extern "C"
void atExit(void)
{
	if (pool) {
		delete pool;
		pool = nullptr;
	}
}
