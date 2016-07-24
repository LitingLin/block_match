#pragma once

#include <cuda_runtime.h>

#if defined _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined __GNUC__
#define FORCE_INLINE __inline__ __attribute__((always_inline))
#endif

#include "thread_pool.h"

struct GlobalContext
{
	GlobalContext();

	unsigned numberOfThreads;
	ThreadPool pool;
	int numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread;
	bool hasGPU;
};

extern GlobalContext globalContext;

struct Context
{
	int matA_M;
	int matA_N;
	int matB_M;
	int matB_N;
	int block_M;
	int block_N;

	int neighbour_M;
	int neighbour_N;
	
	int strideA_M;
	int strideA_N;
	int strideB_M;
	int strideB_N;

	int sequenceAPadding_M;
	int sequenceAPadding_N;
	int sequenceBPadding_M;
	int sequenceBPadding_N;
	
	float *buffer_A;
	float *buffer_B;
	float *result_buffer;
	float *result;
	float *device_buffer_A;
	float *device_buffer_B;
	float *device_result_buffer;

	int *index_buffer;
	int perThreadBufferSize;
	int *index;

	int result_dims[4];

	int retain;
	cudaStream_t *stream;
};


namespace block_match_internal {
	template<typename R, template<typename...> class Params, typename... Args, std::size_t... I>
	unsigned FORCE_INLINE thread_pool_base_function_helper(R(*func)(Args...), Params<Args...> const&params, std::index_sequence<I...>)
	{
		return func(std::get<I>(params)...);
	}

	template<typename FunctionType, FunctionType function, template<typename...> class Params, typename... Args>
	unsigned thread_pool_base_function(void *arg) {
		Params<Args...> const &params = *static_cast<Params<Args...> *>(arg);
		return thread_pool_base_function_helper(function, params, std::index_sequence_for<Args...>{});
	}
	template <typename FunctionType, FunctionType function, template<typename...> class Params, typename... Args>
	void* thread_pool_launcher_helper(ThreadPool &pool, Params<Args...> & params)
	{
		return pool.submit(thread_pool_base_function< FunctionType, function, Params, Args... >, &params);
	}
}

#define thread_pool_launcher(threadPool, function, parameters) block_match_internal::thread_pool_launcher_helper<decltype(function), function>(threadPool, parameters)


void copyBlock(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);
void copyBlockWithSymmetricPaddding(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);

cudaError_t block_match_mse(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_mse(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, int numTasks, cudaStream_t stream);
cudaError_t block_match_cc(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_blockSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_cc(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, int numTasks, cudaStream_t stream);

void block_sort(int *index, float *value, int size);
void block_sort_partial(int *index, float *value, int size, int retain);