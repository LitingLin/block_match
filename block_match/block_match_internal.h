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
	int numberOfGPUProcessorThread;
	bool hasGPU;
};

extern GlobalContext globalContext;

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

	size_t neighbour_M;
	size_t neighbour_N;
	size_t stride_M;
	size_t stride_N;


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

	cudaStream_t *stream;
	ThreadPool *pool;
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

#define thread_pool_launcher(threadPool, function, parameters) block_match_internal::thread_pool_launcher_helper<decltype(function), function>(threadPool, parameters);
