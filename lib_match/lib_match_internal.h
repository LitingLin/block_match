#pragma once

#include <spdlog/spdlog.h>

extern spdlog::logger logger;

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
	bool initialize();

	unsigned numberOfThreads;
	ThreadPool pool;
	int numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread;
	bool hasGPU;
};

extern GlobalContext globalContext;

typedef void PadMethod(const float *old_ptr, float *new_ptr, 
	size_t old_width, size_t old_height, 
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);


/* M means the first dimension, N means the second dimension
** So, two dimensions is assumed
*/

// TODO support int64
struct BlockMatchContext
{
	int matrixA_M;
	int matrixA_N;
	int matrixB_M;
	int matrixB_N;
	int block_M;
	int block_N;
	
	int searchRegion_M;
	int searchRegion_N;
	
	int strideA_M;
	int strideA_N;
	int strideB_M;
	int strideB_N;
	// TODO remove
	int sequenceAPadding_M;
	int sequenceAPadding_N;
	int sequenceBPadding_M;
	int sequenceBPadding_N;

	int sequenceAPadding_M_pre;
	int sequenceAPadding_M_post;
	int sequenceAPadding_N_pre;
	int sequenceAPadding_N_post;
	int sequenceBPadding_M_pre;
	int sequenceBPadding_M_post;
	int sequenceBPadding_N_pre;
	int sequenceBPadding_N_post;
	
	PadMethod* padMethodPointer;
	float *padded_matrixA;
	float *padded_matrixB;

	float *matrixA_buffer;
	float *matrixB_buffer;
	float *matrixC_buffer;
	float *C;
	float *matrixA_deviceBuffer;
	float *matrixB_deviceBuffer;
	float *matrixC_deviceBuffer;

	int *index_x_buffer;
	int *index_y_buffer;
	int *index_x;
	int *index_y;

	int *common_buffer;
	int *indexSorting_buffer;

	int perThreadBufferSize;
	int numberOfBlockBPerBlockA;

	int result_dims[4];

	int numberOfIndexRetain;
	cudaStream_t *stream;

	void *threadPoolTaskHandle;
	std::tuple<float *, float *, float *,
		float *, int, int, int, int, int, int,
		float *, int, int,
		float *,
		int, int,
		int, int,
		int, int,
		int, int,
		int,
		float *, float *, float *,
		int *, int *, int *, int *,
		int *, int *,
		int,
		cudaStream_t, cudaStream_t,
		int, int > *parameterBuffer;
};

struct ArrayMatchContext
{
	int numberOfArray;
	int lengthOfArray;
	float *result;

	float *deviceBufferA;
	float *deviceBufferB;
	float *deviceBufferC;

	int numberOfThreads;
};

namespace lib_match_internal {
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

#define thread_pool_launcher(threadPool, function, parameters) lib_match_internal::thread_pool_launcher_helper<decltype(function), function>(threadPool, parameters)

int getLength(int matSize, int paddingSize, int blockSize, int strideSize);
int determineEndOfIndex(int matSize, int paddingSize, int blockSize);
void generateIndexSequence(int *index, int size);

void copyBlock(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);
void copyBlockWithSymmetricPadding(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);

void standardize_cpu(float *sequence, int size);
cudaError_t standardize(float *sequence, int numberOfBlocks, int size, int numThreads, cudaStream_t stream);

cudaError_t block_match_mse(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_mse_check_border(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_cc(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_blockSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_cc_check_border(float *blocks_A, float *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);

void block_match_mse_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result);
void block_match_cc_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result);

void block_sort(int *index, float *value, int size);
void block_sort_partial(int *index, float *value, int size, int retain);

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations);

void setLastErrorString(const char *string, ...);
void setCudaLastErrorString(cudaError_t cudaError, char *message);

cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads);
cudaError_t arrayMatchMse(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);
cudaError_t arrayMatchCc(float *A, float *B, float *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);


size_t arrayMatchPerThreadDeviceBufferASize(const int numberOfGpuDeviceMultiProcessor,
const int numberOfGpuProcessorThread,
const int lengthOfArray);

size_t arrayMatchPerThreadDeviceBufferBSize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread,
	const int lengthOfArray);

size_t arrayMatchPerThreadDeviceBufferCSize(const int numberOfGpuDeviceMultiProcessor,
	const int numberOfGpuProcessorThread);

void determinePadSizeAccordingToPatchSize(int mat_M, int mat_N, int patch_M, int patch_N,
	int *M_left, int *M_right, int *N_left, int *N_right);
