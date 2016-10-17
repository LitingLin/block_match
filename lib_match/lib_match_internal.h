#pragma once

#include "lib_match.h"

#include <spdlog/spdlog.h>

extern spdlog::logger logger;

#include <cuda_runtime.h>

#if defined _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined __GNUC__
#define FORCE_INLINE __inline__ __attribute__((always_inline))
#endif

#include "thread_pool.h"

#define LIB_MATCH_OUT(PARAMETER) out_##PARAMETER

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


/* 
 * M means the first dimension, N means the second dimension
** So, two dimensions is assumed
*/

struct ExecutionContext
{
	float *matrixA, *matrixB, *matrixC,
		*matrixA_buffer, *matrixB_buffer, *matrixC_buffer,
		*matrixA_deviceBuffer, *matrixB_deviceBuffer, *matrixC_deviceBuffer;
	int matrixA_M, matrixA_N,
		matrixB_M, matrixB_N;
	int *index_x, *index_y, *index_x_buffer, *index_y_buffer,
		*rawIndexTemplate, *rawIndexBuffer,
		block_M, block_N,
		strideA_M, strideA_N,
		strideB_M, strideB_N,
		neighbour_M, neighbour_N,
		numberOfBlockBPerBlockA,
		numberOfIndexRetain,
		startIndexOfMatrixA_M, startIndexOfMatrixA_N, numberOfIteration;

	int matrixBPadding_M_pre, matrixBPadding_N_pre;

	/* Gpu Stuff */
	cudaStream_t streamA, streamB; // TODO: Double buffering
	int maxNumberOfThreadsPerProcessor,
		numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, lengthOfGpuTaskQueue;
};

typedef void PadMethod(const float *old_ptr, float *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);

typedef unsigned ExecutionMethod(ExecutionContext *);

// TODO support int64
struct BlockMatchContext
{
	SearchType searchType;
	LibMatchMeasureMethod measureMethod;

	int matrixA_M;
	int matrixA_N;
	int matrixB_M;
	int matrixB_N;

	int matrixA_padded_M;
	int matrixA_padded_N;
	int matrixB_padded_M;
	int matrixB_padded_N;

	int block_M;
	int block_N;

	int searchRegion_M;
	int searchRegion_N;

	int strideA_M;
	int strideA_N;
	int strideB_M;
	int strideB_N;

	int matrixAPadding_M_pre;
	int matrixAPadding_M_post;
	int matrixAPadding_N_pre;
	int matrixAPadding_N_post;
	int matrixBPadding_M_pre;
	int matrixBPadding_M_post;
	int matrixBPadding_N_pre;
	int matrixBPadding_N_post;

	int numberOfIndexRetain;

	int numberOfThreads;

	PadMethod* padMethod;
	ExecutionMethod *executionMethod;

	int numberOfBlockBPerBlockA_M;
	int numberOfBlockBPerBlockA_N;
	int numberOfBlockBPerBlockA;

	int C_dimensions[4];

	cudaStream_t *stream;

	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;

	void **threadPoolTaskHandle;
	
	struct Buffer {
		float *matrixA_buffer;
		float *matrixB_buffer;
		float *matrixC_buffer;
		float *matrixA_deviceBuffer;
		float *matrixB_deviceBuffer;
		float *matrixC_deviceBuffer;

		int *index_x_sorting_buffer;
		int *index_y_sorting_buffer;

		int *common_buffer; // index template
		int *index_raw_sorting_buffer;
	} buffer;

	struct WorkerContext
	{
		int *numberOfIteration;
		int *rawMatrixCIndex_begin;
		int *beginMatrixAIndex_M;
		int *beginMatrixAIndex_N;
		ExecutionContext *executionContext;
	} workerContext;

	struct OptionalPerThreadBufferPointer
	{
		int *index_x_internal;
		int *index_y_internal;
	} *optionalPerThreadBufferPointer;

	struct OptionalBuffer
	{
		float *matrixA_padded_internal;
		float *matrixB_padded_internal;
		int *index_x_internal;
		int *index_y_internal;
	} optionalBuffer;

	struct PerThreadBufferPointer
	{
		float *matrixA_buffer;
		float *matrixB_buffer;
		float *matrixC_buffer;
		float *matrixA_deviceBuffer;
		float *matrixB_deviceBuffer;
		float *matrixC_deviceBuffer;

		int *index_x_sorting_buffer;
		int *index_y_sorting_buffer;

		int *index_raw_sorting_buffer;
	} *perThreadBufferPointer;
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
int getLength(int matSize, int prePaddingSize, int postPaddingSize, int blockSize, int strideSize);
int determineEndOfIndex(int matSize, int blockSize);
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

BlockMatchContext * allocateContext(const int numberOfThreads);
enum class InternalBufferType
{
	MatrixA_Padded_Buffer,
	MatrixB_Padded_Buffer,
	Index_X_Internal,
	Index_Y_Internal
};

bool allocateInternalBuffer(BlockMatchContext *context, enum class InternalBufferType bufferType);
void initializeWorkerInternalBuffer(BlockMatchContext *context, void *buffer, enum class InternalBufferType bufferType);