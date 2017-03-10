#pragma once

#include "lib_match.h"

#include <spdlog/spdlog.h>
#include "stack_trace.h"

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
	execution_service pool;
	int numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread;
	bool hasGPU;
};

extern GlobalContext globalContext;


/*
 * M means the first dimension, N means the second dimension
** So, two dimensions is assumed
*/

template <typename Type>
struct ExecutionContext
{
	Type *matrixA, *matrixB, *matrixC,
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
		indexA_M_begin, indexA_N_begin,
		indexA_M_end, indexA_N_end,
		startIndexOfMatrixA_M, startIndexOfMatrixA_N, numberOfIteration;
	
	/* Gpu Stuff */
	cudaStream_t streamA, streamB; // TODO: Double buffering
	int maxNumberOfThreadsPerProcessor,
		numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, lengthOfGpuTaskQueue;
};

template <typename Type>
using PadFunction = void(const Type *old_ptr, Type *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);

template <typename Type>
using ExecutionFunction = unsigned(ExecutionContext<Type> *);

// TODO support int64
template <typename Type>
struct BlockMatchContext
{
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

	int indexA_M_begin;
	int indexA_M_end;
	int indexA_N_begin;
	int indexA_N_end;

	int numberOfIndexRetain;

	int numberOfThreads;

	PadFunction<Type> *padMethodA;
	PadFunction<Type> *padMethodB;
	ExecutionFunction<Type> *executionMethod;

	int numberOfBlockBPerBlockA_M;
	int numberOfBlockBPerBlockA_N;
	int numberOfBlockBPerBlockA;

	int C_dimensions[3];

	cudaStream_t *stream;

	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;

	void **threadPoolTaskHandle;

	struct Buffer {
		Type *matrixA_buffer;
		Type *matrixB_buffer;
		Type *matrixC_buffer;
		Type *matrixA_deviceBuffer;
		Type *matrixB_deviceBuffer;
		Type *matrixC_deviceBuffer;

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
		ExecutionContext<Type> *executionContext;
	} workerContext;

	struct OptionalPerThreadBufferPointer
	{
		int *index_x_internal;
		int *index_y_internal;
	} *optionalPerThreadBufferPointer;

	struct OptionalBuffer
	{
		Type *matrixA_padded_internal;
		Type *matrixB_padded_internal;
		int *index_x_internal;
		int *index_y_internal;
	} optionalBuffer;

	struct PerThreadBufferPointer
	{
		Type *matrixA_buffer;
		Type *matrixB_buffer;
		Type *matrixC_buffer;
		Type *matrixA_deviceBuffer;
		Type *matrixB_deviceBuffer;
		Type *matrixC_deviceBuffer;

		int *index_x_sorting_buffer;
		int *index_y_sorting_buffer;

		int *index_raw_sorting_buffer;
	} *perThreadBufferPointer;
};

struct ArrayMatchExecutionContext
{
	float *A;
	float *B;
	float *C;
	float *bufferA;
	float *bufferB;
	float *deviceBufferA;
	float *deviceBufferB;
	float *deviceBufferC;
	int numberOfArrayA;
	int numberOfArrayB;	
	int lengthOfArray;
	int startIndexA;
	int numberOfIteration;
	int numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread;
};

struct ArrayMatchContext
{
	int numberOfArrayA;
	int numberOfArrayB;
	int lengthOfArray;
	float *result;

	float *bufferA;
	float *bufferB;

	float *deviceBufferA;
	float *deviceBufferB;
	float *deviceBufferC;

	int numberOfThreads;

	ArrayMatchExecutionContext *executionContext;
	void **taskHandle;
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
	void* thread_pool_launcher_helper(execution_service &pool, Params<Args...> & params)
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

template <typename Type>
void copyBlock(Type *buf, const Type *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);
template <typename Type>
void copyBlockWithSymmetricPadding(Type *buf, const Type *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);

template <typename Type>
void standardize_cpu(Type *sequence, int size);
template <typename Type>
cudaError_t standardize(Type *sequence, int numberOfBlocks, int size, int numThreads, cudaStream_t stream);

template <typename Type>
cudaError_t block_match_mse(Type *blocks_A, Type *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
cudaError_t block_match_mse_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
cudaError_t block_match_cc(Type *blocks_A, Type *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_blockSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
cudaError_t block_match_cc_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A, int numBlocks_B,
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);

template <typename Type>
void block_match_mse_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result);
template <typename Type>
void block_match_cc_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result);

template <typename Type>
void block_sort(int *index, Type *value, int size);
template <typename Type>
void block_sort_partial(int *index, Type *value, int size, int retain);
template <typename Type>
void block_sort_descend(int *index, Type *value, int size);
template <typename Type>
void block_sort_partial_descend(int *index, Type *value, int size, int retain);

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations);

void setLastErrorString(const char *string, ...);
void setCudaLastErrorString(cudaError_t cudaError, char *message);

template <typename Type>
cudaError_t arrayMatchMse(Type *A, Type *B, Type *C,
	int lengthOfArray,
	int numberOfProcessors, int numberOfThreads);
template <typename Type>
cudaError_t arrayMatchMse(Type *A, Type *B, Type *C,
	int lengthOfArray, int numberOfArray,
	int numberOfProcessors, int numberOfThreads);
template <typename Type>
cudaError_t arrayMatchCc(Type *A, Type *B, Type *C,
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

void appendLastErrorString(const char *string, ...);

template <typename Type>
BlockMatchContext<Type> * allocateContext(const int numberOfThreads);
enum class InternalBufferType
{
	MatrixA_Padded_Buffer,
	MatrixB_Padded_Buffer,
	Index_X_Internal,
	Index_Y_Internal
};

template <typename Type>
bool allocateInternalBuffer(BlockMatchContext<Type> *context, enum class InternalBufferType bufferType);
template <typename Type>
void initializeWorkerInternalBuffer(BlockMatchContext<Type> *context, void *buffer, enum class InternalBufferType bufferType);

bool isInterruptPending();

class StackTracker : private StackWalker
{
public:
	char *getStackTraceMessage();
protected:
	enum { STACKTRACER_MESSAGE_MAX_LENGTH = 4096 };
private:
	virtual void OnOutput(LPCSTR szText) override;
};

__forceinline__
void traceStackToLastErrorString()
{
	StackTracker stackTracker;
	appendLastErrorString(stackTracker.getStackTraceMessage());
}

#ifndef DEBUG
#define CALL_DBG __debugbreak();
#else
#define CALL_DBG
#endif

enum { ERROR_MESSAGE_LENGTH = 8192 };

#define ERROR_CHECK_POINT(message, ...) \
{CALL_DBG \
setLastErrorString("Check point failed in file %s(%d) in function %s\n", __FILE__, __LINE__, __func__); \
appendLastErrorString(message, __VA_ARGS__); \
traceStackToLastErrorString();}

#define CUDA_ERROR_CHECK_POINT(cudaError) \
ERROR_CHECK_POINT("Cuda Error Code: %d, Message: %s", cudaError, cudaGetErrorString(cudaError))

enum class memory_allocation_type
{
	memory,
	page_locked,
	gpu
};

class memory_allocation_counter
{
public:
	memory_allocation_counter();
	void register_allocator(size_t size, memory_allocation_type type);
	void allocated(size_t size, memory_allocation_type type);
	void released(size_t size, memory_allocation_type type);
	void trigger_error(size_t size, memory_allocation_type type) const;
private:
	size_t max_memory_size;
	size_t max_page_locked_memory_size;
	size_t max_gpu_memory_size;
	size_t current_memory_size;
	size_t current_page_locked_memory_size;
	size_t current_gpu_memory_size;
} extern g_memory_allocator;

template <typename Type>
class system_memory_allocator
{
public:
	system_memory_allocator(size_t elem_size, bool is_temp = false);
	~system_memory_allocator();
	Type *alloc();
	void release();
	Type *get();
private:
	void *ptr;
	size_t size;
};

template <typename Type>
system_memory_allocator<Type>::system_memory_allocator(size_t elem_size, bool is_temp)
	: ptr(nullptr), size(elem_size * sizeof(Type))
{
	if (!is_temp)
		g_memory_allocator.register_allocator(size, memory_allocation_type::memory);
}

template <typename Type>
system_memory_allocator<Type>::~system_memory_allocator()
{
	if (ptr)
		release();
}

template <typename Type>
Type* system_memory_allocator<Type>::alloc()
{
	ptr = malloc(size);
	if (ptr)
		g_memory_allocator.allocated(size, memory_allocation_type::memory);
	else
		g_memory_allocator.trigger_error(size, memory_allocation_type::memory);
	return ptr;
}

template <typename Type>
void system_memory_allocator<Type>::release()
{
	free(ptr);
	g_memory_allocator.released(size, memory_allocation_type::memory);
	ptr = nullptr;
}

template <typename Type>
Type* system_memory_allocator<Type>::get()
{
	return ptr;
}

template <typename Type>
class page_locked_memory_allocator
{
public:
	page_locked_memory_allocator(size_t elem_size, bool is_temp = false);
	~page_locked_memory_allocator();
	Type *alloc();
	void release();
	Type *get();
private:
	void *ptr;
	size_t size;
};

template <typename Type>
page_locked_memory_allocator<Type>::page_locked_memory_allocator(size_t elem_size, bool is_temp)
	: ptr(nullptr), size(elem_size * sizeof(Type))
{
	if (!is_temp)
		g_memory_allocator.register_allocator(size, memory_allocation_type::page_locked);
}

template <typename Type>
page_locked_memory_allocator<Type>::~page_locked_memory_allocator()
{
	if (ptr)
		release();
}

template <typename Type>
Type* page_locked_memory_allocator<Type>::alloc()
{
	if (cudaMallocHost(&ptr, size) == cudaSuccess)
		g_memory_allocator.allocated(size, memory_allocation_type::page_locked);
	else
		g_memory_allocator.trigger_error(size, memory_allocation_type::page_locked);
	return ptr;
}

template <typename Type>
void page_locked_memory_allocator<Type>::release()
{
	cudaFreeHost(ptr);
	g_memory_allocator.released(size, memory_allocation_type::page_locked);
	ptr = nullptr;
}

template <typename Type>
Type* page_locked_memory_allocator<Type>::get()
{
	return ptr;
}

template <typename Type>
class gpu_memory_allocator
{
public:
	gpu_memory_allocator(size_t elem_size, bool is_temp = false);
	~gpu_memory_allocator();
	Type *alloc();
	void release();
	Type *get();
private:
	void *ptr;
	size_t size;
};

template <typename Type>
gpu_memory_allocator<Type>::gpu_memory_allocator(size_t elem_size, bool is_temp)
	: ptr(nullptr), size(elem_size * sizeof(Type))
{
	if (!is_temp)
		g_memory_allocator.register_allocator(size, memory_allocation_type::gpu);
}

template <typename Type>
gpu_memory_allocator<Type>::~gpu_memory_allocator()
{
	if (ptr)
		release();
}

template <typename Type>
Type* gpu_memory_allocator<Type>::alloc()
{
	if (cudaMalloc(&ptr, size) == cudaSuccess)
		g_memory_allocator.allocated(size, memory_allocation_type::gpu);
	else
		g_memory_allocator.trigger_error(size, memory_allocation_type::gpu);
	return ptr;
}

template <typename Type>
void gpu_memory_allocator<Type>::release()
{
	cudaFree(ptr);
	g_memory_allocator.released(size, memory_allocation_type::gpu);
	ptr = nullptr;
}

template <typename Type>
Type* gpu_memory_allocator<Type>::get()
{
	return ptr;
}
