#pragma once

#include "lib_match.h"

#include <spdlog/spdlog.h>

namespace std {
	class type_index;
}

extern spdlog::logger logger;

#include <cuda_runtime.h>

#if defined _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined __GNUC__
#define FORCE_INLINE __inline__ __attribute__((always_inline))
#endif

#include "execution_service.h"

#ifndef DEBUG
#define CALL_DBG __debugbreak();
#else
#define CALL_DBG
#endif

std::string getStackTrace();

#include <sstream>

class fatal_error_logging
{
public:
	fatal_error_logging(const char* file, int line, const char* function);

	fatal_error_logging(const char* file, int line, const char* function, const char* exp);

	fatal_error_logging(const char* file, int line, const char* function, const char* exp1, const char* op, const char* exp2);

	~fatal_error_logging() noexcept(false);

	std::ostringstream& stream();
private:
	std::ostringstream str_stream;
};

class warning_logging
{
public:
	warning_logging(const char* file, int line, const char* function);

	warning_logging(const char* file, int line, const char* function, const char* exp);

	warning_logging(const char* file, int line, const char* function, const char* exp1, const char* op, const char* exp2);

	~warning_logging();

	std::ostringstream& stream();
private:
	std::ostringstream str_stream;
};

template <typename T1, typename T2, typename Op>
std::unique_ptr<std::pair<T1, T2>> check_impl(const T1 &a, const T2 &b, Op op) {
	if (op(a, b))
		return nullptr;
	else
		return std::make_unique<std::pair<T1, T2>>(a, b);
}

#define CHECK_POINT(val) \
	if (!(val)) \
fatal_error_logging(__FILE__, __LINE__, __func__, #val).stream()

#define CHECK_POINT_OP(exp1, exp2, op, functional_op)  \
	if (auto _rc = check_impl((exp1), (exp2), functional_op)) \
fatal_error_logging(__FILE__, __LINE__, __func__, #exp1, #op, #exp2).stream() << '(' << _rc->first << " vs. " << _rc->second << ") "

#define CHECK_POINT_EQ(exp1, exp2) \
	CHECK_POINT_OP(exp1, exp2, ==, std::equal_to<>())
#define CHECK_POINT_NE(exp1, exp2) \
	CHECK_POINT_OP(exp1, exp2, !=, std::not_equal_to<>())
#define CHECK_POINT_LE(exp1, exp2) \
	CHECK_POINT_OP(exp1, exp2, <=, std::less_equal<>())
#define CHECK_POINT_LT(exp1, exp2) \
	CHECK_POINT_OP(exp1, exp2, <, std::less<>())
#define CHECK_POINT_GE(exp1, exp2) \
	CHECK_POINT_OP(exp1, exp2, >=, std::greater_equal<>())
#define CHECK_POINT_GT(exp1, exp2) \
	CHECK_POINT_OP(exp1, exp2, >, std::greater<>())

#define CUDA_CHECK_POINT(cudaExp) \
	CHECK_POINT_EQ(cudaExp, cudaSuccess) << "CUDA Error message: " << cudaGetErrorString(_rc->first)

#define NOT_IMPLEMENTED_ERROR \
	fatal_error_logging(__FILE__, __LINE__, __func__).stream() << "Unknown internal error. "

struct GlobalContext
{
	GlobalContext();
	bool initialize();

	unsigned numberOfThreads;
	execution_service exec_serv;
	int numberOfGPUDeviceMultiProcessor;
	const int numberOfGPUProcessorThread;
	bool hasGPU;
};

extern GlobalContext globalContext;

class memory_allocation_counter
{
public:
	memory_allocation_counter();
	void register_allocator(size_t size, memory_type type);
	void unregister_allocator(size_t size, memory_type type);
	void allocated(size_t size, memory_type type);
	void released(size_t size, memory_type type);
	void trigger_error(size_t size, memory_type type) const;
	void get_max_memory_required(size_t *max_memory_size,
		size_t *max_page_locked_memory_size, size_t *max_gpu_memory_size) const;
private:
	size_t max_memory_size;
	size_t max_page_locked_memory_size;
	size_t max_gpu_memory_size;
	size_t current_memory_size;
	size_t current_page_locked_memory_size;
	size_t current_gpu_memory_size;
} extern g_memory_allocator;

template <typename Type, memory_type malloc_type>
struct raw_allocator {
	static Type *alloc(size_t size);
	static void free(Type *ptr);
};

template <typename Type>
struct raw_allocator <Type, memory_type::system>{
	static Type *alloc(size_t size)
	{
		return static_cast<Type*>(::malloc(size));
	}
	static void free(Type *ptr)
	{
		::free(ptr);
	}
};

template <typename Type>
struct raw_allocator <Type, memory_type::page_locked> {
	static Type *alloc(size_t size)
	{
		Type *ptr;
		cudaError_t cudaError = cudaMallocHost(&ptr, size);
		if (cudaError == cudaErrorMemoryAllocation)
			return nullptr;
		CUDA_CHECK_POINT(cudaError);
		return ptr;
	}
	static void free(Type *ptr)
	{
		CUDA_CHECK_POINT(cudaFreeHost(ptr));
	}
};

template <typename Type>
struct raw_allocator <Type, memory_type::gpu> {
	static Type *alloc(size_t size)
	{
		Type *ptr;
		cudaError_t cudaError = cudaMalloc(&ptr, size);
		if (cudaError == cudaErrorMemoryAllocation)
			return nullptr;
		CUDA_CHECK_POINT(cudaError);
		return ptr;
	}
	static void free(Type *ptr)
	{
		CUDA_CHECK_POINT(cudaFree(ptr));
	}
};

template <typename Type, memory_type malloc_type>
class memory_allocator
{
public:
	memory_allocator(size_t elem_size = 0);
	~memory_allocator();
	Type *alloc();
	void release();
	Type *get();
	void resize(size_t elem_size);
private:
	Type *ptr;
	size_t size;
};

template <typename Type, memory_type malloc_type>
memory_allocator<Type, malloc_type>::memory_allocator(size_t elem_size)
	: ptr(nullptr), size(elem_size * sizeof(Type))
{
}

template <typename Type, memory_type malloc_type>
memory_allocator<Type, malloc_type>::~memory_allocator()
{
	if (ptr)
		release();

	g_memory_allocator.unregister_allocator(size, malloc_type);
}

template <typename Type, memory_type malloc_type>
Type *memory_allocator<Type, malloc_type>::alloc()
{
	if (size) {
		ptr = raw_allocator<Type, malloc_type>::alloc(size);
		if (ptr)
			g_memory_allocator.allocated(size, malloc_type);
		else
			g_memory_allocator.trigger_error(size, malloc_type);
		return static_cast<Type*>(ptr);
	}
	return nullptr;
}

template <typename Type, memory_type malloc_type>
void memory_allocator<Type, malloc_type>::release()
{
	raw_allocator<Type, malloc_type>::free(ptr);
	g_memory_allocator.released(size, malloc_type);
	ptr = nullptr;
}

template <typename Type, memory_type malloc_type>
Type* memory_allocator<Type, malloc_type>::get()
{
	return static_cast<Type*>(ptr);
}

template <typename Type, memory_type malloc_type>
void memory_allocator<Type, malloc_type>::resize(size_t elem_size)
{
	if (ptr)
		release();
	g_memory_allocator.unregister_allocator(size, malloc_type);
	size = elem_size * sizeof(Type);
	g_memory_allocator.register_allocator(size, malloc_type);
}

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
	cudaStream_t stream; // TODO: Double buffering
	int maxNumberOfThreadsPerProcessor,
		numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, lengthOfGpuTaskQueue;
};

template <typename Type>
using PadFunction = void(const Type *old_ptr, Type *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);

template <typename Type>
using ExecutionFunction = unsigned(ExecutionContext<Type> *);

class cudaStream_guard
{
public:
	cudaStream_guard();
	~cudaStream_guard();
	operator cudaStream_t() const;
private:
	cudaStream_t stream;
};

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

	std::vector<cudaStream_guard> streams;
	int numberOfSubmitThreadsPerProcessor, numberOfSubmitProcessors, sizeOfGpuTaskQueue;

	std::vector<void *>threadPoolTaskHandle;

	memory_allocator<int, memory_type::system> common_buffer; // index template
	
	struct WorkerContext
	{
		int numberOfIteration;
		int rawMatrixCIndex_begin;
		int beginMatrixAIndex_M;
		int beginMatrixAIndex_N;
		std::unique_ptr<ExecutionContext<Type>> executionContext;
	};
	std::vector<WorkerContext> workerContext;
	/*
	struct OptionalPerThreadBuffer
	{
		memory_allocator<int, memory_type::system> index_x_internal;
		memory_allocator<int, memory_type::system> index_y_internal;
	};
	std::vector<OptionalPerThreadBuffer> optionalPerThreadBuffer;

	struct OptionalBuffer
	{
		memory_allocator<Type, memory_type::system> matrixA_padded_internal;
		memory_allocator<Type, memory_type::system> matrixB_padded_internal;
	};
	std::vector<OptionalBuffer> optionalBuffer;
	*/
	struct PerThreadBuffer
	{
		memory_allocator<Type, memory_type::page_locked> matrixA_buffer;
		memory_allocator<Type, memory_type::page_locked> matrixB_buffer;
		memory_allocator<Type, memory_type::page_locked> matrixC_buffer;
		memory_allocator<Type, memory_type::gpu> matrixA_deviceBuffer;
		memory_allocator<Type, memory_type::gpu> matrixB_deviceBuffer;
		memory_allocator<Type, memory_type::gpu> matrixC_deviceBuffer;

		memory_allocator<int, memory_type::system> index_x_sorting_buffer;
		memory_allocator<int, memory_type::system> index_y_sorting_buffer;

		memory_allocator<int, memory_type::system> index_raw_sorting_buffer;
	};
	std::vector<PerThreadBuffer> perThreadBuffer;
};

template <typename Type>
struct ArrayMatchExecutionContext
{
	Type *A;
	Type *B;
	Type *C;
	Type *bufferA;
	Type *bufferB;
	Type *deviceBufferA;
	Type *deviceBufferB;
	Type *deviceBufferC;
	int numberOfArrayA;
	int numberOfArrayB;	
	int lengthOfArray;
	int startIndexA;
	int numberOfIteration;
	int numberOfGPUDeviceMultiProcessor;
	int numberOfGPUProcessorThread;
};

template <typename Type>
struct ArrayMatchContext
{
	int numberOfArrayA;
	int numberOfArrayB;
	int lengthOfArray;
	Type *result;

	Type *bufferA;
	Type *bufferB;

	Type *deviceBufferA;
	Type *deviceBufferB;
	Type *deviceBufferC;

	int numberOfThreads;

	ArrayMatchExecutionContext<Type> *executionContext;
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
cudaError_t lib_match_mse(Type *blocks_A, Type *blocks_B, int numBlocks_A, 
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
cudaError_t lib_match_mse_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A, 
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
cudaError_t lib_match_cc(Type *blocks_A, Type *blocks_B, int numBlocks_A, 
	int block_B_blockSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);
template <typename Type>
cudaError_t lib_match_cc_check_border(Type *blocks_A, Type *blocks_B, int numBlocks_A,
	int block_B_groupSize, int blockSize, Type *result, int numProcessors, int numThreads, cudaStream_t stream);

template <typename Type>
void lib_match_mse_cpu(Type *block_A, Type *block_B, int blockSize, Type *result);
template <typename Type>
void lib_match_mse_cpu_sse2(Type *block_A, Type *block_B, int blockSize, Type *result);
template <typename Type>
void lib_match_mse_cpu_avx(Type *block_A, Type *block_B, int blockSize, Type *result);
template <typename Type>
void lib_match_mse_cpu_avx2(Type *block_A, Type *block_B, int blockSize, Type *result);

template <typename Type>
void lib_match_cc_cpu(Type *block_A, Type *block_B, int blockSize, Type *result);
template <typename Type>
void lib_match_cc_cpu_sse2(Type *block_A, Type *block_B, int blockSize, Type *result);
template <typename Type>
void lib_match_cc_cpu_avx(Type *block_A, Type *block_B, int blockSize, Type *result);
template <typename Type>
void lib_match_cc_cpu_avx2(Type *block_A, Type *block_B, int blockSize, Type *result);

template <typename Type>
void lib_match_sort(int *index, Type *value, int size);
template <typename Type>
void lib_match_sort_partial(int *index, Type *value, int size, int retain);
template <typename Type>
void lib_match_sort_descend(int *index, Type *value, int size);
template <typename Type>
void lib_match_sort_partial_descend(int *index, Type *value, int size, int retain);

void determineGpuTaskConfiguration(const int maxNumberOfGpuThreads, const int numberOfGpuProcessors, const int numberOfBlockBPerBlockA,
	int *numberOfSubmitThreadsPerProcessor, int *numberOfSubmitProcessors, int *numberOfIterations);

void determinePadSizeAccordingToPatchSize(int mat_M, int mat_N, int patch_M, int patch_N,
	int *M_left, int *M_right, int *N_left, int *N_right);

bool isInterruptPending();

void convert(void *src, std::type_index src_type, void *dst, std::type_index dst_type, size_t size);

void *aligned_block_malloc(size_t size, size_t alignment);
void aligned_free(void *ptr);