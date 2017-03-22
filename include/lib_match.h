#pragma once

#include <string>

#define LIB_MATCH_EXPORT 

enum class LibMatchMeasureMethod { mse, cc };

enum class LibMatchErrorCode
{
	errorMemoryAllocation,
	errorPageLockedMemoryAllocation,
	errorGpuMemoryAllocation,
	errorCuda,
	errorInternal,
	success
};

enum class SearchType
{
	local,
	global
};

enum class PadMethod
{
	zero,
	circular,
	replicate,
	symmetric
};

enum class BorderType
{
	normal,
	includeLastBlock
};

enum class SearchFrom
{
	topLeft,
	center
};
/*
LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArrayA, int numberOfArrayB, int lengthOfArray);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, LibMatchMeasureMethod method,
	float **result);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchFinalize(void *instance);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumMemoryAllocationSize();

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int lengthOfArray);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArrayA, int numberOfArrayB, int lengthOfArray, int numberOfThreads);
*/
// SearchRegion size 0 for full search
template <typename Type>
void blockMatchInitialize(void **instance,
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
	BorderType sequenceABorderType,
	SearchFrom searchFrom,
	bool sort,
	int matrixA_M, int matrixA_N, int matrixB_M, int matrixB_N,
	int searchRegion_M, int searchRegion_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int matrixAPadding_M_pre, int matrixAPadding_M_post,
	int matrixAPadding_N_pre, int matrixAPadding_N_post,
	int matrixBPadding_M_pre, int matrixBPadding_M_post,
	int matrixBPadding_N_pre, int matrixBPadding_N_post,
	int numberOfIndexRetain,
	int *matrixC_M, int *matrixC_N, int *matrixC_X,
	int *matrixA_padded_M = nullptr, int *matrixA_padded_N = nullptr,
	int *matrixB_padded_M = nullptr, int *matrixB_padded_N = nullptr);

template <typename Type>
void blockMatchExecute(void *instance, Type *A, Type *B,
	Type *C,
	Type *padded_A = nullptr, Type *padded_B = nullptr,
	int *index_x = nullptr, int *index_y = nullptr);

template <typename Type>
void blockMatchFinalize(void *instance);

LIB_MATCH_EXPORT
bool libMatchReset();

LIB_MATCH_EXPORT
void libMatchOnLoad();

LIB_MATCH_EXPORT
void libMatchAtExit();

typedef void LibMatchSinkFunction(const char *);
LIB_MATCH_EXPORT
void libMatchRegisterLoggingSinkFunction(LibMatchSinkFunction sinkFunction);

typedef bool LibMatchInterruptPendingFunction();
void libMatchRegisterInterruptPeddingFunction(LibMatchInterruptPendingFunction);

template <typename T>
void zeroPadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
template <typename T>
void circularPadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
template <typename T>
void replicatePadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
template <typename T>
void symmetricPadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);

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
	return static_cast<Type*>(ptr);
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
	return static_cast<Type*>(ptr);
}