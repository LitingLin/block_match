#pragma once

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

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, LibMatchMeasureMethod method,
	float **result);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchFinalize(void *instance);

#define LIB_MATCH_MAX_MESSAGE_LENGTH 32768

LIB_MATCH_EXPORT
void libMatchGetLastErrorString(char *buffer, size_t size);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArray, int lengthOfArray);

// SearchRegion size 0 for full search
template <typename Type>
bool blockMatchInitialize(void **instance,
	SearchType searchType,
	LibMatchMeasureMethod measureMethod,
	PadMethod padMethodA, PadMethod padMethodB,
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
bool blockMatchExecute(void *_instance, Type *A, Type *B,
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