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

// SearchRegion size 0 for full search
LIB_MATCH_EXPORT
bool blockMatchInitialize(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int searchRegion_M, int searchRegion_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N,
	int retain);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, LibMatchMeasureMethod method,
	float **result);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchFinalize(void *instance);

#define LIB_MATCH_MAX_MESSAGE_LENGTH 128

LIB_MATCH_EXPORT
void libMatchGetLastErrorString(char *buffer, size_t size);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
bool blockMatchExecute(void *_instance, float *matA, float *matB, LibMatchMeasureMethod method, 
	int **_index_x, int **_index_y, float **_result, int *dimensionOfResult);

LIB_MATCH_EXPORT
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

template <typename T>
void zeroPadding(T *old_ptr, T *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
template <typename T>
void circularPadding(T *old_ptr, T *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
template <typename T>
void replicatePadding(T *old_ptr, T *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
template <typename T>
void symmetricPadding(T *old_ptr, T *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);