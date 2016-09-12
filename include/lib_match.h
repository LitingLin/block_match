#pragma once

#ifdef __cplusplus
#define LIB_MATCH_EXPORT extern "C"
#else
#define LIB_MATCH_EXPORT
#endif

#include <stdbool.h>
#include <stddef.h>

enum LibMatchMeasureMethod { LIB_MATCH_MSE, LIB_MATCH_CC };

enum LibMatchErrorCode
{
	LibMatchErrorMemoryAllocation,
	LibMatchErrorPageLockedMemoryAllocation,
	LibMatchErrorGpuMemoryAllocation,
	LibMatchErrorCuda,
	LibMatchErrorInternal,
	LibMatchErrorOk
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
enum LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArray, int lengthOfArray);

LIB_MATCH_EXPORT
enum LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, enum LibMatchMeasureMethod method,
	float **result);

LIB_MATCH_EXPORT
enum LibMatchErrorCode arrayMatchFinalize(void *instance);

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
bool blockMatchExecute(void *_instance, float *matA, float *matB, enum LibMatchMeasureMethod method, int **_index_x, int **_index_y, float **_result, int *dimensionOfResult);

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