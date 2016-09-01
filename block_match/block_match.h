#pragma once

#ifdef __cplusplus
#define LIB_BLOCK_MATCH_EXPORT extern "C"
#else
#define LIB_BLOCK_MATCH_EXPORT
#endif

#include <stdbool.h>
#include <stddef.h>

enum Method { MSE, CC };

enum ErrorCode
{
	LibMatchErrorMemoryAllocation,
	LibMatchErrorPageLockedMemoryAllocation,
	LibMatchErrorGpuMemoryAllocation,
	LibMatchErrorCuda,
	LibMatchErrorOk
};

LIB_BLOCK_MATCH_EXPORT
bool initialize(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N,
	int retain);

LIB_BLOCK_MATCH_EXPORT
enum ErrorCode arrayMatchInitialize(void **instance,
	int numberOfArray, int lengthOfArray);

LIB_BLOCK_MATCH_EXPORT
bool arrayMatchExecute(void *instance, float *A, float *B, enum Method method,
	float *result);

LIB_BLOCK_MATCH_EXPORT
void arrayMatchFinalize(void *instance);

LIB_BLOCK_MATCH_EXPORT
void getLastErrorString(char *buffer, size_t size);

LIB_BLOCK_MATCH_EXPORT
size_t arrayMatchGetMaximumMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_BLOCK_MATCH_EXPORT
size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_BLOCK_MATCH_EXPORT
size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArray, int lengthOfArray);

LIB_BLOCK_MATCH_EXPORT
bool execute(void *_instance, float *matA, float *matB, enum Method method, int **_index_x, int **_index_y, float **_result, int *dimensionOfResult);

LIB_BLOCK_MATCH_EXPORT
void finalize(void *instance);

LIB_BLOCK_MATCH_EXPORT
bool reset();

LIB_BLOCK_MATCH_EXPORT
void onLoad();

LIB_BLOCK_MATCH_EXPORT
void atExit();

typedef void SinkFunction(const char *);
LIB_BLOCK_MATCH_EXPORT
void registerLoggingSinkFunction(SinkFunction sinkFunction);