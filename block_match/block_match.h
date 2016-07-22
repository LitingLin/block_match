#pragma once

#ifdef __cplusplus
#define LIB_BLOCK_MATCH_EXPORT extern "C"
#else
#define LIB_BLOCK_MATCH_EXPORT
#endif

#include <stdbool.h>
#include <stddef.h>

enum Method { MSE, CC };

LIB_BLOCK_MATCH_EXPORT
bool initialize(void **_instance,
	int matA_M, int matA_N, int matB_M, int matB_N,
	int block_M, int block_N,
	int strideA_M, int strideA_N,
	int strideB_M, int strideB_N,
	int paddingA_M, int paddingA_N,
	int paddingB_M, int paddingB_N);

LIB_BLOCK_MATCH_EXPORT
bool process(void *instance, float *matA, float *matB, enum Method method, int** index, float **result, int *dimensionOfResult);

LIB_BLOCK_MATCH_EXPORT
void finalize(void *instance);

LIB_BLOCK_MATCH_EXPORT
bool reset();

LIB_BLOCK_MATCH_EXPORT
void onLoad();

LIB_BLOCK_MATCH_EXPORT
void atExit();