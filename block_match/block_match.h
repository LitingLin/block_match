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
bool initialize(void **instance, size_t matA_M, size_t matA_N, size_t matB_M, size_t matB_N, size_t block_M, size_t block_N, size_t neighbour_M, size_t neighbour_N, size_t stride_M, size_t stride_N);
LIB_BLOCK_MATCH_EXPORT
bool process(void *instance, float *matA, float *matB, enum Method method);
LIB_BLOCK_MATCH_EXPORT
void getResult(void *instance, float **result, size_t *result_dim0, size_t *result_dim1, size_t *result_dim2, size_t *result_dim3);
LIB_BLOCK_MATCH_EXPORT
void finalize(void *instance);
LIB_BLOCK_MATCH_EXPORT
bool reset();

LIB_BLOCK_MATCH_EXPORT
void onLoad();
LIB_BLOCK_MATCH_EXPORT
void atExit();