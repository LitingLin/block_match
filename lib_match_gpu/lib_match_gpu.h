#pragma once

#ifdef LIB_MATCH_GPU_BUILD_DLL
#define LIB_MATCH_GPU_EXPORT __declspec(dllexport)
#else
#define LIB_MATCH_GPU_EXPORT __declspec(dllimport)
#endif

#include <cuda_runtime.h>

template <typename Type>
LIB_MATCH_GPU_EXPORT
cudaError_t lib_match_mse(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
LIB_MATCH_GPU_EXPORT
cudaError_t lib_match_mse_check_border(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
LIB_MATCH_GPU_EXPORT
cudaError_t lib_match_cc(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
LIB_MATCH_GPU_EXPORT
cudaError_t lib_match_cc_check_border(const Type *blocks_A, const Type *blocks_B, const int numBlocks_A,
	const int numberOfBlockBPerBlockA, const int blockSize, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream);

template <typename Type>
LIB_MATCH_GPU_EXPORT
cudaError_t lib_match_mse_global(const Type *A, const Type *B, const int numberOfA,
	const int numberOfBPerA, const int size, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream);
template <typename Type>
LIB_MATCH_GPU_EXPORT
cudaError_t lib_match_cc_global(const Type *A, const Type *B, const int numberOfA,
	const int numberOfBPerA, const int size, Type *result, const int numProcessors, const int numThreads, const cudaStream_t stream);