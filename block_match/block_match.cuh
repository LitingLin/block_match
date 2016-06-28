#pragma once

cudaError_t block_match_mse(const float *block_A, const float *block_B, size_t numBlock_A, size_t numBlock_B, size_t blockSize, float *result, cudaStream_t stream);


cudaError_t block_match_mse_async(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_mse_async(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream);

cudaError_t block_match_cc(float * block_A, float * block_B, size_t numBlock_A, size_t numBlock_B, size_t blockSize, float * result, cudaStream_t stream);

cudaError_t block_match_cc_async(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, cudaStream_t stream);
cudaError_t block_match_cc_async(float *blocks_A, float *blocks_B, size_t numBlocks_A, size_t numBlocks_B, size_t block_B_groupSize, size_t blockSize, float *result, int numProcessors, int numThreads, size_t numTasks, cudaStream_t stream);