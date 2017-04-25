#include "lib_match_internal.h"

/* 
 * RAII warpers
 */

cudaStream_guard::cudaStream_guard()
{
	CUDA_CHECK_POINT(cudaStreamCreate(&stream));
}

cudaStream_guard::~cudaStream_guard()
{
	CUDA_CHECK_POINT(cudaStreamDestroy(stream));
}

cudaStream_guard::operator cudaStream_t() const
{
	return stream;
}

