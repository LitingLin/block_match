#include "lib_match_internal.h"

/* 
 * RAII warpers
 */

cudaStreamWarper::cudaStreamWarper()
{
	CUDA_CHECK_POINT(cudaStreamCreate(&stream));
}

cudaStreamWarper::~cudaStreamWarper()
{
	CUDA_CHECK_POINT(cudaStreamDestroy(stream));
}

cudaStreamWarper::operator cudaStream_t() const
{
	return stream;
}

