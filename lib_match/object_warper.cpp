#include "lib_match_internal.h"

cudaStreamWarper::cudaStreamWarper()
{
	CUDA_CHECK_POINT(cudaStreamCreate(&stream));
}

cudaStreamWarper::~cudaStreamWarper()
{
	CUDA_CHECK_POINT(cudaStreamDestroy(stream));
}

cudaStreamWarper::operator cudaStream_t()
{
	return stream;
}

