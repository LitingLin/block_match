#include "lib_match.h"
#include "lib_match_internal.h"

bool libMatchReset()
{
	cudaError_t cuda_error = cudaDeviceReset();
	return cuda_error == cudaSuccess;
}

void libMatchOnLoad(void)
{
}

void libMatchAtExit(void)
{
}
