#include "lib_match.h"
#include "lib_match_internal.h"

extern "C"
bool libMatchReset()
{
	cudaError_t cuda_error = cudaDeviceReset();
	return cuda_error == cudaSuccess;
}
extern "C"
void libMatchOnLoad(void)
{
}

extern "C"
void libMatchAtExit(void)
{
	globalContext.pool.shutdown();
}
