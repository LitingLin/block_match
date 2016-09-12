#include "lib_match.h"
#include "lib_match_internal.h"

extern "C"
bool reset()
{
	cudaError_t cuda_error = cudaDeviceReset();
	return cuda_error == cudaSuccess;
}
extern "C"
void onLoad(void)
{
}

extern "C"
void atExit(void)
{
	globalContext.pool.shutdown();
}
