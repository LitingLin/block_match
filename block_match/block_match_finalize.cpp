#include "block_match.h"

#include "block_match_internal.h"

extern "C"
void finalize(void *_instance)
{
	Context *instance = (Context *)_instance;

	const unsigned numberOfThreads = globalContext.numberOfThreads;

	for (unsigned i = 0; i < numberOfThreads; ++i) {
		cudaStreamDestroy(instance->stream[i]);
	}

	cudaFreeHost(instance->buffer_A);
	cudaFree(instance->device_buffer_A);

	delete[] instance->stream;
	delete[] instance->index;

	free(instance);
}