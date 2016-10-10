#include "test_common.h"

void fillWithSequence(float *ptr, size_t size)
{
	for (size_t i = 0; i < size; ++size)
		ptr[i] = i;
}

BOOST_AUTO_TEST_CASE(test_allocate_internal_buffer)
{
	const int numberOfThreads = 4;
	BlockMatchContext *context = allocateContext(4);

	context->matrixA_padded_M = 4;
	context->matrixA_padded_N = 4;

	BOOST_CHECK(allocateInternalBuffer(context, InternalBufferType::MatrixA_Padded_Buffer));

	fillWithSequence(context->optionalBuffer.matrixA_padded_internal, numberOfThreads);
	BOOST_CHECK_EQUAL(context->optionalPerThreadBufferPointer.matrixA_padded_internal[0][0],
		context->optionalBuffer.matrixA_padded_internal[0]);

	BOOST_CHECK_EQUAL(context->optionalPerThreadBufferPointer.matrixA_padded_internal[1][0],
		context->optionalBuffer.matrixA_padded_internal[numberOfThreads * 1]);

	free(context);
}