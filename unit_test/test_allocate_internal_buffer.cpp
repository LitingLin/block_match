#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_allocate_internal_buffer)
{
	const int numberOfThreads = 4;
	BlockMatchContext *context = allocateContext(4);

	context->C_dimensions[0] = context->C_dimensions[1] = context->C_dimensions[2] = 4;

	BOOST_CHECK(allocateInternalBuffer(context, InternalBufferType::Index_X_Internal));

	fillWithSequence(context->optionalBuffer.matrixA_padded_internal, numberOfThreads);
	BOOST_CHECK_EQUAL(context->optionalPerThreadBufferPointer[0].index_x_internal[0],
		context->optionalBuffer.matrixA_padded_internal[0]);

	BOOST_CHECK_EQUAL(context->optionalPerThreadBufferPointer[1].index_x_internal[0],
		context->optionalBuffer.matrixA_padded_internal[numberOfThreads * 1]);

	free(context);
}