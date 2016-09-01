#include "common.h"

#include <stdarg.h>

struct ArrayMatchMexErrorWithMessage generateErrorMessage(enum ArrayMatchMexError error, char message[ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH], ...)
{
	struct ArrayMatchMexErrorWithMessage errorWithMessage = { error, "" };
	va_list args;
	va_start(args, message);
	snprintf(errorWithMessage.message, ARRAY_MATCH_MEX_MAX_MESSAGE_LENGTH, message, args);
	va_end(args);
	return errorWithMessage;
}

void convertDoubleToFloat(const double *source, float *destination, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		destination[i] = source[i];
	}
}

size_t conjectureMaximumMemoryAllocationSize(int lengthOfArray, int numberOfArray)
{
	return (lengthOfArray * numberOfArray * 2 + numberOfArray) * sizeof(float);
}