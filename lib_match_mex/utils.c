#include "lib_match_mex_common.h"
#include <stdarg.h>

struct LibMatchMexErrorWithMessage generateErrorMessage(enum ArrayMatchMexError error, char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH], ...)
{
	struct LibMatchMexErrorWithMessage errorWithMessage = { error, "" };
	va_list args;
	va_start(args, message);
	snprintf(errorWithMessage.message, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, message, args);
	va_end(args);
	return errorWithMessage;
}

void convertArrayFromDoubleToFloat(const double *source, float *destination, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		destination[i] = source[i];
	}
}

void convertArrayFromFloatToDouble(const float *source, double *destination, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		destination[i] = source[i];
	}
}