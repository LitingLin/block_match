#include "common.h"

#include <string.h>

void floatToDouble(const float *in, double *out, size_t n)
{
	for (size_t i = 0; i != n; i++)
	{
		out[i] = in[i];
	}
}

void doubleToFloat(const double *in, float *out, size_t n)
{
	for (size_t i = 0; i != n; i++)
	{
		out[i] = in[i];
	}
}

void intToDouble(const int *in, double *out, size_t n)
{
	for (size_t i=0;i<n;i++)
	{
		out[i] = in[i];
	}
}

void intToDoublePlusOne(const int *in, double *out, size_t n)
{
	for (size_t i = 0; i<n; i++)
	{
		out[i] = in[i] + 1;
	}
}

struct LibBlockMatchMexErrorWithMessage generateErrorMessage(enum LibBlockMatchMexError error, char message[LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH])
{
	struct LibBlockMatchMexErrorWithMessage error_with_message = { error, "" };
	strncpy_s(error_with_message.message, LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH, message, LIB_BLOCK_MATCH_MEX_MAX_MESSAGE_LENGTH);
	return error_with_message;
}