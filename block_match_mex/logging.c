#include <mex.h>

void logging_function(const char* msg)
{
	mexPrintf(msg);
}