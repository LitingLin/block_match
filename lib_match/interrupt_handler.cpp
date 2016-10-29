#include "lib_match_internal.h"

LibMatchInterruptPendingFunction *interruptPendingFunction;

bool isInterruptPending()
{
	return interruptPendingFunction();
}

void libMatchRegisterInterruptPeddingFunction(LibMatchInterruptPendingFunction function)
{
	interruptPendingFunction = function;
}