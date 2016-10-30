#include "lib_match_mex_internal.h"
#include "lib_match_mex_common.h"

bool isLoaded = false;

void libMatchMexInitalize()
{
	if (!isLoaded) {
		disableInterruptHandle();
		libMatchOnLoad();
		libMatchRegisterLoggingSinkFunction(logging_function);
		libMatchRegisterInterruptPeddingFunction(libMatchMexIsInterruptPendingFunction);
		mexAtExit(libMatchAtExit);
		isLoaded = true;
	}
}
