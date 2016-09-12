#include "lib_match_mex_internal.h"
#include "lib_match_mex_common.h"

bool isLoaded = false;

void libMatchMexInitalize()
{
	if (!isLoaded) {
		libMatchOnLoad();
		libMatchRegisterLoggingSinkFunction(logging_function);
		mexAtExit(libMatchAtExit);
		isLoaded = true;
	}
}
