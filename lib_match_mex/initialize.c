#include "lib_match_mex_internal.h"
#include "lib_match_mex_common.h"

bool isLoaded = false;

void libMatchMexInitalize()
{
	if (!isLoaded) {
		onLoad();
		registerLoggingSinkFunction(logging_function);
		mexAtExit(atExit);
		isLoaded = true;
	}
}
