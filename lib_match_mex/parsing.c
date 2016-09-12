#include "lib_match_mex_common.h"

enum LibMatchMexError getStringFromMxArray(const mxArray *pa, char *buffer, int bufferLength)
{
	mxClassID classId = mxGetClassID(pa);
	if (classId != mxCHAR_CLASS)
		return libMatchMexErrorTypeOfArgument;

	mxGetNChars(pa, buffer, bufferLength);

	return libMatchMexOk;
}