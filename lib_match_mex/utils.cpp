#include "lib_match_mex_common.h"
#include <stdarg.h>
#include <type_traits>
#include <memory>

LibMatchMexErrorWithMessage generateErrorMessage(LibMatchMexError error, char message[LIB_MATCH_MEX_MAX_MESSAGE_LENGTH], ...)
{
	LibMatchMexErrorWithMessage errorWithMessage = { error, "" };
	va_list args;
	va_start(args, message);
	snprintf(errorWithMessage.message, LIB_MATCH_MEX_MAX_MESSAGE_LENGTH, message, args);
	va_end(args);
	return errorWithMessage;
}

template <typename Type1, typename Type2, typename std::enable_if<!std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n)
{
	for (size_t i = 0; i<n; i++)
	{
		out[i] = in[i];
	}
}

template <typename Type1, typename Type2, typename std::enable_if<std::is_same<Type1, Type2>::value>::type * = nullptr>
void convertArrayType(const Type1 *in, Type2 *out, size_t n)
{
	memcpy(out, in, n * sizeof(Type1));
}

LibMatchMexErrorWithMessage internalErrorMessage()
{
	return generateErrorMessage(LibMatchMexError::errorInternal, "Unknown internal error.");
}

template
void convertArrayType(const float*, double *, size_t);
template
void convertArrayType(const double*, float *, size_t);
template
void convertArrayType(const int*, double *, size_t);
template
void convertArrayType(const int*, float *, size_t);
template
void convertArrayType(const float*, float *, size_t);
template
void convertArrayType(const double*, double *, size_t);