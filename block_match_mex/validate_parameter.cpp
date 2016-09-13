#include "common.h"

struct LibMatchMexErrorWithMessage validateParameter(struct BlockMatchMexContext *context)
{
	if (!context->sort && context->retain)
		return generateErrorMessage(LibMatchMexError::errorInvalidValue, "Argument Retain cannot be positive integer when argument Sort is not true");
	
	return generateErrorMessage(LibMatchMexError::success, "");
}