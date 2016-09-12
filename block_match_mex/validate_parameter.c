#include "common.h"

struct LibMatchMexErrorWithMessage validateParameter(struct LibBlockMatchMexContext *context)
{
	if (!context->sort && context->retain)
		return generateErrorMessage(libMatchMexErrorInvalidValue, "Argument Retain cannot be positive integer when argument Sort is not true");
	
	return generateErrorMessage(libMatchMexOk, "");
}