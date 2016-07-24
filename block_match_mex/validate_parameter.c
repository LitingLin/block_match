#include "common.h"

struct LibBlockMatchMexErrorWithMessage validateParameter(struct LibBlockMatchMexContext *context)
{
	if (!context->sort && context->retain)
		return generateErrorMessage(blockMatchMexErrorInvalidValue, "Argument Retain cannot be positive integer when argument Sort is not true");
	
	return generateErrorMessage(blockMatchMexOk, "");
}