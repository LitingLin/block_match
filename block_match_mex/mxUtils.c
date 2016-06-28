#include "common.h"

#include "mxUtils.h"

bool mxTypeConvert(const void *l, mxClassID lClass, void *r, mxClassID rClass)
{
	if (lClass == mxINT8_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(int8_t*)r;
	}
	else if (lClass == mxINT16_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(int16_t*)r;
	}
	else if (lClass == mxINT32_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(int32_t*)r;
	}
	else if (lClass == mxINT64_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(int64_t*)r;
	}
	else if (lClass == mxUINT8_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(uint8_t*)r;
	}
	else if (lClass == mxUINT16_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(uint16_t*)r;
	}
	else if (lClass == mxUINT32_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(uint32_t*)r;
	}
	else if (lClass == mxUINT64_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(uint64_t*)r;
	}
	else if (lClass == mxSINGLE_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(float*)r;
	}
	else if (lClass == mxDOUBLE_CLASS && rClass == mxUINT32_CLASS)
	{
		*(uint32_t*)r = *(double*)r;
	}
	else
		return false;
	return true;
}

bool getUInt32(const mxArray *p, uint32_t *v)
{
	if (!mxIsScalar(p) || !mxIsNumeric(p))
		return false;

	return mxTypeConvert(mxGetData(p), mxGetClassID(p), v, mxUINT32_CLASS);
}