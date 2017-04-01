#include <typeindex>
#include <stdint.h>
#include <cstring>

template <typename T1, typename T2>
void convert(T1 *src, T2 *dst, size_t size)
{
	for (size_t i=0;i<size;++i)
	{
		dst[i] = (T2)src[i];
	}
}

void convert(void *src, std::type_index src_type, void *dst, std::type_index dst_type, size_t size)
{
	if (src_type == typeid(uint8_t))
	{
		if (dst_type == typeid(uint8_t))
			memcpy(src, dst, sizeof(uint8_t) * size);
		else if (dst_type == typeid(int8_t))
			convert((uint8_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((uint8_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((uint8_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((uint8_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((uint8_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((uint8_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((uint8_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((uint8_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((uint8_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(int8_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((int8_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			memcpy(src, dst, sizeof(int8_t) * size);
		else if (dst_type == typeid(uint16_t))
			convert((int8_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((int8_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((int8_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((int8_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((int8_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((int8_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((int8_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((int8_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(uint16_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((uint16_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((uint16_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			memcpy(src, dst, sizeof(uint16_t) * size);
		else if (dst_type == typeid(int16_t))
			convert((uint16_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((uint16_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((uint16_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((uint16_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((uint16_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((uint16_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((uint16_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(int16_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((int16_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((int16_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((int16_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			memcpy(src, dst, sizeof(int16_t) * size);
		else if (dst_type == typeid(uint32_t))
			convert((int16_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((int16_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((int16_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((int16_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((int16_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((int16_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(uint32_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((uint32_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((uint32_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((uint32_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((uint32_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			memcpy(src, dst, sizeof(uint32_t) * size);
		else if (dst_type == typeid(int32_t))
			convert((uint32_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((uint32_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((uint32_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((uint32_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((uint32_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(int32_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((int32_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((int32_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((int32_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((int32_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((int32_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			memcpy(src, dst, sizeof(int32_t) * size);
		else if (dst_type == typeid(uint64_t))
			convert((int32_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((int32_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((int32_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((int32_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(uint64_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((uint64_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((uint64_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((uint64_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((uint64_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((uint64_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((uint64_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			memcpy(src, dst, sizeof(uint64_t) * size);
		else if (dst_type == typeid(int64_t))
			convert((uint64_t*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((uint64_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((uint64_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(int64_t))
	{
		if (dst_type == typeid(uint8_t))
			convert((int64_t*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((int64_t*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((int64_t*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((int64_t*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((int64_t*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((int64_t*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((int64_t*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			memcpy(src, dst, sizeof(int64_t) * size);
		else if (dst_type == typeid(float))
			convert((int64_t*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			convert((int64_t*)src, (double*)dst, size);
	}
	else if (src_type == typeid(float))
	{
		if (dst_type == typeid(uint8_t))
			convert((float*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((float*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((float*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((float*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((float*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((float*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((float*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((float*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			memcpy(src, dst, sizeof(float) * size);
		else if (dst_type == typeid(double))
			convert((float*)src, (double*)dst, size);
	}
	else if (src_type == typeid(double))
	{
		if (dst_type == typeid(uint8_t))
			convert((double*)src, (uint8_t*)dst, size);
		else if (dst_type == typeid(int8_t))
			convert((double*)src, (int8_t*)dst, size);
		else if (dst_type == typeid(uint16_t))
			convert((double*)src, (uint16_t*)dst, size);
		else if (dst_type == typeid(int16_t))
			convert((double*)src, (int16_t*)dst, size);
		else if (dst_type == typeid(uint32_t))
			convert((double*)src, (uint32_t*)dst, size);
		else if (dst_type == typeid(int32_t))
			convert((double*)src, (int32_t*)dst, size);
		else if (dst_type == typeid(uint64_t))
			convert((double*)src, (uint64_t*)dst, size);
		else if (dst_type == typeid(int64_t))
			convert((double*)src, (int64_t*)dst, size);
		else if (dst_type == typeid(float))
			convert((double*)src, (float*)dst, size);
		else if (dst_type == typeid(double))
			memcpy(src, dst, sizeof(double) * size);
	}
}