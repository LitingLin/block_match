#pragma once

#include <new>
template <typename T, size_t stackBufferSize>
class StackVector
{
public:
	StackVector(size_t size)
		: m_size(size)
	{
		if (size <= stackBufferSize)
			m_isOnStack = true;
		else
		{
			m_isOnStack = false;
			ptr = new T[size];
		}
	}
	~StackVector()
	{
		if (!m_isOnStack)
			if (ptr)
				delete[]ptr;
	}
	T& operator[](size_t i)
	{
		if (m_isOnStack)
			return stackBuffer[i];
		else
			return ptr[i];
	}
	const T& operator[](size_t i) const
	{
		if (m_isOnStack)
			return stackBuffer[i];
		else
			return ptr[i];
	}
	bool bad_alloc()
	{
		if (!m_isOnStack && ptr == nullptr)
			return true;
		else
			return false;
	}
private:
	size_t m_size;
	bool m_isOnStack;
	union {
		T *ptr;
		T stackBuffer[stackBufferSize];
	};
};