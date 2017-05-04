#include <memory>
#include <type_traits>
#include "template_instantiate_helper.h"

template <typename Type1, typename Type2,
	typename std::enable_if<std::is_same<Type1, Type2>::value>::type* = nullptr>
	void copyArray_helper(Type1 *buf, const Type2 *src, int size)
{
	memcpy(buf, src, size * sizeof(Type1));
}

#ifdef _MSC_VER
#pragma warning( disable : 4800 )  
#endif
template <typename Type1, typename Type2,
	typename std::enable_if<!std::is_same<Type1, Type2>::value>::type* = nullptr>
	void copyArray_helper(Type1 *buf, const Type2 *src, int size)
{
	for (int i = 0; i < size; ++i)
	{
		*buf++ = static_cast<Type1>(*src++);
	}
}
#ifdef _MSC_VER
#pragma warning( default : 4800 )  
#endif

// Workaround: MSVC SFINAE function pointer type cast bug
template <typename Type1, typename Type2>
void copyArray(Type1 *buf, const Type2 *src, int size)
{
	copyArray_helper(buf, src, size);
}

#define EXP(type1, type2) \
template \
void copyArray(type1 *, const type2 *, int);
InstantiateTemplate2(EXP);
#undef EXP
