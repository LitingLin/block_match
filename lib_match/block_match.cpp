#include "lib_match.h"

template <typename Type>
BlockMatch<Type>::BlockMatch()
	: m_instance(nullptr)
{	
}


template <typename Type>
BlockMatch<Type>::~BlockMatch()
{
	if (m_instance)
		destroy();
}

LIB_MATCH_EXPORT
template
BlockMatch<float>::BlockMatch();
LIB_MATCH_EXPORT
template
BlockMatch<double>::BlockMatch();

LIB_MATCH_EXPORT
template
BlockMatch<float>::~BlockMatch();
LIB_MATCH_EXPORT
template
BlockMatch<double>::~BlockMatch();
