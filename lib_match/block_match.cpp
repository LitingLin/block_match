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


template <typename Type>
BlockMatch<Type>::Diagnose::Diagnose(void* instance)
	: m_instance(instance)
{
}
template <typename Type>
double BlockMatch<Type>::Diagnose::getFinishedPercentage()
{
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

LIB_MATCH_EXPORT
template
BlockMatch<float>::Diagnose::Diagnose(void* instance);
LIB_MATCH_EXPORT
template
BlockMatch<double>::Diagnose::Diagnose(void* instance);

LIB_MATCH_EXPORT
template 
double BlockMatch<float>::Diagnose::getFinishedPercentage();
LIB_MATCH_EXPORT
template
double BlockMatch<double>::Diagnose::getFinishedPercentage();