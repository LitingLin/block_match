#include "lib_match_internal.h"

template <typename Type>
void blockMatchFinalize(void *_instance)
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(_instance);
	delete instance;
}

template
void blockMatchFinalize<float>(void *_instance);
template
void blockMatchFinalize<double>(void *_instance);