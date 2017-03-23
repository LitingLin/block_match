#include <lib_match.h>

void instruction_set_check()
{
#ifdef __AVX__
	if (!InstructionSet::AVX())
		throw std::runtime_error("The processor has no AVX Instruction Set supoort. Please re-compile with no AVX support");
#endif
#ifdef __AVX2__
	if (!InstructionSet::AVX2())
		throw std::runtime_error("The processor has no AVX2 Instruction Set supoort. Please re-compile with no AVX2 support");
#endif
}