#include "lib_match.h"

#include <intrin.h>


std::string InstructionSet::Vendor()
{
	return CPU_Rep.vendor_;
}

std::string InstructionSet::Brand()
{
	return CPU_Rep.brand_;
}

bool InstructionSet::SSE3()
{
	return CPU_Rep.f_1_ECX_[0];
}

bool InstructionSet::PCLMULQDQ()
{
	return CPU_Rep.f_1_ECX_[1];
}

bool InstructionSet::MONITOR()
{
	return CPU_Rep.f_1_ECX_[3];
}

bool InstructionSet::SSSE3()
{
	return CPU_Rep.f_1_ECX_[9];
}

bool InstructionSet::FMA()
{
	return CPU_Rep.f_1_ECX_[12];
}

bool InstructionSet::CMPXCHG16B()
{
	return CPU_Rep.f_1_ECX_[13];
}

bool InstructionSet::SSE41()
{
	return CPU_Rep.f_1_ECX_[19];
}

bool InstructionSet::SSE42()
{
	return CPU_Rep.f_1_ECX_[20];
}

bool InstructionSet::MOVBE()
{
	return CPU_Rep.f_1_ECX_[22];
}

bool InstructionSet::POPCNT()
{
	return CPU_Rep.f_1_ECX_[23];
}

bool InstructionSet::AES()
{
	return CPU_Rep.f_1_ECX_[25];
}

bool InstructionSet::XSAVE()
{
	return CPU_Rep.f_1_ECX_[26];
}

bool InstructionSet::OSXSAVE()
{
	return CPU_Rep.f_1_ECX_[27];
}

bool InstructionSet::AVX()
{
	return CPU_Rep.f_1_ECX_[28];
}

bool InstructionSet::F16C()
{
	return CPU_Rep.f_1_ECX_[29];
}

bool InstructionSet::RDRAND()
{
	return CPU_Rep.f_1_ECX_[30];
}

bool InstructionSet::MSR()
{
	return CPU_Rep.f_1_EDX_[5];
}

bool InstructionSet::CX8()
{
	return CPU_Rep.f_1_EDX_[8];
}

bool InstructionSet::SEP()
{
	return CPU_Rep.f_1_EDX_[11];
}

bool InstructionSet::CMOV()
{
	return CPU_Rep.f_1_EDX_[15];
}

bool InstructionSet::CLFSH()
{
	return CPU_Rep.f_1_EDX_[19];
}

bool InstructionSet::MMX()
{
	return CPU_Rep.f_1_EDX_[23];
}

bool InstructionSet::FXSR()
{
	return CPU_Rep.f_1_EDX_[24];
}

bool InstructionSet::SSE()
{
	return CPU_Rep.f_1_EDX_[25];
}

bool InstructionSet::SSE2()
{
	return CPU_Rep.f_1_EDX_[26];
}

bool InstructionSet::FSGSBASE()
{
	return CPU_Rep.f_7_EBX_[0];
}

bool InstructionSet::BMI1()
{
	return CPU_Rep.f_7_EBX_[3];
}

bool InstructionSet::HLE()
{
	return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4];
}

bool InstructionSet::AVX2()
{
	return CPU_Rep.f_7_EBX_[5];
}

bool InstructionSet::BMI2()
{
	return CPU_Rep.f_7_EBX_[8];
}

bool InstructionSet::ERMS()
{
	return CPU_Rep.f_7_EBX_[9];
}

bool InstructionSet::INVPCID()
{
	return CPU_Rep.f_7_EBX_[10];
}

bool InstructionSet::RTM()
{
	return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11];
}

bool InstructionSet::AVX512F()
{
	return CPU_Rep.f_7_EBX_[16];
}

bool InstructionSet::RDSEED()
{
	return CPU_Rep.f_7_EBX_[18];
}

bool InstructionSet::ADX()
{
	return CPU_Rep.f_7_EBX_[19];
}

bool InstructionSet::AVX512PF()
{
	return CPU_Rep.f_7_EBX_[26];
}

bool InstructionSet::AVX512ER()
{
	return CPU_Rep.f_7_EBX_[27];
}

bool InstructionSet::AVX512CD()
{
	return CPU_Rep.f_7_EBX_[28];
}

bool InstructionSet::SHA()
{
	return CPU_Rep.f_7_EBX_[29];
}

bool InstructionSet::PREFETCHWT1()
{
	return CPU_Rep.f_7_ECX_[0];
}

bool InstructionSet::LAHF()
{
	return CPU_Rep.f_81_ECX_[0];
}

bool InstructionSet::LZCNT()
{
	return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5];
}

bool InstructionSet::ABM()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5];
}

bool InstructionSet::SSE4a()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6];
}

bool InstructionSet::XOP()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11];
}

bool InstructionSet::TBM()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21];
}

bool InstructionSet::SYSCALL()
{
	return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11];
}

bool InstructionSet::MMXEXT()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22];
}

bool InstructionSet::RDTSCP()
{
	return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27];
}

bool InstructionSet::_3DNOWEXT()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30];
}

bool InstructionSet::_3DNOW()
{
	return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31];
}

InstructionSet::InstructionSet_Internal::InstructionSet_Internal() : nIds_{ 0 },
nExIds_{ 0 },
isIntel_{ false },
isAMD_{ false },
f_1_ECX_{ 0 },
f_1_EDX_{ 0 },
f_7_EBX_{ 0 },
f_7_ECX_{ 0 },
f_81_ECX_{ 0 },
f_81_EDX_{ 0 },
data_{},
extdata_{}
{
	//int cpuInfo[4] = {-1};  
	std::array<int, 4> cpui;

	// Calling __cpuid with 0x0 as the function_id argument  
	// gets the number of the highest valid function ID.  
	__cpuid(cpui.data(), 0);
	nIds_ = cpui[0];

	for (int i = 0; i <= nIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		data_.push_back(cpui);
	}

	// Capture vendor string  
	char vendor[0x20];
	memset(vendor, 0, sizeof(vendor));
	*reinterpret_cast<int*>(vendor) = data_[0][1];
	*reinterpret_cast<int*>(vendor + 4) = data_[0][3];
	*reinterpret_cast<int*>(vendor + 8) = data_[0][2];
	vendor_ = vendor;
	if (vendor_ == "GenuineIntel")
	{
		isIntel_ = true;
	}
	else if (vendor_ == "AuthenticAMD")
	{
		isAMD_ = true;
	}

	// load bitset with flags for function 0x00000001  
	if (nIds_ >= 1)
	{
		f_1_ECX_ = data_[1][2];
		f_1_EDX_ = data_[1][3];
	}

	// load bitset with flags for function 0x00000007  
	if (nIds_ >= 7)
	{
		f_7_EBX_ = data_[7][1];
		f_7_ECX_ = data_[7][2];
	}

	// Calling __cpuid with 0x80000000 as the function_id argument  
	// gets the number of the highest valid extended ID.  
	__cpuid(cpui.data(), 0x80000000);
	nExIds_ = cpui[0];

	char brand[0x40];
	memset(brand, 0, sizeof(brand));

	for (int i = 0x80000000; i <= nExIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		extdata_.push_back(cpui);
	}

	// load bitset with flags for function 0x80000001  
	if (nExIds_ >= 0x80000001)
	{
		f_81_ECX_ = extdata_[1][2];
		f_81_EDX_ = extdata_[1][3];
	}

	// Interpret CPU brand string if reported  
	if (nExIds_ >= 0x80000004)
	{
		memcpy(brand, extdata_[2].data(), sizeof(cpui));
		memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
		memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
		brand_ = brand;
	}
}

// Initialize static member data  
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;