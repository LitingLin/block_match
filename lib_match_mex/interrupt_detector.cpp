#pragma comment(lib, "libut.lib")
extern bool utIsInterruptPending();

#ifdef _MSC_VER
	
#endif

bool libMatchMexIsInterruptPendingFunction()
{
	return utIsInterruptPending();
}