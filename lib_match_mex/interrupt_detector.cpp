#ifdef _MSC_VER
#pragma comment(lib, "libut.lib")
#endif
extern "C" bool utSetInterruptEnabled(bool);
extern "C" bool utIsInterruptPending();
extern "C" bool utSetInterruptHandled(bool);

void disableInterruptHandle()
{
	utSetInterruptEnabled(false);
}

bool libMatchMexIsInterruptPendingFunction()
{
	utSetInterruptHandled(true);
	return utIsInterruptPending();
}