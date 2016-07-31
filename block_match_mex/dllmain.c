#define VC_EXTRALEAN
#define WIN32_MEAN_AND_LEAN
#include <windows.h>

#include <block_match.h>
#include "common.h"

BOOL WINAPI DllMain(
	_In_ HINSTANCE hinstDLL,
	_In_ DWORD     fdwReason,
	_In_ LPVOID    lpvReserved
)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		onLoad();
		registerLoggingSinkFunction(logging_function);
		break;
	case DLL_PROCESS_DETACH:
		break;
	default:
		break;
	}
	return TRUE;
}