#define WIN32_MEAN_AND_LEAN
#include <windows.h>

#include <block_match.h>

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
		break;
	case DLL_PROCESS_DETACH:
		atExit();
		break;
	default:
		break;
	}
	return TRUE;
}