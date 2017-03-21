#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define DLL_EXPORT_SYM __attribute__ ((dllexport))
#else
#define DLL_EXPORT_SYM __declspec(dllexport)
#endif
#else
#define DLL_EXPORT_SYM __attribute__ ((visibility ("default")))
#endif

#include <mex.h>
MEXFUNCTION_LINKAGE
void mexfilerequiredapiversion(unsigned int* built_by_rel, unsigned int* target_api_ver)
{
  *built_by_rel = 0x2016b;
  *target_api_ver = MX_TARGET_API_VER;
}