#ifdef _MSC_VER

#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h>
#include <Dbghelp.h>
#include <string>
#include <limits>
#include <mutex>

std::mutex stack_trace_locker;
HANDLE hProcess = GetCurrentProcess();

std::string getStackTrace()
{
	std::string message;
	SYMBOL_INFO *symbol = (SYMBOL_INFO*)malloc(sizeof(SYMBOL_INFO) + MAX_SYM_NAME);
	memset(symbol, 0, sizeof(SYMBOL_INFO));

	symbol->MaxNameLen = MAX_SYM_NAME;
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

	stack_trace_locker.lock();
	static bool stack_trace_initialized = false;
	if (!stack_trace_initialized)
	{
		if (!SymInitialize(hProcess, nullptr, TRUE))
		{
			message += ("Warning! SymInitialize() failed with "
			"Win32 Error Code: " + std::to_string(GetLastError()) + "\n");			
		}
		stack_trace_initialized = true;
	}
	const unsigned short max_frames = std::numeric_limits<unsigned short>::max();
	void *stacks[max_frames];
	
	unsigned short frames = CaptureStackBackTrace(1, max_frames, stacks, nullptr);
	const unsigned address_buffer_length = sizeof(ptrdiff_t) * 2 + 1;
	char buffer[address_buffer_length];

	for (unsigned short i = 0;i<frames;++i)
	{
		if (!SymFromAddr(hProcess, (DWORD64)stacks[i], 0, symbol))
		{
			symbol->Name[0] = '\0';
		}
		snprintf(buffer, address_buffer_length, "%llx", symbol->Address);
		message += std::to_string(i) + "\t0x" + buffer + "\t" + (symbol->Name[0] == '\0' ? "(unknown)" : symbol->Name) + "\n";
	}
	stack_trace_locker.unlock();

	free(symbol);

	return message;
}

#elif __unix__

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdlib.h>
#include <string>
#include <stdint.h>

#define MAX_SYM_NAME 2000

std::string getStackTrace()
{
	std::string message;
	unw_cursor_t cursor;
	unw_context_t context;

	int rc;
	// Initialize cursor to current frame for local unwinding.
	rc = unw_getcontext(&context);
	if (rc!=0)
	{
		message += "Warning! unw_getcontext() failed. rc: " + std::to_string(rc) + ", expected: 0.\n";
		return message;
	}
	rc = unw_init_local(&cursor, &context);
	if (rc != 0)
	{
		message += "Warning! unw_init_local() failed. rc: " + std::to_string(rc) + ", expected: 0.\n";
		return message;
	}

	const unsigned address_buffer_length = sizeof(ptrdiff_t) * 2 + 1;
	char buffer[address_buffer_length];

	char sym_name[MAX_SYM_NAME];

	uint32_t stack_index = 0;
	if (unw_step(&cursor) <= 0)
		return message;

	// Unwind frames one by one, going up the frame stack.
	while (unw_step(&cursor) > 0) {
		unw_word_t offset, pc;

		rc = unw_get_reg(&cursor, UNW_REG_IP, &pc);

		if (rc != 0)
		{
			message += "Warning! unw_get_reg() failed. rc: " + std::to_string(rc) + ", expected: 0.\n";
			return message;
		}

		if (pc == 0)
			break;

		snprintf(buffer, address_buffer_length, "%llx", pc);

		rc = unw_get_proc_name(&cursor, sym_name, MAX_SYM_NAME, &offset);

		message += std::to_string(stack_index) + "\t0x" + buffer + "\t" + (rc ? "(unknown)" : sym_name) + "\n";
		++stack_index;
	}

	return message;
}

#endif