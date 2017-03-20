#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h>
#include <Dbghelp.h>
#include <string>
#include <limits>
#include <mutex>
#include <iostream>

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
			message += ("Warning! SymInitialize() failed.\n"
			"Win32 Error Code: " + std::to_string(GetLastError()) + "\n");			
		}
	}
	const unsigned short max_frames = std::numeric_limits<unsigned short>::max();
	void *stacks[max_frames];
	
	unsigned short frames = CaptureStackBackTrace(1, max_frames, stacks, nullptr);
	for (unsigned short i = 0;i<frames;++i)
	{
		const unsigned address_buffer_length = sizeof(ptrdiff_t) * 2 + 1;
		char buffer[address_buffer_length];
		if (!SymFromAddr(hProcess, (DWORD64)stacks[i], 0, symbol))
		{
			message += "Warning! SymFromAddr() failed.\n"
				"Win32 Error Code: " + std::to_string(GetLastError()) + "\n";
		}
		snprintf(buffer, address_buffer_length, "%llx", symbol->Address);
		message += std::to_string(i) + "\t0x" + buffer + "\t" + (symbol->Name[0] == '\0' ? "(unknown)" : symbol->Name) + "\n";
	}
	stack_trace_locker.unlock();

	free(symbol);

	return message;
}


int main()
{
	std::cout << getStackTrace();
	system("pause");
}

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdio.h>

// Call this function to get a backtrace.
void backtrace() {
	unw_cursor_t cursor;
	unw_context_t context;

	// Initialize cursor to current frame for local unwinding.
	unw_getcontext(&context);
	unw_init_local(&cursor, &context);

	// Unwind frames one by one, going up the frame stack.
	while (unw_step(&cursor) > 0) {
		unw_word_t offset, pc;
		unw_get_reg(&cursor, UNW_REG_IP, &pc);
		if (pc == 0) {
			break;
		}
		printf("0x%lx:", pc);

		char sym[256];
		if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
			printf(" (%s+0x%lx)\n", sym, offset);
		}
		else {
			printf(" -- error: unable to obtain symbol name for this frame\n");
		}
	}
}

void foo() {
	backtrace(); // <-------- backtrace here!
}

void bar() {
	foo();
}

int main(int argc, char **argv) {
	bar();

	return 0;
}