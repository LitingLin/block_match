project(lib_match)
set(SRC_LIST array_match_execute.cpp array_match_finalize.cpp array_match_initialize.cpp
        block_match_cc_cpu.cpp block_match_execution_launcher.cpp block_match_finalize.cpp block_match_initialize.cpp
        block_match_mse_cpu.cpp block_match_sort.cpp block_padding.cpp interrupt_handler.cpp lib_match.cpp
        lib_match_internal.cpp logging.cpp memory_allocation_counter.cpp stack_trace.cpp thread_pool.cpp)

add_library(lib_match ${SRC_LIST})