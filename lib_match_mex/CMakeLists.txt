project(lib_match_mex)

file(GLOB SRC_LIST
        "*.h"
        "*.cpp"
        )

list(APPEND SRC_LIST ../include/lib_match_mex_common.h)

include_directories(../include ../spdlog/include)

add_library(lib_match_mex STATIC ${SRC_LIST})

target_link_libraries(lib_match_mex lib_match ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})