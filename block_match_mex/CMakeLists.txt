project(block_match_mex)

file(GLOB SRC_LIST
        "*.h"
        "*.cpp"
        )

include_directories(../include ../spdlog/include)

add_library(block_match_mex SHARED ${SRC_LIST})
set_target_properties(block_match_mex PROPERTIES SUFFIX ${Matlab_MEX_EXTENSION})
target_link_libraries(block_match_mex lib_match_mex lib_match)