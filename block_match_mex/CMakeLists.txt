project(block_match_mex)

set(SRC_LIST generate_result.cpp mexFunction.cpp parse_parameter.cpp utils.cpp validate_parameter.cpp)

add_library(block_match_mex SHARED ${SRC_LIST})
set_target_properties(block_match_mex PROPERTIES SUFFIX ${Matlab_MEX_EXTENSION})
target_link_libraries(block_match_mex lib_match_mex lib_match)