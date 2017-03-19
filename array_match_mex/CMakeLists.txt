project(array_match_mex)

set(SRC_LIST mexFunction.cpp parse_parameter.cpp utils.cpp)

add_library(array_match_mex SHARED ${SRC_LIST})
set_target_properties(array_match_mex PROPERTIES SUFFIX ${Matlab_MEX_EXTENSION})
target_link_libraries(array_match_mex lib_match_mex lib_match)