project(pad_2d_mex)

set(SRC_LIST mexFunction.cpp parse_parameter.cpp)

add_library(pad_2d_mex SHARED ${SRC_LIST})
set_target_properties(pad_2d_mex SUFFIX mex_suffix)
target_link_libraries(pad_2d_mex lib_match_mex lib_match)