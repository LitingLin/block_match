project(lib_match_mex)

set(SRC_LIST initialize.cpp interrupt_detector.cpp logging.cpp parsing.cpp utils.cpp)

add_library(lib_match_mex STATIC ${SRC_LIST})

target_link_libraries(lib_match_mex lib_match ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})