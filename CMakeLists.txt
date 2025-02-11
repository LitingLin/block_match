cmake_minimum_required(VERSION 2.6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if (MSVC)
    set(APPEND CMAKE_CXX_FLAGS /Zc:throwingNew /MP)
    set(APPEND CMAKE_C_FLAGS /MP)
endif()

find_package(CUDA REQUIRED)

find_package(Matlab COMPONENTS MX_LIBRARY)

add_subdirectory(lib_match)
if(Matlab_FOUND)
    add_subdirectory(lib_match_mex)
    add_subdirectory(block_match_mex)
    add_subdirectory(array_match_mex)
    add_subdirectory(pad_2d_mex)
endif()

find_package (Boost COMPONENTS unit_test_framework)
if (Boost_UNIT_TEST_FRAMEWORK_FOUND)
    enable_testing()
    add_test(TEST test COMMAND test)
    add_test(TEST unit_test COMMAND unit_test)
endif()