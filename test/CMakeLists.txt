project(test)

file(GLOB SRC_LIST
        "*.h"
        "*.cpp"
        )

include_directories(../include ../spdlog/include)

add_executable(test ${SRC_LIST})

target_link_libraries(test boost_unit_test_framework lib_match)