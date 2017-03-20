project(unit_test)


file(GLOB SRC_LIST
        "*.h"
        "*.cpp"
        )

include_directories(../include ../spdlog/include)

add_executable(unit_test ${SRC_LIST})

target_link_libraries(unit_test boost_unit_test_framework lib_match)