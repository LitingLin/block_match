project(lib_match)

file(GLOB SRC_LIST
        "*.h"
        "*.cpp"
        )

file(GLOB CUDA_SRC_LIST
        "*.cuh"
        "*.cu")

list(APPEND SRC_LIST ../include/lib_match.h)

include_directories(${CUDA_INCLUDE_DIRS} ../include ../spdlog/include)

add_library(lib_match STATIC ${SRC_LIST} ${CUDA_SRC_LIST})
target_link_libraries(lib_match ${CUDA_CUDART_LIBRARY})