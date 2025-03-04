cmake_minimum_required(VERSION 3.19)

project(
    moonshine_cpp
    LANGUAGES CXX
    VERSION 0.1.0
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(src)

# only add the example directory if we are building the project standalone
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(example)
endif()
