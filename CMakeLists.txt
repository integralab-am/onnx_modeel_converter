cmake_minimum_required(VERSION 3.15)
project(FlatVR_ModelConverter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(model_converter model_converter.cpp)

if(WIN32)
    # Link urlmon just in case downloading is needed in the future
    target_link_libraries(model_converter PRIVATE urlmon)
endif()
