cmake_minimum_required(VERSION 3.23)

project(OpenZoom LANGUAGES CXX)

if(OPENZOOM_ENABLE_TESTS)
    enable_testing()
endif()

add_subdirectory(cmake)
