# Project-wide configuration options and defaults.

option(OPENZOOM_ENABLE_CUDA "Enable CUDA interop path" ON)
option(OPENZOOM_ENABLE_TESTS "Enable building tests" OFF)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
