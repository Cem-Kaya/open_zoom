cmake_minimum_required(VERSION 3.23)

project(OpenZoom LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OPENZOOM_ENABLE_CUDA "Enable CUDA interop path" ON)
option(OPENZOOM_ENABLE_TESTS "Enable building tests" OFF)

if (OPENZOOM_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "OPENZOOM_ENABLE_CUDA=OFF: building without CUDA support")
endif()

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

file(GLOB OPENZOOM_CPP_SOURCES CONFIGURE_DEPENDS src/*.cpp)

if (OPENZOOM_ENABLE_CUDA)
    file(GLOB OPENZOOM_CUDA_SOURCES CONFIGURE_DEPENDS src/*.cu)
endif()

add_executable(open_zoom ${OPENZOOM_CPP_SOURCES})

target_sources(open_zoom PRIVATE
    include/openzoom/app.hpp
)

target_include_directories(open_zoom PRIVATE include)

set_target_properties(open_zoom PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if (OPENZOOM_ENABLE_CUDA)
    target_sources(open_zoom PRIVATE ${OPENZOOM_CUDA_SOURCES})
    set_target_properties(open_zoom PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86;89"
    )
endif()

target_compile_definitions(open_zoom PRIVATE
    OPENZOOM_ENABLE_CUDA=$<BOOL:OPENZOOM_ENABLE_CUDA>
    OPENZOOM_HAS_CUDA_EXT_MEMORY=$<IF:$<BOOL:OPENZOOM_ENABLE_CUDA>,1,0>
)

if (WIN32)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)

    if (OPENZOOM_ENABLE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(open_zoom PRIVATE CUDA::cudart CUDA::cuda_driver)
    endif()

    target_link_libraries(open_zoom PRIVATE
        Qt6::Widgets
        d3d12
        dxgi
        mfplat
        mfreadwrite
        mfuuid
        dxguid
        d3dcompiler
        ole32
        mf
        shlwapi
        uuid
        Synchronization
        windowscodecs
        comctl32
    )

    target_compile_definitions(open_zoom PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)

    if (OPENZOOM_ENABLE_CUDA)
        add_subdirectory(sandbox/dx12_cuda_minimal)
    endif()
else()
    message(WARNING "OpenZoom is intended to be built on Windows with CUDA, Qt6, and Direct3D12 support.")
endif()

if (OPENZOOM_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
