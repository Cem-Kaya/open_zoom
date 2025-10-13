cmake_minimum_required(VERSION 3.23)

project(OpenZoom LANGUAGES CXX CUDA)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OPENZOOM_ENABLE_TESTS "Enable building tests" OFF)

file(GLOB OPENZOOM_SOURCES
    CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
)

add_executable(open_zoom ${OPENZOOM_SOURCES})

target_sources(open_zoom PRIVATE
    include/openzoom/app.hpp
)

target_include_directories(open_zoom PRIVATE include)

set_target_properties(open_zoom PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86;89"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if (WIN32)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(open_zoom PRIVATE CUDA::cudart CUDA::cuda_driver)

    target_link_libraries(open_zoom PRIVATE
        Qt6::Widgets
        d3d12
        dxgi
        dxguid
        d3dcompiler
        ole32
        mfplat
        mf
        mfreadwrite
        mfuuid
        shlwapi
        uuid
        Synchronization
        windowscodecs
        comctl32
    )

    target_compile_definitions(open_zoom PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)

    add_subdirectory(sandbox/dx12_cuda_minimal)
else()
    message(WARNING "OpenZoom is intended to be built on Windows with CUDA, Qt6, and Direct3D12 support.")
endif()

if (OPENZOOM_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
