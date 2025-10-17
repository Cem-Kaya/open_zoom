set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(ProjectOptions.cmake)

# Enable CUDA if requested
if (OPENZOOM_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "OPENZOOM_ENABLE_CUDA=OFF: building without CUDA support")
endif()

file(GLOB_RECURSE OPENZOOM_CPP_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
if (OPENZOOM_ENABLE_CUDA)
    file(GLOB_RECURSE OPENZOOM_CUDA_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cu)
endif()

add_executable(open_zoom ${OPENZOOM_CPP_SOURCES})

if (OPENZOOM_ENABLE_CUDA)
    target_sources(open_zoom PRIVATE ${OPENZOOM_CUDA_SOURCES})
    set_target_properties(open_zoom PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86;89"
    )
endif()

target_sources(open_zoom PRIVATE
    ${PROJECT_SOURCE_DIR}/include/openzoom/app/app.hpp
    ${PROJECT_SOURCE_DIR}/include/openzoom/cuda/cuda_interop.hpp
    ${PROJECT_SOURCE_DIR}/include/openzoom/cuda/cuda_kernels.hpp
)

target_include_directories(open_zoom PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(open_zoom PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

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
        set(SANDBOX_DIR ${PROJECT_SOURCE_DIR}/sandbox/dx12_cuda_minimal)
        if (EXISTS "${SANDBOX_DIR}/CMakeLists.txt")
            add_subdirectory(${SANDBOX_DIR} ${CMAKE_BINARY_DIR}/sandbox_dx12_cuda_minimal)
        endif()
    endif()
else()
    message(WARNING "OpenZoom is intended to be built on Windows with CUDA, Qt6, and Direct3D12 support.")
endif()

if (OPENZOOM_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests ${CMAKE_BINARY_DIR}/tests)
endif()
