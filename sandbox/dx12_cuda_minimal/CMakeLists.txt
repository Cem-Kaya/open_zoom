if (NOT WIN32)
    return()
endif()

add_executable(dx12_cuda_minimal
    main.cpp
    minimal_kernel.cu
)

target_include_directories(dx12_cuda_minimal PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(dx12_cuda_minimal PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    d3d12
    dxgi
    dxguid
    d3dcompiler
    Synchronization
)

target_compile_definitions(dx12_cuda_minimal PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)

set_target_properties(dx12_cuda_minimal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86;89"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
