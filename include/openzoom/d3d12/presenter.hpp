#pragma once

#ifdef _WIN32

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <vector>

#include "openzoom/cuda/cuda_interop.hpp"

struct ID3D12Device;
struct ID3D12Fence;
struct ID3D12Resource;
struct IDXGIFactory6;
struct IDXGISwapChain3;

namespace openzoom {

class D3D12Presenter {
public:
    D3D12Presenter();
    ~D3D12Presenter();

    void Initialize(HWND hwnd, UINT width, UINT height);
    bool IsInitialized() const;
    void Resize(UINT width, UINT height);
    void Present(const uint8_t* data, UINT width, UINT height);
    void PresentFromTexture(ID3D12Resource* texture,
                            UINT width,
                            UINT height,
                            const FenceSyncParams* fenceSync = nullptr);

    ID3D12Device* GetDevice() const;
    ID3D12Fence* GetFence() const;
    UINT64 GetLastSignaledFenceValue() const;

private:
    void CreateDevice();
    void CreateCommandObjects();
    void CreateFenceObjects();
    void CreateSwapChain(UINT width, UINT height);
    void AcquireBackBuffers();
    void EnsureUploadBuffer(UINT width, UINT height);
    void CopyToUpload(const uint8_t* data, UINT width, UINT height);
    void WaitForGpu();
    void WaitForFenceValue(UINT64 value);

    HWND hwnd_{};
    UINT width_{};
    UINT height_{};
    bool initialized_{};

    Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> backBuffers_;

    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    UINT64 fenceValue_{};
    HANDLE fenceEvent_{nullptr};

    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT uploadFootprint_{};
    UINT uploadNumRows_{};
    UINT64 uploadRowSizeInBytes_{};
    UINT64 uploadTotalBytes_{};
    uint8_t* uploadMappedPtr_{};
    UINT uploadWidth_{};
    UINT uploadHeight_{};
};

} // namespace openzoom

#endif // _WIN32

