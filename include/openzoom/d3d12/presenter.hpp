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

    // Copy a GPU texture into a CPU buffer (BGRA8). Blocking; waits for GPU completion.
    bool ReadbackTexture(ID3D12Resource* texture,
                         UINT width,
                         UINT height,
                         std::vector<uint8_t>& outBgra);

    // Enqueue an async copy of texture into a ring slot; returns false if no
    // slot is free (both in flight) — caller just skips this frame's readback.
    // Never blocks the CPU; the result surfaces one or more frames later via
    // TryGetCompletedReadback.
    bool RequestReadback(ID3D12Resource* texture, UINT width, UINT height);

    // If a previously requested readback has completed, move its pixels into
    // outBgra (BGRA8 tightly packed) and return true. Returns the OLDEST
    // completed request; at most one result per call, so poll every tick to
    // drain. Pending requests are silently dropped by Resize (dimensions are
    // changing anyway) — callers get no result for those frames.
    bool TryGetCompletedReadback(std::vector<uint8_t>& outBgra,
                                 UINT& outWidth,
                                 UINT& outHeight);

    ID3D12Device* GetDevice() const;
    ID3D12Fence* GetFence() const;
    UINT64 GetLastSignaledFenceValue() const;

    // Block until the GPU has drained all submitted work. Must be called before
    // releasing resources that in-flight frames may still reference (e.g. the
    // CUDA shared texture) now that Present paths no longer stall per frame.
    void WaitForIdle();

private:
    // Frames in flight; matches the swap-chain buffer count so the current
    // back-buffer index doubles as the per-frame resource slot.
    static constexpr UINT kFrameCount = 2;

    void CreateDevice();
    void CreateCommandObjects();
    void CreateFenceObjects();
    void CreateSwapChain(UINT width, UINT height);
    void AcquireBackBuffers();
    void EnsureUploadBuffer(UINT width, UINT height);
    void CopyToUpload(const uint8_t* data, UINT width, UINT height, UINT slot);
    void WaitForGpu();
    void WaitForFenceValue(UINT64 value);
    void WaitForFrameSlot(UINT slot);

    HWND hwnd_{};
    UINT width_{};
    UINT height_{};
    bool initialized_{};

    Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> frameCommandAllocators_[kFrameCount];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> readbackCommandAllocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> backBuffers_;

    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    UINT64 fenceValue_{};
    UINT64 frameFenceValues_[kFrameCount]{};
    HANDLE fenceEvent_{nullptr};

    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffers_[kFrameCount];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT uploadFootprint_{};
    UINT uploadNumRows_{};
    UINT64 uploadRowSizeInBytes_{};
    UINT64 uploadTotalBytes_{};
    uint8_t* uploadMappedPtrs_[kFrameCount]{};
    UINT uploadWidth_{};
    UINT uploadHeight_{};

    Microsoft::WRL::ComPtr<ID3D12Resource> readbackBuffer_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT readbackFootprint_{};
    UINT readbackNumRows_{};
    UINT64 readbackRowSizeInBytes_{};
    UINT64 readbackTotalBytes_{};
    UINT readbackWidth_{};
    UINT readbackHeight_{};

    // Async readback ring. Each slot owns its buffer and command allocator so
    // an in-flight copy never blocks presentation or the synchronous readback
    // path. A slot's allocator is only Reset while the slot is free, i.e.
    // after its previous fence value passed (or after a full queue drain).
    struct AsyncReadbackSlot {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 totalBytes{};
        UINT width{};
        UINT height{};
        UINT64 fenceValue{};
        bool inFlight{};
    };
    static constexpr UINT kAsyncReadbackSlotCount = 2;
    AsyncReadbackSlot asyncReadbackSlots_[kAsyncReadbackSlotCount];
};

} // namespace openzoom

#endif // _WIN32

