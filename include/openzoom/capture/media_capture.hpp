#pragma once

#ifdef _WIN32

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace openzoom {

struct MediaFrame {
    std::vector<uint8_t> data;
    GUID subtype{GUID_NULL};
    UINT width{0};
    UINT height{0};
    UINT stride{0};
    size_t dataSize{0};
};

using FrameCallback = std::function<void(const MediaFrame& frame)>;
using CaptureErrorCallback = std::function<void(const std::string& message)>;

struct CameraDescriptor {
    std::wstring name;
    std::wstring symbolicLink;
    Microsoft::WRL::ComPtr<IMFActivate> activation;
};

struct VideoFormat {
    UINT width{0};
    UINT height{0};
    UINT numerator{0};
    UINT denominator{0};
};

// Plain-language classification of the most recent capture failure, derived
// from the final HRESULT of StartCapture or from a mid-stream capture error.
enum class CameraFailureKind { None, DeviceBusy, DeviceMissing, AccessDenied, Other };

class MediaCapture {
public:
    MediaCapture();
    ~MediaCapture();

    bool Initialize();
    void Shutdown();

    std::vector<CameraDescriptor> EnumerateCameras();
    std::vector<VideoFormat> EnumerateFormats(const CameraDescriptor& descriptor);

    bool StartCapture(const CameraDescriptor& descriptor,
                      FrameCallback callback,
                      GUID preferredSubtype = MFVideoFormat_NV12,
                      CaptureErrorCallback errorCallback = {});
    void StopCapture();

    const std::string& LastError() const { return lastError_; }
    CameraFailureKind LastFailureKind() const { return lastFailureKind_.load(); }

    // Atomically returns whether the capture thread detected mid-stream device
    // loss, clearing the flag. Poll from the app's frame tick to drive reconnection.
    bool ConsumeDeviceLost();

    // Symbolic link of the device from the most recent StartCapture. Kept across
    // device loss and StopCapture so the app can re-enumerate and find the same
    // physical camera again when reconnecting.
    const std::wstring& LastSymbolicLink() const { return lastSymbolicLink_; }

private:
    struct FrameFormat {
        GUID subtype{GUID_NULL};
        UINT width{0};
        UINT height{0};
        UINT stride{0};
    };

    bool ConfigureReader(IMFSourceReader* reader,
                         GUID preferredSubtype,
                         FrameFormat& outFormat);
    bool ReadCurrentFormat(IMFSourceReader* reader, FrameFormat& outFormat);
    HRESULT TryOpenDevice(const CameraDescriptor& descriptor,
                          Microsoft::WRL::ComPtr<IMFMediaSource>& outSource,
                          Microsoft::WRL::ComPtr<IMFSourceReader>& outReader,
                          const char*& failedStage);
    void CaptureLoop(FrameCallback callback, CaptureErrorCallback errorCallback);
    std::vector<VideoFormat> ExtractFormats(IMFSourceReader* reader);
    static std::string HrToString(HRESULT hr);

    Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource_;
    Microsoft::WRL::ComPtr<IMFSourceReader> sourceReader_;
    Microsoft::WRL::ComPtr<IMFActivate> activeActivation_;
    std::thread captureThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> deviceLost_{false};
    std::atomic<CameraFailureKind> lastFailureKind_{CameraFailureKind::None};
    FrameFormat currentFormat_{};
    std::wstring lastSymbolicLink_;
    std::string lastError_;
};

} // namespace openzoom

#endif // _WIN32
