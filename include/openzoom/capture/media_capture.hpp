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
                      GUID preferredSubtype = MFVideoFormat_ARGB32);
    void StopCapture();

    const std::string& LastError() const { return lastError_; }

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
    void CaptureLoop(FrameCallback callback);
    std::vector<VideoFormat> ExtractFormats(IMFSourceReader* reader);
    static std::string HrToString(HRESULT hr);

    Microsoft::WRL::ComPtr<IMFMediaSource> mediaSource_;
    Microsoft::WRL::ComPtr<IMFSourceReader> sourceReader_;
    std::thread captureThread_;
    std::atomic<bool> running_{false};
    FrameFormat currentFormat_{};
    std::string lastError_;
};

} // namespace openzoom

#endif // _WIN32
