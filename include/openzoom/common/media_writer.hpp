#pragma once

#ifdef _WIN32

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <wrl/client.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace openzoom {

// Simple Media Foundation sink-writer wrapper for H.264 MP4 recording.
class VideoRecorder {
public:
    VideoRecorder();
    ~VideoRecorder();

    bool Start(const std::wstring& filePath, UINT width, UINT height, UINT fps);
    void Stop();
    bool IsRecording() const { return recording_; }

    bool AddFrame(const uint8_t* bgraData, size_t strideBytes);

    double DurationSeconds() const;
    const std::string& LastError() const { return lastError_; }

private:
    bool InitializeSink(const std::wstring& filePath, UINT width, UINT height, UINT fps);
    void SetError(const std::string& err);

    IMFTransform* colorConverter_{nullptr}; // unused for now; reserved.
    Microsoft::WRL::ComPtr<IMFSinkWriter> sinkWriter_;
    DWORD streamIndex_{0};
    UINT frameWidth_{0};
    UINT frameHeight_{0};
    UINT fps_{30};
    LONGLONG rtStart_{0};
    LONGLONG rtDuration_{0};
    bool recording_{false};
    std::string lastError_;
};

} // namespace openzoom

#endif // _WIN32
