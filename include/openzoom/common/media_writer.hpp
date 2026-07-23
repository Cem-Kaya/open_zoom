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

// Media Foundation sink-writer wrapper for live AV1 or H.264 recording.
// The output container is fragmented MP4 (fMP4): fragments are flushed to disk
// while recording, so the file stays playable up to the last completed fragment
// even if the process dies before Finalize(). The file keeps the .mp4 extension.
class VideoRecorder {
public:
    enum class Codec { Av1, H264 };

    // Why the recorder last transitioned from recording to stopped.
    enum class StopReason { None, Manual, DiskFull, WriteFailed };

    VideoRecorder();
    ~VideoRecorder();

    // Refuses to start when the target volume has less than 500 MB free
    // (LastError() explains why). While recording, free space is re-checked
    // every ~5 seconds; below 200 MB the recording is finalized cleanly and
    // AddFrame() returns false with StopReason::DiskFull.
    bool Start(const std::wstring& filePath,
               UINT width,
               UINT height,
               UINT fps,
               Codec codec);
    void Stop();
    bool IsRecording() const { return recording_; }

    bool AddFrame(const uint8_t* bgraData, size_t strideBytes);

    double DurationSeconds() const;
    const std::string& LastError() const { return lastError_; }
    StopReason LastStopReason() const { return stopReason_; }
    Codec ActiveCodec() const { return activeCodec_; }
    static const char* CodecName(Codec codec);

private:
    bool InitializeSink(const std::wstring& filePath,
                        UINT width,
                        UINT height,
                        UINT fps,
                        Codec codec);
    void FinalizeAndStop(StopReason reason);
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
    std::wstring targetPath_;
    ULONGLONG lastSpaceCheckTicks_{0};
    StopReason stopReason_{StopReason::None};
    Codec activeCodec_{Codec::H264};
    std::string lastError_;
};

} // namespace openzoom

#endif // _WIN32
