#pragma once

#ifdef _WIN32

#include "openzoom/common/media_writer.hpp"

#include <QElapsedTimer>
#include <QString>

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

QT_BEGIN_NAMESPACE
class QPushButton;
QT_END_NAMESPACE

namespace openzoom {

struct CapturedFrame {
    std::vector<uint8_t> pixels;
    UINT width{};
    UINT height{};

    bool IsValid() const {
        return !pixels.empty() && width > 0 && height > 0;
    }
};

enum class RecordingState {
    Idle,
    Starting,
    Recording,
    Stopping,
    Error
};

class RecordingManager {
public:
    using StatusCallback = std::function<void(const QString&, int)>;

    RecordingManager(QPushButton* recordButton, StatusCallback statusCallback);
    ~RecordingManager();

    void SetRequested(bool requested);
    void Stop(const QString& message = {});

    bool IsActive() const;
    RecordingState State() const { return state_; }
    const QString& CodecName() const { return codecName_; }

    void StorePendingOriginal(UINT64 requestId, CapturedFrame frame);
    void HandleProcessedReadback(UINT64 requestId,
                                 const uint8_t* processedData,
                                 UINT processedWidth,
                                 UINT processedHeight);
    void ClearPendingReadbacks();
    void AddFrame(const uint8_t* processedData,
                  UINT processedWidth,
                  UINT processedHeight,
                  const CapturedFrame& originalFrame);

private:
    struct StartParameters {
        UINT processedWidth{};
        UINT processedHeight{};
        UINT originalWidth{};
        UINT originalHeight{};
    };

    bool SetRecordingState(RecordingState state,
                           const StartParameters* startParameters = nullptr);
    QString EnsureOutputDirectory() const;
    void ShowStatus(const QString& message, int durationMs = 7000) const;
    void UpdateButton();

    QPushButton* recordButton_{};
    StatusCallback statusCallback_;
    VideoRecorder processedRecorder_;
    VideoRecorder originalRecorder_;
    std::unordered_map<UINT64, CapturedFrame> pendingOriginalReadbacks_;
    RecordingState state_{RecordingState::Idle};
    QString sessionTimestamp_;
    QString codecName_;
    QElapsedTimer recordingTimer_;
    uint64_t frameCount_{};
    UINT processedWidth_{};
    UINT processedHeight_{};
    UINT originalWidth_{};
    UINT originalHeight_{};
};

} // namespace openzoom

#endif // _WIN32
