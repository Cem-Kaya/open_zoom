#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QObject>
#include <QString>

#include <cstdint>
#include <memory>

QT_BEGIN_NAMESPACE
class QNetworkAccessManager;
class QNetworkReply;
class QProcess;
QT_END_NAMESPACE

namespace openzoom {

class AssistiveRuntime : public QObject {
    Q_OBJECT
public:
    explicit AssistiveRuntime(QObject* parent = nullptr);
    ~AssistiveRuntime() override;

    void SetModes(bool ocrEnabled, bool vlmEnabled);
    bool WantsAnalysis() const;
    bool IsBusy() const;

    void SubmitFrame(const uint8_t* bgraData, int width, int height);

signals:
    void OverlayUpdated(const QString& title, const QString& body, bool visible);

private:
    void RefreshOverlay();
    void StartOcr(const uint8_t* bgraData, int width, int height);
    void StartVlm(const uint8_t* bgraData, int width, int height);
    void FinishOcrSuccess(const QString& text);
    void FinishOcrError(const QString& errorText);
    void FinishVlmSuccess(const QString& text);
    void FinishVlmError(const QString& errorText);
    QString TesseractProgram() const;
    bool VlmConfigured() const;

    bool ocrEnabled_{false};
    bool vlmEnabled_{false};
    bool ocrHardUnavailable_{false};
    bool vlmHardUnavailable_{false};

    QString ocrText_;
    QString vlmText_;
    QString ocrStatus_;
    QString vlmStatus_;

    std::unique_ptr<QProcess> ocrProcess_;
    QString pendingOcrImagePath_;
    QNetworkAccessManager* networkManager_{};
    QNetworkReply* activeReply_{};
};

} // namespace openzoom

#endif // defined(_WIN32) || defined(Q_MOC_RUN)
