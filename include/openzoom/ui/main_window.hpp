#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QAbstractNativeEventFilter>
#include <QMainWindow>
#include <QWidget>
#include <QPointF>
#include <QRectF>

#include <array>

QT_BEGIN_NAMESPACE
class QAbstractButton;
class QComboBox;
class QCheckBox;
class QListWidgetItem;
class QParallelAnimationGroup;
class QSlider;
class QPaintEngine;
class QPushButton;
class QToolButton;
class QLabel;
class QLineEdit;
class QListWidget;
class QTimer;
class QEvent;
class QShowEvent;
class QResizeEvent;
class QPaintEvent;
class QMouseEvent;
class QPlainTextEdit;
class QWheelEvent;
class QKeyEvent;
class QTextBrowser;
QT_END_NAMESPACE

namespace openzoom {

class OpenZoomApp;
class D3D12Presenter;

class RenderWidget : public QWidget {
    Q_OBJECT
public:
    explicit RenderWidget(QWidget* parent = nullptr);

    QPaintEngine* paintEngine() const override;
    void setPresenter(D3D12Presenter* presenter);
    bool isPresenterReady() const;

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    bool EnsurePresenter();

    D3D12Presenter* presenter_{};
};

class AssistiveOverlay : public QWidget {
    Q_OBJECT
public:
    explicit AssistiveOverlay(QWidget* parent = nullptr);

    void SetContent(const QString& title, const QString& body, bool visible);
    void SetBusy(bool busy);
    std::array<QWidget*, 5> FocusTargets() const;

signals:
    void Dismissed();
    void ReadAloudRequested(const QString& text);
    void QuestionSubmitted(const QString& question);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    void UpdatePlacement();
    void BeginDrag(const QPoint& globalPosition);
    void ContinueDrag(const QPoint& globalPosition);
    void BeginResize(const QPoint& localPosition, const QPoint& globalPosition);
    void ContinueResize(const QPoint& globalPosition);
    void UpdateResizeCursor(const QPoint& localPosition);
    Qt::Edges ResizeEdgesAt(const QPoint& localPosition) const;
    QRect ConstrainedGeometry(const QRect& requested) const;
    void SubmitQuestion();

    QString title_;
    QString body_;
    QWidget* headerWidget_{};
    QLabel* titleLabel_{};
    QTextBrowser* bodyView_{};
    QLineEdit* questionEdit_{};
    QPushButton* askButton_{};
    QPushButton* readAloudButton_{};
    QToolButton* closeButton_{};
    QPoint pointerStartGlobal_;
    QPoint parentOrigin_;
    QRect pointerStartGeometry_;
    Qt::Edges resizeEdges_{};
    bool dragging_{};
    bool resizing_{};
    bool placementInitialized_{};
};

class JoystickOverlay : public QWidget {
    Q_OBJECT
public:
    explicit JoystickOverlay(QWidget* parent = nullptr);

    void ResetKnob();

signals:
    void JoystickChanged(float normX, float normY);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    QRectF KnobRect() const;
    void UpdatePlacement();
    void UpdateFromPosition(const QPointF& pos);
    void UpdateMask();

    bool dragging_{};
    QPointF knobPos_{};
};

class MainWindow : public QMainWindow, public QAbstractNativeEventFilter {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow() override;

    void setApp(OpenZoomApp* app);

    RenderWidget* renderWidget() const;
    QComboBox* cameraCombo() const;
    QListWidget* presetList() const;
    QLabel* presetDescriptionLabel() const;
    QPushButton* promotePresetButton() const;
    QCheckBox* blackWhiteCheckbox() const;
    QSlider* blackWhiteSlider() const;
    QCheckBox* zoomCheckbox() const;
    QSlider* zoomSlider() const;
    QPushButton* debugButton() const;
    QCheckBox* focusMarkerCheckbox() const;
    QSlider* zoomCenterXSlider() const;
    QSlider* zoomCenterYSlider() const;
    QCheckBox* joystickCheckbox() const;
    QToolButton* controlsToggleButton() const;
    QWidget* controlsContainer() const;
    QCheckBox* blurCheckbox() const;
    QSlider* blurSigmaSlider() const;
    QSlider* blurRadiusSlider() const;
    QLabel* blurSigmaValueLabel() const;
    QLabel* blurRadiusValueLabel() const;
    QListWidget* cameraModesList() const;
    QPushButton* capturePhotoButton() const;
    QPushButton* recordButton() const;
    QCheckBox* temporalSmoothCheckbox() const;
    QSlider* temporalSmoothSlider() const;
    QLabel* temporalSmoothValueLabel() const;
    QCheckBox* ocrAssistCheckbox() const;
    QCheckBox* vlmAssistCheckbox() const;
    QCheckBox* assistiveOverlayCheckbox() const;
    QCheckBox* spatialSharpenCheckbox() const;
    QComboBox* spatialBackendCombo() const;
    QSlider* spatialSharpnessSlider() const;
    QLabel* spatialSharpnessValueLabel() const;
    QLabel* processingStatusLabel() const;
    QComboBox* rotationCombo() const;

    // Two-speed UI: Simple overlays compact corner controls on the full render
    // surface; Advanced adds a right-side inspector.
    void setSimpleMode(bool simple);
    bool isSimpleMode() const;
    QAbstractButton* simpleModeButton() const;
    QAbstractButton* advancedModeButton() const;
    QPushButton* explainNowButton() const;
    QPushButton* readTextButton() const;
    QCheckBox* stabilizationCheckbox() const;
    QSlider* stabilizationStrengthSlider() const;
    QCheckBox* keystoneCheckbox() const;
    QCheckBox* autoContrastCheckbox() const;
    QSlider* autoContrastStrengthSlider() const;
    QComboBox* displayColorCombo() const;
    QSlider* contrastSlider() const;
    QSlider* brightnessSlider() const;
    QPushButton* aiSettingsButton() const;
    QPushButton* openNotesButton() const;
    QLabel* assistantConnectionLabel() const;
    QLabel* assistantUsageLabel() const;
    QPushButton* assistantConnectButton() const;
    QTextBrowser* assistantTranscript() const;
    QPlainTextEdit* assistantPromptEdit() const;
    QCheckBox* assistantAttachFrameCheckbox() const;
    QPushButton* assistantSendButton() const;
    QPushButton* assistantStopButton() const;
    QPushButton* assistantNewButton() const;
    QListWidget* assistantHistoryList() const;
    QPushButton* assistantRenameButton() const;
    QPushButton* assistantExportButton() const;
    QPushButton* assistantDeleteButton() const;

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;
    bool nativeEventFilter(const QByteArray& eventType, void* message, qintptr* result) override;

private:
    void ActivatePresetRow(int row);
    void ActivateRelativePreset(int offset);
    void ToggleModeGrid();
    void UpdateCurrentPresetUi(QListWidgetItem* current, QListWidgetItem* previous);
    void ShowModeAnnouncement(const QString& label, const QString& profileName);
    void UpdateSimpleChromeGeometry();
    void RevealSimpleChrome();
    void FadeSimpleChrome();
    void SetChromeOpacity(qreal opacity, int durationMs, bool hideWhenFinished);
    bool SimpleChromeHasFocus() const;

    RenderWidget* renderWidget_{};
    QComboBox* cameraCombo_{};
    QListWidget* presetList_{};
    QLabel* presetDescriptionLabel_{};
    QPushButton* promotePresetButton_{};
    QCheckBox* bwCheckbox_{};
    QSlider* bwSlider_{};
    QCheckBox* zoomCheckbox_{};
    QSlider* zoomSlider_{};
    QCheckBox* blurCheckbox_{};
    QSlider* blurSigmaSlider_{};
    QSlider* blurRadiusSlider_{};
    QLabel* blurSigmaValueLabel_{};
    QLabel* blurRadiusValueLabel_{};
    QListWidget* cameraModesList_{};
    QPushButton* capturePhotoButton_{};
    QPushButton* recordButton_{};
    QCheckBox* temporalSmoothCheckbox_{};
    QSlider* temporalSmoothSlider_{};
    QLabel* temporalSmoothValueLabel_{};
    QCheckBox* ocrAssistCheckbox_{};
    QCheckBox* vlmAssistCheckbox_{};
    QCheckBox* assistiveOverlayCheckbox_{};
    QCheckBox* spatialSharpenCheckbox_{};
    QComboBox* spatialBackendCombo_{};
    QSlider* spatialSharpnessSlider_{};
    QLabel* spatialSharpnessValueLabel_{};
    QPushButton* debugButton_{};
    QCheckBox* focusMarkerCheckbox_{};
    QSlider* zoomCenterXSlider_{};
    QSlider* zoomCenterYSlider_{};
    QCheckBox* joystickCheckbox_{};
    QToolButton* controlsToggleButton_{};
    QWidget* controlsContainer_{};
    QLabel* processingStatusLabel_{};
    OpenZoomApp* app_{};
    QComboBox* rotationCombo_{};
    QWidget* advancedPanel_{};
    QWidget* topLeftPanel_{};
    QWidget* bottomLeftPanel_{};
    QWidget* bottomRightPanel_{};
    QWidget* modeGridPopup_{};
    QWidget* modeToast_{};
    QLabel* modeToastTitle_{};
    QLabel* modeToastSubtitle_{};
    QToolButton* modeGridButton_{};
    QPushButton* previousModeButton_{};
    QPushButton* currentModeButton_{};
    QPushButton* nextModeButton_{};
    QTimer* simpleChromeIdleTimer_{};
    QTimer* modeToastTimer_{};
    QParallelAnimationGroup* chromeAnimation_{};
    bool simpleChromeVisible_{true};
    bool chromePinned_{};
    QPushButton* simpleModeButton_{};
    QPushButton* advancedModeButton_{};
    QPushButton* explainNowButton_{};
    QPushButton* readTextButton_{};
    QCheckBox* stabilizationCheckbox_{};
    QSlider* stabilizationStrengthSlider_{};
    QCheckBox* keystoneCheckbox_{};
    QCheckBox* autoContrastCheckbox_{};
    QSlider* autoContrastStrengthSlider_{};
    QComboBox* displayColorCombo_{};
    QSlider* contrastSlider_{};
    QSlider* brightnessSlider_{};
    QPushButton* aiSettingsButton_{};
    QPushButton* openNotesButton_{};
    QLabel* assistantConnectionLabel_{};
    QLabel* assistantUsageLabel_{};
    QPushButton* assistantConnectButton_{};
    QTextBrowser* assistantTranscript_{};
    QPlainTextEdit* assistantPromptEdit_{};
    QCheckBox* assistantAttachFrameCheckbox_{};
    QPushButton* assistantSendButton_{};
    QPushButton* assistantStopButton_{};
    QPushButton* assistantNewButton_{};
    QListWidget* assistantHistoryList_{};
    QPushButton* assistantRenameButton_{};
    QPushButton* assistantExportButton_{};
    QPushButton* assistantDeleteButton_{};
};

} // namespace openzoom

#endif // _WIN32
