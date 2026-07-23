#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QAbstractNativeEventFilter>
#include <QMainWindow>
#include <QWidget>
#include <QPointF>
#include <QRectF>

#include <array>

#include "openzoom/ui/render_widget.hpp"
#include "openzoom/ui/assistive_overlay.hpp"
#include "openzoom/ui/joystick_overlay.hpp"

QT_BEGIN_NAMESPACE
class QAbstractButton;
class QComboBox;
class QCheckBox;
class QListWidgetItem;
class QParallelAnimationGroup;
class QSlider;
class QSplitter;
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
class ColorSchemePicker;



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
    QComboBox* viewportRateCombo() const;
    QComboBox* viewportFitCombo() const;

    // Two-speed UI: Simple overlays compact corner controls on the full render
    // surface; Advanced adds a right-side inspector.
    void setSimpleMode(bool simple);
    bool isSimpleMode() const;
    int advancedPanelWidth() const;
    void setAdvancedPanelWidth(int width);
    QAbstractButton* simpleModeButton() const;
    QAbstractButton* advancedModeButton() const;
    QPushButton* explainNowButton() const;
    QPushButton* readTextButton() const;
    QCheckBox* stabilizationCheckbox() const;
    QSlider* stabilizationStrengthSlider() const;
    QCheckBox* keystoneCheckbox() const;
    QCheckBox* autoContrastCheckbox() const;
    QSlider* autoContrastStrengthSlider() const;
    QCheckBox* simpleTextClarityCheckbox() const;
    QCheckBox* textClarityCheckbox() const;
    QCheckBox* backgroundFlattenCheckbox() const;
    QSlider* backgroundFlattenStrengthSlider() const;
    QCheckBox* adaptiveBinarizationCheckbox() const;
    QSlider* sauvolaStrengthSlider() const;
    QSlider* binarizationSoftnessSlider() const;
    QComboBox* textPolarityCombo() const;
    QSlider* strokeWeightSlider() const;
    QCheckBox* smartSharpenCheckbox() const;
    QSlider* smartSharpenStrengthSlider() const;
    QCheckBox* claheCheckbox() const;
    QSlider* claheClipLimitSlider() const;
    QCheckBox* twoColorTextCheckbox() const;
    QCheckBox* textHysteresisCheckbox() const;
    QSlider* textHysteresisStrengthSlider() const;
    QCheckBox* selectiveSharpenCheckbox() const;
    QCheckBox* focusDetectionCheckbox() const;
    QSlider* focusThresholdSlider() const;
    QCheckBox* glareSuppressionCheckbox() const;
    QSlider* glareSuppressionStrengthSlider() const;
    QCheckBox* mlTextSuperResolutionCheckbox() const;
    QSlider* mlTextSuperResolutionStrengthSlider() const;
    QCheckBox* mlTextSuperResolutionPrefer2xCheckbox() const;
    QCheckBox* mlTextSuperResolutionUltra1440pCheckbox() const;
    ColorSchemePicker* displayColorPicker() const;
    QSlider* contrastSlider() const;
    QSlider* brightnessSlider() const;
    QPushButton* aiSettingsButton() const;
    QPushButton* openNotesButton() const;
    QPushButton* setupAssistantButton() const;
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
    bool isMaxineRuntimeInstalled() const;
    void setMaxineRuntimeInstalled(bool installed);
    void setSuperResStatus(const QString& status,
                           bool active,
                           bool performanceLimited = false);
    void setSuperResPerformanceOverrideChecked(bool checked);
    void setKeystoneTrackingControls(bool active,
                                     bool available,
                                     bool paused,
                                     bool canStepBack,
                                     bool canStepForward,
                                     bool stepPending,
                                     int position,
                                     int count);

signals:
    void keystoneStepBackRequested();
    void keystonePauseResumeRequested();
    void keystoneStepForwardRequested();
    void resetCurrentProfileRequested();
    void superResPerformanceOverrideChanged(bool enabled);

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
    void ApplyAdvancedPanelWidth();
    void ShowHelpDialog();

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
    QToolButton* helpButton_{};
    QPushButton* resetProfileButton_{};
    QToolButton* controlsToggleButton_{};
    QWidget* controlsContainer_{};
    QLabel* processingStatusLabel_{};
    QLabel* maxineAttribution_{};
    bool maxineRuntimeInstalled_{};
    OpenZoomApp* app_{};
    QComboBox* rotationCombo_{};
    QComboBox* viewportRateCombo_{};
    QComboBox* viewportFitCombo_{};
    QWidget* advancedPanel_{};
    QSplitter* contentSplitter_{};
    int advancedPanelPreferredWidth_{520};
    QWidget* topLeftPanel_{};
    QWidget* bottomLeftPanel_{};
    QWidget* keystoneTrackingPanel_{};
    QWidget* bottomRightPanel_{};
    QWidget* modeGridPopup_{};
    QWidget* modeToast_{};
    QLabel* modeToastTitle_{};
    QLabel* modeToastSubtitle_{};
    QToolButton* modeGridButton_{};
    QPushButton* previousModeButton_{};
    QPushButton* currentModeButton_{};
    QPushButton* nextModeButton_{};
    QPushButton* simpleKeystoneBackButton_{};
    QPushButton* simpleKeystonePauseButton_{};
    QPushButton* simpleKeystoneNextButton_{};
    QTimer* simpleChromeIdleTimer_{};
    QTimer* modeToastTimer_{};
    QParallelAnimationGroup* chromeAnimation_{};
    bool simpleChromeVisible_{true};
    bool chromePinned_{};
    bool keystoneTrackingActive_{};
    QPushButton* simpleModeButton_{};
    QPushButton* advancedModeButton_{};
    QPushButton* explainNowButton_{};
    QPushButton* readTextButton_{};
    QCheckBox* stabilizationCheckbox_{};
    QSlider* stabilizationStrengthSlider_{};
    QCheckBox* keystoneCheckbox_{};
    QWidget* advancedKeystoneTrackingRow_{};
    QPushButton* advancedKeystoneBackButton_{};
    QPushButton* advancedKeystonePauseButton_{};
    QPushButton* advancedKeystoneNextButton_{};
    QCheckBox* autoContrastCheckbox_{};
    QSlider* autoContrastStrengthSlider_{};
    QCheckBox* simpleTextClarityCheckbox_{};
    QCheckBox* textClarityCheckbox_{};
    QCheckBox* backgroundFlattenCheckbox_{};
    QSlider* backgroundFlattenStrengthSlider_{};
    QCheckBox* adaptiveBinarizationCheckbox_{};
    QSlider* sauvolaStrengthSlider_{};
    QSlider* binarizationSoftnessSlider_{};
    QComboBox* textPolarityCombo_{};
    QSlider* strokeWeightSlider_{};
    QCheckBox* smartSharpenCheckbox_{};
    QSlider* smartSharpenStrengthSlider_{};
    QCheckBox* claheCheckbox_{};
    QSlider* claheClipLimitSlider_{};
    QCheckBox* twoColorTextCheckbox_{};
    QCheckBox* textHysteresisCheckbox_{};
    QSlider* textHysteresisStrengthSlider_{};
    QCheckBox* selectiveSharpenCheckbox_{};
    QCheckBox* focusDetectionCheckbox_{};
    QSlider* focusThresholdSlider_{};
    QCheckBox* glareSuppressionCheckbox_{};
    QSlider* glareSuppressionStrengthSlider_{};
    QCheckBox* mlTextSuperResolutionCheckbox_{};
    QSlider* mlTextSuperResolutionStrengthSlider_{};
    QCheckBox* mlTextSuperResolutionPrefer2xCheckbox_{};
    QCheckBox* mlTextSuperResolutionUltra1440pCheckbox_{};
    QLabel* mlTextSuperResolutionStatusLabel_{};
    QCheckBox* mlTextSuperResolutionOverrideCheckbox_{};
    ColorSchemePicker* displayColorPicker_{};
    QSlider* contrastSlider_{};
    QSlider* brightnessSlider_{};
    QPushButton* aiSettingsButton_{};
    QPushButton* openNotesButton_{};
    QPushButton* setupAssistantButton_{};
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
