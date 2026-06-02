#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QMainWindow>
#include <QWidget>
#include <QPointF>
#include <QRectF>

QT_BEGIN_NAMESPACE
class QComboBox;
class QCheckBox;
class QSlider;
class QPaintEngine;
class QPushButton;
class QToolButton;
class QLabel;
class QListWidget;
class QEvent;
class QShowEvent;
class QResizeEvent;
class QPaintEvent;
class QMouseEvent;
class QWheelEvent;
class QKeyEvent;
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

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void UpdatePlacement();

    QString title_;
    QString body_;
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

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();

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

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
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
};

} // namespace openzoom

#endif // _WIN32
