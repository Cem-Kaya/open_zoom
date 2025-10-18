#ifdef _WIN32

#include "openzoom/ui/main_window.hpp"

#include "openzoom/app/app.hpp"
#include "openzoom/app/constants.hpp"
#include "openzoom/d3d12/presenter.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QCursor>
#include <QEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPushButton>
#include <QRegion>
#include <QResizeEvent>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSlider>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <utility>

namespace openzoom {

using namespace app_constants;

RenderWidget::RenderWidget(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_NativeWindow);
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::StrongFocus);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

QPaintEngine* RenderWidget::paintEngine() const
{
    return nullptr;
}

void RenderWidget::setPresenter(D3D12Presenter* presenter)
{
    presenter_ = presenter;
}

bool RenderWidget::isPresenterReady() const
{
    return presenter_ && presenter_->IsInitialized();
}

void RenderWidget::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    EnsurePresenter();
}

void RenderWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (EnsurePresenter()) {
        presenter_->Resize(std::max(1, width()), std::max(1, height()));
    }
}

bool RenderWidget::EnsurePresenter()
{
    if (!presenter_ || presenter_->IsInitialized()) {
        return presenter_ != nullptr;
    }

    HWND hwnd = reinterpret_cast<HWND>(winId());
    presenter_->Initialize(hwnd, std::max(1, width()), std::max(1, height()));
    return true;
}

JoystickOverlay::JoystickOverlay(QWidget* parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setAttribute(Qt::WA_TranslucentBackground, true);
    setVisible(false);
    if (parent) {
        parent->installEventFilter(this);
    }
    setFixedSize(160, 160);
    UpdateMask();
    ResetKnob();
}

void JoystickOverlay::ResetKnob()
{
    knobPos_ = QPointF(width() / 2.0, height() / 2.0);
    update();
}

bool JoystickOverlay::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == parentWidget()) {
        if (event->type() == QEvent::Resize) {
            UpdatePlacement();
        }
    }
    return QWidget::eventFilter(watched, event);
}

void JoystickOverlay::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    UpdatePlacement();
}

void JoystickOverlay::paintEvent(QPaintEvent*)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(rect(), Qt::transparent);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

    painter.setBrush(QColor(60, 60, 60, 180));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(rect());

    painter.setBrush(QColor(230, 230, 230, 230));
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(KnobRect());
}

void JoystickOverlay::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        dragging_ = true;
        UpdateFromPosition(event->position());
    }
}

void JoystickOverlay::mouseMoveEvent(QMouseEvent* event)
{
    if (dragging_) {
        UpdateFromPosition(event->position());
    }
}

void JoystickOverlay::mouseReleaseEvent(QMouseEvent* event)
{
    if (dragging_ && event->button() == Qt::LeftButton) {
        dragging_ = false;
        ResetKnob();
        emit JoystickChanged(0.0f, 0.0f);
        update();
    }
}

void JoystickOverlay::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    UpdateMask();
}

QRectF JoystickOverlay::KnobRect() const
{
    constexpr qreal knobRadius = 24.0;
    return QRectF(knobPos_.x() - knobRadius,
                  knobPos_.y() - knobRadius,
                  knobRadius * 2.0,
                  knobRadius * 2.0);
}

void JoystickOverlay::UpdatePlacement()
{
    if (!parentWidget()) {
        return;
    }
    const int margin = 20;
    const int x = parentWidget()->width() - width() - margin;
    const int y = parentWidget()->height() - height() - margin;
    move(std::max(0, x), std::max(0, y));
}

void JoystickOverlay::UpdateFromPosition(const QPointF& pos)
{
    const QPointF center(width() / 2.0, height() / 2.0);
    QPointF delta = pos - center;
    const qreal maxRadius = width() / 2.0;
    if (delta.manhattanLength() < 0.0001) {
        delta = QPointF(0, 0);
    }
    const qreal distance = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y());
    if (distance > maxRadius) {
        delta *= maxRadius / distance;
    }
    knobPos_ = center + delta;
    update();

    float normX = static_cast<float>(delta.x() / maxRadius);
    float normY = static_cast<float>(delta.y() / maxRadius);
    normX = std::clamp(normX, -1.0f, 1.0f);
    normY = std::clamp(normY, -1.0f, 1.0f);

    emit JoystickChanged(normX, -normY);
}

void JoystickOverlay::UpdateMask()
{
    if (width() <= 0 || height() <= 0) {
        clearMask();
        return;
    }
    QRegion region(rect(), QRegion::Ellipse);
    setMask(region);
}

MainWindow::MainWindow()
{
    setWindowTitle("OpenZoom");
    resize(1280, 720);

    auto* central = new QWidget(this);
    auto* rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(12, 12, 12, 12);
    rootLayout->setSpacing(8);

    controlsToggleButton_ = new QToolButton();
    controlsToggleButton_->setText("Hide Controls");
    controlsToggleButton_->setCheckable(true);
    controlsToggleButton_->setChecked(true);
    controlsToggleButton_->setArrowType(Qt::DownArrow);
    controlsToggleButton_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    joystickCheckbox_ = new QCheckBox("Virtual Joystick");

    auto* headerLayout = new QHBoxLayout();
    headerLayout->setSpacing(8);
    headerLayout->addWidget(controlsToggleButton_);
    headerLayout->addStretch(1);
    headerLayout->addWidget(joystickCheckbox_);
    rootLayout->addLayout(headerLayout);

    controlsContainer_ = new QWidget();
    auto* controlsLayout = new QVBoxLayout(controlsContainer_);
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->setSpacing(8);

    auto* cameraLayout = new QHBoxLayout();
    cameraLayout->setSpacing(8);
    auto* rotationLabel = new QLabel("Rotation:");
    rotationCombo_ = new QComboBox();
    rotationCombo_->addItem("0°");
    rotationCombo_->addItem("90°");
    rotationCombo_->addItem("180°");
    rotationCombo_->addItem("270°");
    rotationCombo_->setCurrentIndex(0);
    rotationCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    rotationCombo_->setEditable(false);
    rotationCombo_->setMinimumWidth(110);
    rotationCombo_->setToolTip("Rotate input/output clockwise");
    rotationLabel->setBuddy(rotationCombo_);
    cameraLayout->addWidget(rotationLabel);
    cameraLayout->addWidget(rotationCombo_);
    cameraLayout->addSpacing(12);

    auto* cameraLabel = new QLabel("Camera:");
    cameraCombo_ = new QComboBox();
    cameraCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    cameraLayout->addWidget(cameraLabel);
    cameraLayout->addWidget(cameraCombo_, 1);
    controlsLayout->addLayout(cameraLayout);

    auto* bwLayout = new QHBoxLayout();
    bwLayout->setSpacing(8);
    bwCheckbox_ = new QCheckBox("Black && White");
    bwSlider_ = new QSlider(Qt::Horizontal);
    bwSlider_->setRange(0, 255);
    bwSlider_->setPageStep(8);
    bwSlider_->setValue(128);
    bwSlider_->setEnabled(false);
    bwLayout->addWidget(bwCheckbox_);
    bwLayout->addWidget(bwSlider_, 1);
    controlsLayout->addLayout(bwLayout);

    auto* zoomLayout = new QHBoxLayout();
    zoomLayout->setSpacing(8);
    zoomCheckbox_ = new QCheckBox("Zoom");
    zoomSlider_ = new QSlider(Qt::Horizontal);
    zoomSlider_->setRange(kZoomSliderScale, kZoomSliderMaxMultiplier * kZoomSliderScale);
    zoomSlider_->setPageStep(10);
    zoomSlider_->setValue(kZoomSliderScale);
    zoomSlider_->setEnabled(false);
    zoomLayout->addWidget(zoomCheckbox_);
    zoomLayout->addWidget(zoomSlider_, 1);
    controlsLayout->addLayout(zoomLayout);

    auto* blurLayout = new QHBoxLayout();
    blurLayout->setSpacing(8);
    blurCheckbox_ = new QCheckBox("Gaussian Blur");
    blurSigmaSlider_ = new QSlider(Qt::Horizontal);
    blurSigmaSlider_->setRange(kBlurSigmaSliderMin, kBlurSigmaSliderMax);
    blurSigmaSlider_->setPageStep(2);
    blurSigmaSlider_->setSingleStep(1);
    blurSigmaSlider_->setValue(10);
    blurSigmaSlider_->setEnabled(false);
    blurSigmaValueLabel_ = new QLabel("1.0");
    blurSigmaValueLabel_->setMinimumWidth(40);

    blurRadiusSlider_ = new QSlider(Qt::Horizontal);
    blurRadiusSlider_->setRange(kSupportedBlurRadii.front(), kSupportedBlurRadii.back());
    blurRadiusSlider_->setPageStep(1);
    blurRadiusSlider_->setSingleStep(1);
    blurRadiusSlider_->setTickInterval(1);
    blurRadiusSlider_->setTickPosition(QSlider::TicksBelow);
    blurRadiusSlider_->setValue(3);
    blurRadiusSlider_->setEnabled(false);
    blurRadiusValueLabel_ = new QLabel("3");
    blurRadiusValueLabel_->setMinimumWidth(40);

    blurLayout->addWidget(blurCheckbox_);
    blurLayout->addWidget(new QLabel("Sigma:"));
    blurLayout->addWidget(blurSigmaSlider_, 1);
    blurLayout->addWidget(blurSigmaValueLabel_);
    blurLayout->addSpacing(12);
    blurLayout->addWidget(new QLabel("Radius:"));
    blurLayout->addWidget(blurRadiusSlider_, 1);
    blurLayout->addWidget(blurRadiusValueLabel_);
    controlsLayout->addLayout(blurLayout);

    auto* temporalLayout = new QHBoxLayout();
    temporalLayout->setSpacing(8);
    temporalSmoothCheckbox_ = new QCheckBox("Temporal Smooth");
    temporalSmoothCheckbox_->setChecked(false);
    temporalSmoothSlider_ = new QSlider(Qt::Horizontal);
    temporalSmoothSlider_->setRange(5, 100);
    temporalSmoothSlider_->setPageStep(5);
    temporalSmoothSlider_->setValue(25);
    temporalSmoothSlider_->setEnabled(false);
    temporalSmoothValueLabel_ = new QLabel("0.25");
    temporalSmoothValueLabel_->setMinimumWidth(40);
    temporalLayout->addWidget(temporalSmoothCheckbox_);
    temporalLayout->addSpacing(12);
    temporalLayout->addWidget(new QLabel("Blend:"));
    temporalLayout->addWidget(temporalSmoothSlider_, 1);
    temporalLayout->addWidget(temporalSmoothValueLabel_);
    controlsLayout->addLayout(temporalLayout);

    auto* spatialRow = new QHBoxLayout();
    spatialRow->setSpacing(8);
    spatialSharpenCheckbox_ = new QCheckBox("Spatial Sharpen");
    spatialSharpenCheckbox_->setChecked(false);
    spatialBackendCombo_ = new QComboBox();
    spatialBackendCombo_->addItem("AMD FSR 1.0 (EASU + RCAS)");
    spatialBackendCombo_->addItem("NVIDIA Image Scaling (default)");
    spatialBackendCombo_->setEnabled(false);
    spatialSharpnessSlider_ = new QSlider(Qt::Horizontal);
    spatialSharpnessSlider_->setRange(0, 100);
    spatialSharpnessSlider_->setPageStep(5);
    spatialSharpnessSlider_->setValue(25);
    spatialSharpnessSlider_->setEnabled(false);
    spatialSharpnessValueLabel_ = new QLabel("0.25");
    spatialSharpnessValueLabel_->setMinimumWidth(40);

    spatialRow->addWidget(spatialSharpenCheckbox_);
    spatialRow->addSpacing(12);
    spatialRow->addWidget(new QLabel("Backend:"));
    spatialRow->addWidget(spatialBackendCombo_, 1);
    spatialRow->addSpacing(12);
    spatialRow->addWidget(new QLabel("Sharpness:"));
    spatialRow->addWidget(spatialSharpnessSlider_, 1);
    spatialRow->addWidget(spatialSharpnessValueLabel_);
    controlsLayout->addLayout(spatialRow);

    auto* focusLayout = new QHBoxLayout();
    focusLayout->setSpacing(8);
    auto* focusXLabel = new QLabel("Focus X:");
    zoomCenterXSlider_ = new QSlider(Qt::Horizontal);
    zoomCenterXSlider_->setRange(0, kZoomFocusSliderScale);
    zoomCenterXSlider_->setPageStep(5);
    zoomCenterXSlider_->setValue(kZoomFocusSliderScale / 2);
    focusLayout->addWidget(focusXLabel);
    focusLayout->addWidget(zoomCenterXSlider_, 1);

    auto* focusYLabel = new QLabel("Focus Y:");
    zoomCenterYSlider_ = new QSlider(Qt::Horizontal);
    zoomCenterYSlider_->setRange(0, kZoomFocusSliderScale);
    zoomCenterYSlider_->setPageStep(5);
    zoomCenterYSlider_->setValue(kZoomFocusSliderScale / 2);
    focusLayout->addWidget(focusYLabel);
    focusLayout->addWidget(zoomCenterYSlider_, 1);
    controlsLayout->addLayout(focusLayout);

    auto* debugLayout = new QHBoxLayout();
    debugLayout->setSpacing(8);
    debugButton_ = new QPushButton("Debug View");
    debugButton_->setCheckable(true);
    debugButton_->setChecked(false);
    debugLayout->addWidget(debugButton_);
    focusMarkerCheckbox_ = new QCheckBox("Show Focus Point");
    focusMarkerCheckbox_->setChecked(false);
    focusMarkerCheckbox_->setToolTip("Overlay a red marker at the current zoom focus");
    debugLayout->addWidget(focusMarkerCheckbox_);
    processingStatusLabel_ = new QLabel("Processing: CPU");
    processingStatusLabel_->setObjectName("processingStatusLabel");
    processingStatusLabel_->setMinimumWidth(120);
    debugLayout->addWidget(processingStatusLabel_);
    debugLayout->addStretch(1);
    controlsLayout->addLayout(debugLayout);

    rootLayout->addWidget(controlsContainer_, 0);

    renderWidget_ = new RenderWidget();
    renderWidget_->installEventFilter(this);
    rootLayout->addWidget(renderWidget_, 1);

    setCentralWidget(central);
}

void MainWindow::setApp(OpenZoomApp* app)
{
    app_ = app;
}

RenderWidget* MainWindow::renderWidget() const
{
    return renderWidget_;
}

QComboBox* MainWindow::cameraCombo() const { return cameraCombo_; }
QCheckBox* MainWindow::blackWhiteCheckbox() const { return bwCheckbox_; }
QSlider* MainWindow::blackWhiteSlider() const { return bwSlider_; }
QCheckBox* MainWindow::zoomCheckbox() const { return zoomCheckbox_; }
QSlider* MainWindow::zoomSlider() const { return zoomSlider_; }
QPushButton* MainWindow::debugButton() const { return debugButton_; }
QComboBox* MainWindow::rotationCombo() const { return rotationCombo_; }
QCheckBox* MainWindow::focusMarkerCheckbox() const { return focusMarkerCheckbox_; }
QSlider* MainWindow::zoomCenterXSlider() const { return zoomCenterXSlider_; }
QSlider* MainWindow::zoomCenterYSlider() const { return zoomCenterYSlider_; }
QCheckBox* MainWindow::joystickCheckbox() const { return joystickCheckbox_; }
QToolButton* MainWindow::controlsToggleButton() const { return controlsToggleButton_; }
QWidget* MainWindow::controlsContainer() const { return controlsContainer_; }
QCheckBox* MainWindow::blurCheckbox() const { return blurCheckbox_; }
QSlider* MainWindow::blurSigmaSlider() const { return blurSigmaSlider_; }
QSlider* MainWindow::blurRadiusSlider() const { return blurRadiusSlider_; }
QLabel* MainWindow::blurSigmaValueLabel() const { return blurSigmaValueLabel_; }
QLabel* MainWindow::blurRadiusValueLabel() const { return blurRadiusValueLabel_; }
QCheckBox* MainWindow::temporalSmoothCheckbox() const { return temporalSmoothCheckbox_; }
QSlider* MainWindow::temporalSmoothSlider() const { return temporalSmoothSlider_; }
QLabel* MainWindow::temporalSmoothValueLabel() const { return temporalSmoothValueLabel_; }
QCheckBox* MainWindow::spatialSharpenCheckbox() const { return spatialSharpenCheckbox_; }
QComboBox* MainWindow::spatialBackendCombo() const { return spatialBackendCombo_; }
QSlider* MainWindow::spatialSharpnessSlider() const { return spatialSharpnessSlider_; }
QLabel* MainWindow::spatialSharpnessValueLabel() const { return spatialSharpnessValueLabel_; }
QLabel* MainWindow::processingStatusLabel() const { return processingStatusLabel_; }

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (app_ && app_->HandlePanKey(event->key(), true)) {
        event->accept();
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void MainWindow::keyReleaseEvent(QKeyEvent* event)
{
    if (app_ && app_->HandlePanKey(event->key(), false)) {
        event->accept();
        return;
    }
    QMainWindow::keyReleaseEvent(event);
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == renderWidget_) {
        switch (event->type()) {
        case QEvent::Wheel: {
            auto* wheel = static_cast<QWheelEvent*>(event);
            if (wheel->modifiers() & Qt::ControlModifier) {
                if (app_) {
                    app_->HandleZoomWheel(wheel->angleDelta().y(), wheel->position());
                }
                event->accept();
                return true;
            } else if (app_ && app_->HandlePanScroll(wheel)) {
                event->accept();
                return true;
            }
            break;
        }
        case QEvent::MouseButtonPress: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::MiddleButton) {
                if (app_) {
                    app_->BeginMousePan(mouse->position(), renderWidget_->size());
                }
                event->accept();
                return true;
            }
            break;
        }
        case QEvent::MouseMove: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (app_ && app_->UpdateMousePan(mouse->position())) {
                event->accept();
                return true;
            }
            break;
        }
        case QEvent::MouseButtonRelease: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::MiddleButton) {
                if (app_) {
                    app_->EndMousePan();
                }
                event->accept();
                return true;
            }
            break;
        }
        default:
            break;
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

} // namespace openzoom

#endif // _WIN32
