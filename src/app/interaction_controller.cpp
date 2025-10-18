#ifdef _WIN32

#include "openzoom/app/interaction_controller.hpp"

#include "openzoom/app/app.hpp"
#include "openzoom/app/constants.hpp"
#include "openzoom/ui/main_window.hpp"

#include <QCursor>
#include <QtGlobal>
#include <QSignalBlocker>
#include <QWheelEvent>
#include <QCheckBox>
#include <QSlider>

#include <algorithm>
#include <cmath>

namespace openzoom {

InteractionController::InteractionController(OpenZoomApp& app)
    : app_(app) {}

bool InteractionController::HandlePanKey(int key, bool pressed)
{
    switch (key) {
    case Qt::Key_Left:
        panLeftPressed_ = pressed;
        return true;
    case Qt::Key_Right:
        panRightPressed_ = pressed;
        return true;
    case Qt::Key_Up:
        panUpPressed_ = pressed;
        return true;
    case Qt::Key_Down:
        panDownPressed_ = pressed;
        return true;
    default:
        break;
    }
    return false;
}

bool InteractionController::HandlePanScroll(const QWheelEvent* wheelEvent)
{
    if (!wheelEvent || !app_.renderWidget_) {
        return false;
    }

    if (app_.debugViewEnabled_) {
        return false;
    }

    if (!app_.zoomEnabled_ || app_.zoomAmount_ <= 1.0f) {
        return false;
    }

    QPointF pixelDelta = wheelEvent->pixelDelta();
    float deltaX = 0.0f;
    float deltaY = 0.0f;
    bool hasPixelPrecision = false;

    if (!pixelDelta.isNull()) {
        deltaX = pixelDelta.x();
        deltaY = pixelDelta.y();
        hasPixelPrecision = true;
    } else {
        QPoint angleDelta = wheelEvent->angleDelta();
        if (angleDelta.isNull()) {
            return false;
        }
        deltaX = static_cast<float>(angleDelta.x()) / 120.0f;
        deltaY = static_cast<float>(angleDelta.y()) / 120.0f;
    }

    const float zoomFactor = std::max(app_.zoomAmount_, 1.0f);
    float moveX = 0.0f;
    float moveY = 0.0f;

    if (hasPixelPrecision) {
        const float widgetWidth = static_cast<float>(std::max(1, app_.renderWidget_->width()));
        const float widgetHeight = static_cast<float>(std::max(1, app_.renderWidget_->height()));
        moveX = -deltaX / widgetWidth / zoomFactor;
        moveY = -deltaY / widgetHeight / zoomFactor;
    } else {
        constexpr float wheelStepScale = 1.2f;
        moveX = -deltaX * app_constants::kPanKeyboardStep * wheelStepScale / zoomFactor;
        moveY = -deltaY * app_constants::kPanKeyboardStep * wheelStepScale / zoomFactor;
    }

    if (std::abs(moveX) < 1e-6f && std::abs(moveY) < 1e-6f) {
        return false;
    }

    app_.SetZoomCenter(app_.zoomCenterX_ + moveX,
                       app_.zoomCenterY_ + moveY,
                       true);
    return true;
}

void InteractionController::HandleZoomWheel(int delta, const QPointF& localPos)
{
    if (!app_.zoomSlider_) {
        return;
    }

    float focusU = 0.0f;
    float focusV = 0.0f;
    bool hasFocus = app_.MapViewToSource(localPos, focusU, focusV);
    if (app_.debugViewEnabled_) {
        hasFocus = false;
    }

    if (!app_.zoomEnabled_) {
        if (app_.zoomCheckbox_) {
            QSignalBlocker blocker(app_.zoomCheckbox_);
            app_.zoomCheckbox_->setChecked(true);
        }
        app_.zoomEnabled_ = true;
        app_.zoomSlider_->setEnabled(true);
    }

    const int stepUnits = (delta / 120);
    if (stepUnits == 0) {
        return;
    }
    const float prevZoom = app_.zoomAmount_;
    const int stepSize = std::max(app_.zoomSlider_->pageStep() / 2, 1);
    const int deltaValue = stepUnits * stepSize;
    const int newValue = std::clamp(app_.zoomSlider_->value() + deltaValue,
                                    app_.zoomSlider_->minimum(),
                                    app_.zoomSlider_->maximum());

    if (newValue == app_.zoomSlider_->value()) {
        return;
    }

    QSignalBlocker blockSlider(app_.zoomSlider_);
    app_.zoomSlider_->setValue(newValue);
    blockSlider.unblock();
    app_.OnZoomAmountChanged(newValue);
    const float newZoom = app_.zoomAmount_;

    if (hasFocus) {
        const float factor = (prevZoom <= 0.0f || newZoom <= 0.0f) ? 1.0f : (prevZoom / newZoom);
        const float newCenterX = focusU - (focusU - app_.zoomCenterX_) * factor;
        const float newCenterY = focusV - (focusV - app_.zoomCenterY_) * factor;
        app_.SetZoomCenter(newCenterX, newCenterY, true);
    }
}

void InteractionController::ApplyInputForces()
{
    float moveX = 0.0f;
    if (panLeftPressed_) {
        moveX -= 1.0f;
    }
    if (panRightPressed_) {
        moveX += 1.0f;
    }
    moveX += joystickPanX_;

    float moveY = 0.0f;
    if (panUpPressed_) {
        moveY -= 1.0f;
    }
    if (panDownPressed_) {
        moveY += 1.0f;
    }
    moveY += -joystickPanY_;

    if (std::abs(moveX) < 1e-5f && std::abs(moveY) < 1e-5f) {
        return;
    }

    const float length = std::sqrt(moveX * moveX + moveY * moveY);
    float normalizedX = moveX;
    float normalizedY = moveY;
    if (length > 1e-5f) {
        normalizedX /= length;
        normalizedY /= length;
    }

    const bool keyboardActive = panLeftPressed_ || panRightPressed_ || panUpPressed_ || panDownPressed_;
    const float analogStrength = std::sqrt(joystickPanX_ * joystickPanX_ + joystickPanY_ * joystickPanY_);

    float baseStep = 0.0f;
    if (keyboardActive) {
        baseStep = app_constants::kPanKeyboardStep;
    }
    if (analogStrength > 0.001f) {
        baseStep = std::max(baseStep, app_constants::kPanJoystickStep * std::clamp(analogStrength, 0.1f, 1.0f));
    }
    if (baseStep <= 0.0f) {
        baseStep = app_constants::kPanJoystickStep;
    }

    const float zoomFactor = std::max(1.0f, app_.zoomAmount_);
    const float step = baseStep / zoomFactor;

    app_.SetZoomCenter(app_.zoomCenterX_ + normalizedX * step,
                       app_.zoomCenterY_ + normalizedY * step,
                       true);
}

void InteractionController::BeginMousePan(const QPointF& pos, const QSize& widgetSize)
{
    Q_UNUSED(widgetSize);
    middlePanActive_ = true;
    middlePanLastPos_ = pos;
    if (app_.renderWidget_) {
        app_.renderWidget_->setCursor(Qt::ClosedHandCursor);
        app_.renderWidget_->grabMouse(Qt::ClosedHandCursor);
    }
}

bool InteractionController::UpdateMousePan(const QPointF& pos)
{
    if (!middlePanActive_) {
        return false;
    }

    float prevU = 0.0f;
    float prevV = 0.0f;
    float currU = 0.0f;
    float currV = 0.0f;

    bool prevValid = app_.MapViewToSource(middlePanLastPos_, prevU, prevV);
    bool currValid = app_.MapViewToSource(pos, currU, currV);

    if (!prevValid || !currValid) {
        middlePanLastPos_ = pos;
        return false;
    }

    const float deltaX = prevU - currU;
    const float deltaY = prevV - currV;
    if (std::abs(deltaX) < 1e-5f && std::abs(deltaY) < 1e-5f) {
        middlePanLastPos_ = pos;
        return false;
    }

    app_.SetZoomCenter(app_.zoomCenterX_ + deltaX,
                       app_.zoomCenterY_ + deltaY,
                       true);
    middlePanLastPos_ = pos;
    return true;
}

void InteractionController::EndMousePan()
{
    if (!middlePanActive_) {
        return;
    }
    middlePanActive_ = false;
    if (app_.renderWidget_) {
        app_.renderWidget_->releaseMouse();
        app_.renderWidget_->unsetCursor();
    }
}

void InteractionController::ResetJoystick()
{
    joystickPanX_ = 0.0f;
    joystickPanY_ = 0.0f;
}

void InteractionController::SetJoystickAxes(float x, float y)
{
    joystickPanX_ = std::clamp(x, -1.0f, 1.0f);
    joystickPanY_ = std::clamp(y, -1.0f, 1.0f);
}

} // namespace openzoom

#endif // _WIN32
