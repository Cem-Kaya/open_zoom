#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QPointF>
#include <QSize>

class QWheelEvent;

namespace openzoom {

class OpenZoomApp;

class InteractionController {
public:
    explicit InteractionController(OpenZoomApp& app);

    bool HandlePanKey(int key, bool pressed);
    bool HandlePanScroll(const QWheelEvent* wheelEvent);
    void HandleZoomWheel(int delta, const QPointF& localPos);

    void ApplyInputForces();

    void BeginMousePan(const QPointF& pos, const QSize& widgetSize);
    bool UpdateMousePan(const QPointF& pos);
    void EndMousePan();
    bool IsMousePanActive() const { return middlePanActive_; }

    void ResetJoystick();
    void SetJoystickAxes(float x, float y);

private:
    OpenZoomApp& app_;
    bool panLeftPressed_{false};
    bool panRightPressed_{false};
    bool panUpPressed_{false};
    bool panDownPressed_{false};
    float joystickPanX_{0.0f};
    float joystickPanY_{0.0f};
    bool middlePanActive_{false};
    QPointF middlePanLastPos_{};
};

} // namespace openzoom

#endif // _WIN32
