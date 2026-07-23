#pragma once

#include <QComboBox>
#include <QSlider>
#include <QWheelEvent>

namespace openzoom {

// A wheel over a settings selector belongs to the surrounding scroll panel.
// Selection remains available by click and keyboard without accidental edits.
class WheelSafeComboBox final : public QComboBox {
public:
    explicit WheelSafeComboBox(QWidget* parent = nullptr) : QComboBox(parent) {}

protected:
    void wheelEvent(QWheelEvent* event) override
    {
        event->ignore();
    }
};

// A wheel over a settings slider also belongs to the surrounding scroll panel.
// Values remain editable by dragging, clicking, and keyboard input.
class WheelSafeSlider final : public QSlider {
public:
    explicit WheelSafeSlider(Qt::Orientation orientation, QWidget* parent = nullptr)
        : QSlider(orientation, parent)
    {
    }

protected:
    void wheelEvent(QWheelEvent* event) override
    {
        event->ignore();
    }
};

} // namespace openzoom
