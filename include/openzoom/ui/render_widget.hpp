#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QWidget>

QT_BEGIN_NAMESPACE
class QPaintEngine;
class QResizeEvent;
class QShowEvent;
class QTimer;
QT_END_NAMESPACE

namespace openzoom {

class D3D12Presenter;

class RenderWidget final : public QWidget {
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
    void ApplyPendingPresenterResize();

    D3D12Presenter* presenter_{};
    QTimer* presenterResizeTimer_{};
};

} // namespace openzoom

#endif // defined(_WIN32) || defined(Q_MOC_RUN)
