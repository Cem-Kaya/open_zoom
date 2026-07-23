#pragma once

#if defined(_WIN32) || defined(Q_MOC_RUN)

#include <QRect>
#include <QWidget>

#include <array>

QT_BEGIN_NAMESPACE
class QEvent;
class QLabel;
class QLineEdit;
class QMouseEvent;
class QPushButton;
class QShowEvent;
class QTextBrowser;
class QToolButton;
QT_END_NAMESPACE

namespace openzoom {

class AssistiveOverlay : public QWidget {
    Q_OBJECT
public:
    explicit AssistiveOverlay(QWidget* parent = nullptr);

    void SetContent(const QString& title, const QString& body, bool visible);
    void SetBusy(bool busy);
    void RestoreRelativeGeometry(const QRect& geometry);
    QRect RelativeGeometry() const;
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
    bool busy_{};
    bool placementInitialized_{};
    QRect restoredRelativeGeometry_;
};

} // namespace openzoom

#endif // _WIN32
