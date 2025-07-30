#encoding: utf-8
#author: vectorwang@hotmail.com

from PyQt6.QtWidgets import (
    QWidget, QGraphicsScene,
    QGraphicsView, QGraphicsItem,
)
from PyQt6.QtCore import (
    QPointF, Qt, QSize, pyqtSignal
)
from PyQt6.QtSvgWidgets import (
    QGraphicsSvgItem,
)
from PyQt6.QtGui import (
    QMouseEvent,
    QResizeEvent,
    QTransform,
)


class EngineOrderTele(QGraphicsView):
    valueChanged = pyqtSignal(float)

    def __init__(self, parent=None, value_lim=100):
        super().__init__(parent)

        self.top_margin_rate = 0.07
        self.top_deadzone_rate = 0.44
        self.bottom_deadzone_rate = 0.54
        self.bottom_margin_rate = 0.1

        # this is also the reference value for scale ratio
        self.m_originalHeight = 240
        self.m_originalWidth = 140
        # transform original center
        self.m_originalHsiCtr = QPointF(120.0, 70.0)

        # z value is for stack order
        # high z value items are "on top of" low z values
        self.m_faceZ = -20
        self.m_caseZ = 20
        self.m_cursorZ = 10

        self.m_value = 0
        self.m_value_lim = value_lim
        self.m_itemFace = None
        self.m_itemCase = None
        self.m_itemCursor = None

        self.m_scaleX = None
        self.m_scaleY = None

        self.reset()
        self.m_scene = QGraphicsScene(self)
        self.setScene(self.m_scene)
        self.m_scene.clear()
        self.init()

        self.setStyleSheet("background:transparent;border:0px")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        #self.setFixedSize(self.m_originalWidth, self.m_originalHeight)
        self.setFixedSize(int(self.m_originalWidth*1.5), int(self.m_originalHeight*1.5))

    def reinit(self):
        if self.m_scene is not None:
            self.m_scene.clear()
            self.init()

    def update(self):
        self.updateView()

    def getValue(self) -> float:
        return self.m_value

    def setValue(
        self,
        value: float,
    ):
        if value > self.m_value_lim:
            value = self.m_value_lim
        elif value < -self.m_value_lim:
            value = -self.m_value_lim
        self.m_value = value
        self.valueChanged.emit(self.m_value)
        self.update()

    def sizeHint(self):
        return QSize(300, 300)

    def minimumSizeHint(self):
        return QSize(30, 30)

    def resizeEvent(
        self,
        event: QResizeEvent,
    ):
        super().resizeEvent(event)
        self.reinit()

    def init(self):
        self.m_scaleX = self.width() / self.m_originalWidth
        self.m_scaleY = self.height() / self.m_originalHeight
        self.reset()

        self.m_itemFace = QGraphicsSvgItem("./USVWidgets/svg/EngineOrderTelegraphFace.svg")
        self.m_itemFace.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemFace.setZValue(self.m_faceZ)
        self.m_itemFace.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemFace.setTransformOriginPoint(self.m_originalHsiCtr)
        self.m_scene.addItem(self.m_itemFace)

        self.m_itemCursor = QGraphicsSvgItem("./USVWidgets/svg/EngineOrderTelegraphCursor.svg")
        self.m_itemCursor.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemCursor.setZValue(self.m_cursorZ)
        self.m_itemCursor.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_scene.addItem(self.m_itemCursor)

        self.m_itemCase = QGraphicsSvgItem("./USVWidgets/svg/EngineOrderTelegraphCase.svg")
        self.m_itemCase.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemCase.setZValue(self.m_caseZ)
        self.m_itemCase.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemCase.setTransformOriginPoint(self.m_originalHsiCtr)
        self.m_scene.addItem(self.m_itemCase)

        super().centerOn(self.width()/2, self.height()/2)
        self.updateView()

    def reset(self):
        self.m_itemFace = 0
        self.m_itemCursor = 0
        self.m_value = 0.0

    def updateView(self):
        self.update_cursor_from_value()
        self.m_scene.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.m_dragging = True
            self.updateCursorFromMouse(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.m_dragging:
            self.updateCursorFromMouse(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.m_dragging = False
        super().mouseReleaseEvent(event)

    def update_cursor_from_value(self):
        cursor_y_position = self.value_to_height(self.m_value)
        self.m_itemCursor.setPos(self.width() / 2 - self.m_itemCursor.boundingRect().width() / 2, cursor_y_position - self.m_itemCursor.boundingRect().height() / 2)

    @property
    def top_margin_y(self):
        return self.top_margin_rate * self.height()

    @property
    def bottom_margin_y(self):
        return (1-self.bottom_margin_rate) * self.height()

    @property
    def top_deadzone_y(self):
        return self.top_deadzone_rate * self.height()

    @property
    def bottom_deadzone_y(self):
        return self.bottom_deadzone_rate * self.height()

    @property
    def top_margin_y(self):
        return self.top_margin_rate * self.height()

    def value_to_height(self, value):
        if value == 0:
            return (self.top_deadzone_y + self.bottom_deadzone_y)/2
        elif value > 0:
            rate = value / self.m_value_lim
            return rate * self.top_margin_y + (1-rate) * self.top_deadzone_y
        else:
            rate = abs(value / self.m_value_lim)
            return rate * self.bottom_margin_y + (1-rate) * self.bottom_deadzone_y

    def height_to_value(self, height):
        if height < self.top_margin_y:
            return self.m_value_lim
        elif height >= self.top_margin_y and height < self.top_deadzone_y:
            rate = (height - self.top_deadzone_y) / (self.top_margin_y - self.top_deadzone_y)
            return self.m_value_lim * rate
        elif height >= self.top_deadzone_y and height <= self.bottom_deadzone_y:
            return 0
        elif height > self.bottom_deadzone_y and height <= self.bottom_margin_y:
            rate = (height - self.bottom_deadzone_y) / (self.bottom_margin_y - self.bottom_deadzone_y)
            return -self.m_value_lim * rate
        else:
            return - self.m_value_lim

    def updateCursorFromMouse(self, pos):
        y = pos.y()
    
        new_value = self.height_to_value(y)
        self.setValue(new_value)
    