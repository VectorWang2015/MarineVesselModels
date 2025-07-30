#encoding: utf-8
#author: vectorwang@hotmail.com

from PyQt6.QtWidgets import (
    QWidget, QGraphicsScene,
    QGraphicsView, QGraphicsItem,
)
from PyQt6.QtCore import (
    QPointF, Qt, QSize, QTimer
)
from PyQt6.QtSvgWidgets import (
    QGraphicsSvgItem,
)
from PyQt6.QtGui import (
    QResizeEvent,
    QTransform,
)


class PropellerLaser(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # default width, height = 240, 240
        # this is also the reference value for scale ratio
        self.m_originalHeight = 240
        self.m_originalWidth = 240
        self.m_refreshInterval = 20
        # transform original center
        self.m_originalPropCtr = QPointF(120.0, 120.0)

        self.m_rpm = 0
        self.m_angle = 0
        self.m_itemProp = None
        self.m_scaleX = None
        self.m_scaleY = None


        #self.reset()
        self.m_scene = QGraphicsScene(self)
        self.setScene(self.m_scene)
        self.m_scene.clear()
        self.init()

        self.setStyleSheet("background:#303030;border:0px")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.timer = QTimer()
        self.timer.timeout.connect(self.propRotate)
        self.timer.start(self.m_refreshInterval)

    def reinit(self):
        if self.m_scene is not None:
            self.m_scene.clear()
            self.init()

    def update(self):
        self.updateView()

    def getRpm(self) -> float:
        return self.m_rpm

    def setRpm(
        self,
        rpm: float,
    ):
        self.m_rpm = rpm
        #self.update()

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
        #self.reset()

        self.m_itemProp = QGraphicsSvgItem("./USVWidgets/svg/propeller_laser.svg")
        self.m_itemProp.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        #self.m_itemProp.setZValue(self.m_faceZ)
        self.m_itemProp.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemProp.setTransformOriginPoint(self.m_originalPropCtr)
        self.m_scene.addItem(self.m_itemProp)

        super().centerOn(self.width()/2, self.height()/2)
        self.updateView()

    def updateView(self):
        self.m_itemProp.setRotation(self.m_angle)
        self.m_scene.update()

    def propRotate(self):
        self.m_angle += self.m_rpm * 360 / 60 / 1000 * self.m_refreshInterval
        self.updateView()
