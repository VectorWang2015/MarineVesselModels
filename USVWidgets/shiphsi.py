#encoding: utf-8
#author: vectorwang@hotmail.com

from PyQt6.QtWidgets import (
    QWidget, QGraphicsScene,
    QGraphicsView, QGraphicsItem,
)
from PyQt6.QtCore import (
    QPointF, Qt, QSize
)
from PyQt6.QtSvgWidgets import (
    QGraphicsSvgItem,
)
from PyQt6.QtGui import (
    QResizeEvent,
    QTransform,
)


class ShipHSI(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # default width, height = 240, 240
        # this is also the reference value for scale ratio
        self.m_originalHeight = 240
        self.m_originalWidth = 240
        # transform original center
        self.m_originalHsiCtr = QPointF(120.0, 120.0)

        # z value is for stack order
        # high z value items are "on top of" low z values
        self.m_faceZ = -20
        self.m_caseZ = 10

        self.reset()
        self.m_scene = QGraphicsScene(self)
        self.setScene(self.m_scene)
        self.m_scene.clear()
        self.init()

        self.setStyleSheet("background:transparent;border:0px")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.m_itemFace = None
        self.m_itemCase = None

        self.m_heading = None
        self.m_scaleX = None
        self.m_scaleY = None

    def reinit(self):
        if self.m_scene is not None:
            self.m_scene.clear()
            self.init()

    def update(self):
        self.updateView()

    def getHeading(self) -> float:
        return self.m_heading

    def setHeading(
        self,
        heading: float,
    ):
        self.m_heading = heading
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

        self.m_itemFace = QGraphicsSvgItem("./USVWidgets/svg/hsi_face.svg")
        self.m_itemFace.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemFace.setZValue(self.m_faceZ)
        self.m_itemFace.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemFace.setTransformOriginPoint(self.m_originalHsiCtr)
        self.m_scene.addItem(self.m_itemFace)

        self.m_itemCase = QGraphicsSvgItem("./USVWidgets/svg/ship_hsi_case.svg")
        self.m_itemCase.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemCase.setZValue(self.m_caseZ)
        self.m_itemCase.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_scene.addItem(self.m_itemCase)

        super().centerOn(self.width()/2, self.height()/2)
        self.updateView()

    def reset(self):
        self.m_itemFace = 0
        self.m_itemCase = 0
        self.m_heading = 0.0

    def updateView(self):
        self.m_itemFace.setRotation(- self.m_heading)
        self.m_scene.update()


class ShipHSIDoublePlate(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # default width, height = 240, 240
        # this is also the reference value for scale ratio
        self.m_originalHeight = 240
        self.m_originalWidth = 240
        # transform original center
        self.m_originalHsiCtr = QPointF(120.0, 120.0)

        # z value is for stack order
        # high z value items are "on top of" low z values
        self.m_face1Z = -20
        self.m_face2Z = -10
        self.m_caseZ = 10

        self.reset()
        self.m_scene = QGraphicsScene(self)
        self.setScene(self.m_scene)
        self.m_scene.clear()
        self.init()

        self.setStyleSheet("background:transparent;border:0px")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.m_itemFace1 = None
        self.m_itemFace2 = None
        self.m_itemCase = None

        self.m_heading = None
        self.m_scaleX = None
        self.m_scaleY = None

    def reinit(self):
        if self.m_scene is not None:
            self.m_scene.clear()
            self.init()

    def update(self):
        self.updateView()

    def getHeading(self) -> float:
        return self.m_heading

    def setHeading(
        self,
        heading: float,
    ):
        self.m_heading = heading
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

        self.m_itemFace1 = QGraphicsSvgItem("./USVWidgets/svg/hsi_multi_face1.svg")
        self.m_itemFace1.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemFace1.setZValue(self.m_face1Z)
        self.m_itemFace1.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemFace1.setTransformOriginPoint(self.m_originalHsiCtr)
        self.m_scene.addItem(self.m_itemFace1)

        self.m_itemFace2 = QGraphicsSvgItem("./USVWidgets/svg/hsi_multi_face2.svg")
        self.m_itemFace2.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemFace2.setZValue(self.m_face2Z)
        self.m_itemFace2.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemFace2.setTransformOriginPoint(self.m_originalHsiCtr)
        self.m_scene.addItem(self.m_itemFace2)

        self.m_itemCase = QGraphicsSvgItem("./USVWidgets/svg/hsi_multi_case.svg")
        self.m_itemCase.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemCase.setZValue(self.m_caseZ)
        self.m_itemCase.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_scene.addItem(self.m_itemCase)

        super().centerOn(self.width()/2, self.height()/2)
        self.updateView()

    def reset(self):
        self.m_itemFace1 = 0
        self.m_itemFace2 = 0
        self.m_itemCase = 0
        self.m_heading = 0.0

    def updateView(self):
        self.m_itemFace1.setRotation(- self.m_heading)
        self.m_itemFace2.setRotation(- (self.m_heading % 10)*36)
        self.m_scene.update()
