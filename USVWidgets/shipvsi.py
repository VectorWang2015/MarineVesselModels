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


class ShipVSI(QGraphicsView):
    def __init__(self, parent=None, face_svg="vsi_face_lim10.svg", v_lim=10):
        super().__init__(parent)
        self.face_svg = face_svg
        self.v_lim = v_lim

        # default width, height = 240, 240
        # this is also the reference value for scale ratio
        self.m_originalHeight = 240
        self.m_originalWidth = 240
        # transform original center
        self.m_originalVsiCtr = QPointF(120.0, 120.0)

        # z value is for stack order
        # high z value items are "on top of" low z values
        self.m_faceZ = -20
        self.m_caseZ = 10
        self.m_handZ = 0

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
        self.m_itemHand = None

        self.m_v = None
        self.m_scaleX = None
        self.m_scaleY = None

    def reinit(self):
        if self.m_scene is not None:
            self.m_scene.clear()
            self.init()

    def update(self):
        self.updateView()

    def getV(self) -> float:
        return self.m_v

    def setV(
        self,
        v: float,
    ):
        self.m_v = v
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

        self.m_itemFace = QGraphicsSvgItem("./USVWidgets/svg/"+self.face_svg)
        self.m_itemFace.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemFace.setZValue(self.m_faceZ)
        self.m_itemFace.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemFace.setTransformOriginPoint(self.m_originalVsiCtr)
        self.m_scene.addItem(self.m_itemFace)

        self.m_itemHand = QGraphicsSvgItem("./USVWidgets/svg/vsi_hand.svg")
        self.m_itemHand.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.m_itemHand.setZValue(self.m_handZ)
        self.m_itemHand.setTransform(
            QTransform.fromScale(self.m_scaleX, self.m_scaleY),
            True,
        )
        self.m_itemHand.setTransformOriginPoint(self.m_originalVsiCtr)
        self.m_scene.addItem(self.m_itemHand)

        self.m_itemCase = QGraphicsSvgItem("./USVWidgets/svg/vsi_case.svg")
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
        self.m_v = 0.0

    def updateView(self):
        if abs(self.m_v) >= self.v_lim:
            self.m_itemHand.setRotation(172)
        else:
            self.m_itemHand.setRotation(172/self.v_lim*self.m_v)
        self.m_scene.update()
