#encoding: utf-8
#author: vectorwang@hotmail.com

import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from collections import namedtuple

from USVWidgets.palette_dark import darkPalette


kvlcc_30pix_poly = QtGui.QPolygon([
        QtCore.QPoint(-3, -14), QtCore.QPoint(-3, 11),
        QtCore.QPoint(0, 15), QtCore.QPoint(3, 11), QtCore.QPoint(3, -14),
])

usv_30pix_poly = QtGui.QPolygon([
        QtCore.QPoint(-2.5, -5), QtCore.QPoint(-2.5, -15), QtCore.QPoint(-7.5, -15), QtCore.QPoint(-7.5, 12),
        QtCore.QPoint(-5, 15), QtCore.QPoint(-2.5, 12), QtCore.QPoint(-2.5, 5),
        QtCore.QPoint(2.5, 5), QtCore.QPoint(2.5, 12), QtCore.QPoint(5, 15), 
        QtCore.QPoint(7.5, 12), QtCore.QPoint(7.5, -15), QtCore.QPoint(2.5, -15), QtCore.QPoint(2.5, -5),
])

class ShipCanvas(QtWidgets.QLabel):
    def __init__(
              self,
              *args,
              #enable_central = True,
              pool_width = 1600,
              pool_height = 900,
              #ship_length = 7,
              #ship_height = 2,
              meter_to_pixel_scale = 10,
              model=kvlcc_30pix_poly,
              **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # default displaying with 2.9m length and 0.6m width
        self.ship_poly = model
        self.x = pool_width/2
        self.y = pool_height/2
        self.pool_width = pool_width
        self.pool_height = pool_height
        self.psi = -180
        self.maxPt = 500
        self.pts = []
        # to plot key points as circle
        self.circle_every = 50
        # to make key points relatively static
        self.circle_offset = 0
        # disable central not tested yet
        self.enable_central = True
        self.scale = meter_to_pixel_scale

        self.setFixedSize(self.pool_width, self.pool_height)

    def update_ship_state(self, x, y, psi):
        self.x = y * self.scale + self.pool_width/2
        self.y = - x * self.scale + self.pool_height/2
        self.psi = psi / 2 / np.pi * 360 - 180

        if len(self.pts) >= self.maxPt:
            del(self.pts[0])
            self.circle_offset += 1
            self.circle_offset %= self.circle_every
        self.pts.append((self.x, self.y))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QColor("#65A6D1"))
        painter.setPen(QtGui.QColor("#65A6D1"))

        painter.drawRect(QtCore.QRect(0, 0, self.width(), self.height()))
        painter.end()

        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(0, 0, 0))
        if len(self.pts) > 0:
            for i in range(len(self.pts)-1):
                x0, y0 = self.pts[i]
                x1, y1 = self.pts[i+1]
                if self.enable_central:
                    painter.drawLine(
                        QtCore.QPointF(x0-self.x+self.pool_width/2, y0-self.y+self.pool_height/2),
                        QtCore.QPointF(x1-self.x+self.pool_width/2, y1-self.y+self.pool_height/2),
                    )
                    if (i+self.circle_offset) % self.circle_every == 0:
                        painter.drawEllipse(
                            QtCore.QPointF(x1-self.x+self.pool_width/2, y1-self.y+self.pool_height/2),
                            2,
                            2,
                        )
                else:
                    painter.drawLine(x0, y0, x1, y1)
                    if (i+self.circle_offset) % self.circle_every == 0:
                        painter.drawEllipse(
                            QtCore.QPointF(x1-self.x+self.pool_width/2, y1-self.y+self.pool_height/2),
                            2,
                            2,
                        )
        painter.end()

        # transform ship polygon
        rotate_trans = QtGui.QTransform()
        rotate_trans.rotate(self.psi)
        translate_trans = QtGui.QTransform()
        if self.enable_central:
            translate_trans.translate(self.pool_width/2, self.pool_height/2)
        else:
            translate_trans.translate(self.x, self.y)
        ship_shape = rotate_trans.map(self.ship_poly)
        ship_shape = translate_trans.map(ship_shape)

        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QColor(255, 255, 0))
        painter.setPen(QtGui.QColor(0, 0, 0))
        painter.drawPolygon(ship_shape)
        painter.end()