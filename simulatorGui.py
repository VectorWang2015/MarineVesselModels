import sys
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QPlainTextEdit, QSlider, QPushButton,
)

from USVWidgets.canvas import KVLCCCanvas
from USVWidgets.control import EngineOrderTele
from USVWidgets.palette_dark import darkPalette
from xinput_support import get_control

pool_width = 1600
pool_height = 900


class simulatorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        """
        canvas 1600x900 -> -80m~80m, -45mx45m
        - [ ] scale
        - [ ] rebuild ui
        """
        super().__init__()


        self.canvasLayout = QHBoxLayout()
        self.infoLayout = QVBoxLayout()
        self.inputLayout = QVBoxLayout()
        self.eotLayout = QHBoxLayout()
        self.buttonsLayout = QHBoxLayout()

        self.canvas = KVLCCCanvas(pool_height=pool_height, pool_width=pool_width)

        self.StartButton = QPushButton("Start")
        self.SuspendButton = QPushButton("Suspend")
        self.SuspendButton.setEnabled(False)
        self.Switch = QPushButton("Using EOT")

        self.InfoBox = QPlainTextEdit("Debug info:\n")
        self.InfoBox.setReadOnly(True)
        self.InfoBox.setMaximumHeight(200)

        self.left_EOT = EngineOrderTele(value_lim=100)
        self.right_EOT = EngineOrderTele(value_lim=100)
        self.left_EOT.setEnabled(False)
        self.right_EOT.setEnabled(False)

        self.canvasLayout.addWidget(self.canvas)
        self.canvasLayout.addLayout(self.infoLayout)

        self.infoLayout.addWidget(self.InfoBox)
        self.infoLayout.addStretch()
        self.infoLayout.addLayout(self.inputLayout)
        self.infoLayout.addStretch()
        self.infoLayout.addLayout(self.buttonsLayout)

        self.inputLayout.addLayout(self.eotLayout)
        self.inputLayout.addWidget(self.Switch)

        self.eotLayout.addWidget(self.left_EOT)
        self.eotLayout.addWidget(self.right_EOT)
        self.buttonsLayout.addWidget(self.StartButton)
        self.buttonsLayout.addWidget(self.SuspendButton)

        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.canvasLayout)
        self.setCentralWidget(self.centralWidget)


        self.StartButton.clicked.connect(self.start_clicked)
        self.SuspendButton.clicked.connect(self.suspend_clicked)
        self.Switch.clicked.connect(self.switch_clicked)

        self.l_thrust_input = 0
        self.r_thrust_input = 0

        self.using_joystick = False

        self.left_EOT.setValue(self.l_thrust_input)
        self.right_EOT.setValue(self.r_thrust_input)
        self.left_EOT.valueChanged.connect(self.left_EOT_changed)
        self.right_EOT.valueChanged.connect(self.right_EOT_changed)

        """
        self.CurrentControlSlider.setValue(self.delta)
        self.InputControlSlider.setValue(self.desire_delta)

        self.InputControlSlider.valueChanged.connect(self.slider_input_changed)

        """
        self.first_start = True
        # msec
        self.simul_time_period = 10
        self.ui_refresh = 50
        self.control_refresh = 20

        self.simul_timer = QtCore.QTimer(self)
        self.ui_timer = QtCore.QTimer(self)
        self.control_timer = QtCore.QTimer(self)

        #self.simul_timer.setInterval(self.simul_time_period)
        #self.ui_timer.setInterval(self.ui_refresh)
        self.control_timer.setInterval(self.control_refresh)

        #self.simul_timer.timeout.connect(self.simulator_step)
        #self.ui_timer.timeout.connect(self.refresh_canvas)
        self.control_timer.timeout.connect(self.refresh_control)

    def suspend_clicked(self):
        self.StartButton.setEnabled(True)
        self.SuspendButton.setEnabled(False)
        self.left_EOT.setEnabled(False)
        self.right_EOT.setEnabled(False)

        self.simul_timer.stop()
        self.ui_timer.stop()
        self.control_timer.stop()

        self.setStyleSheet(
            "QPlainTextEdit {background-color: red;}"
        )

    def start_clicked(self):
        if self.first_start:
            self.first_start = False

            """
            u = float(self.uEdit.text())
            v = float(self.vEdit.text())
            psi = float(self.psiEdit.text())

            self.uEdit.setEnabled(False)
            self.vEdit.setEnabled(False)
            self.psiEdit.setEnabled(False)

            self.current_state = np.array([0, 0, psi, u, v, 0]).reshape([6, 1])
            # self.current_state = ShipState(x=0, y=0, psi=psi, u=u, v=v, r=0)
            # self.simulator = KVLCC2(
            #     x=0,
            #     y=0,
            #     u=u,
            #     v=v,
            #     psi=psi,
            #     fixed_time_step=self.simul_time_period / 1000,
            # )
            """

        self.StartButton.setEnabled(False)
        self.SuspendButton.setEnabled(True)

        self.left_EOT.setEnabled(True)
        self.right_EOT.setEnabled(True)

        self.simul_timer.start()
        self.ui_timer.start()
        self.control_timer.start()

        self.setStyleSheet(
            "QPlainTextEdit {background-color: green;}"
        )

    def switch_clicked(self):
        if self.using_joystick:
            self.using_joystick = False
            self.Switch.setText("Using EOT")
        else:
            self.using_joystick = True
            self.Switch.setText("Using Joystick")

    def left_EOT_changed(self, v):
        if not self.using_joystick:
            self.l_thrust_input = v
            self.add_line_to_info(f"New l_thrust_input: {v}")

    def right_EOT_changed(self, v):
        if not self.using_joystick:
            self.r_thrust_input = v
            self.add_line_to_info(f"New r_thrust_input: {v}")

    def add_line_to_info(self, line):
        self.InfoBox.appendPlainText(line)

    def refresh_control(self):
        if self.using_joystick:
            joystick_cmd = get_control(reverse=False)
            if joystick_cmd is not None:
                l_thrust, r_thrust = joystick_cmd
                self.l_thrust_input = l_thrust / 255 * 100
                self.r_thrust_input = r_thrust / 255 * 100
                self.left_EOT.setValue(l_thrust)
                self.right_EOT.setValue(r_thrust)
                self.add_line_to_info(f"New thrust_input: {self.l_thrust_input}, {self.r_thrust_input}")
            else:
                self.add_line_to_info("Xbox controller undetected!")

    """
    def simulator_step(self):
        delta = self.delta / 180 * np.pi
        #print(delta)
        #self.current_state = self.simulator.step(delta)

    def refresh_canvas(self):
        heading = self.current_state[2][0] / np.pi * 180
        self.canvas.update_ship_state(self.current_state)
    """


app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
app.setPalette(darkPalette)
mw = simulatorWindow()
mw.show()
app.exec()