import sys
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget, QFormLayout,
    QLabel, QPlainTextEdit, QSlider, QPushButton,
    QLineEdit,
)

from USVWidgets.canvas import ShipCanvas, usv_30pix_poly
from USVWidgets.control import EngineOrderTele
from USVWidgets.palette_dark import darkPalette
from xinput_support import get_control

from MarineVesselModels.simulator import FossenSimulator
from MarineVesselModels.thrusters import NaiveDoubleThruster

pool_width = 1600
pool_height = 900


class simulatorWindow(QtWidgets.QMainWindow):
    def __init__(self, hydro_params, b):
        """
        hydro_params: hydo parameters for simualtion
        b: width of the USV
        """
        super().__init__()
        self.hydro_params = hydro_params
        self.b = b

        self.canvasLayout = QHBoxLayout()
        self.infoLayout = QVBoxLayout()
        self.inputLayout = QVBoxLayout()
        self.eotLayout = QHBoxLayout()
        self.buttonsLayout = QHBoxLayout()
        self.uvrLayout = QFormLayout()

        self.canvas = ShipCanvas(pool_height=pool_height, pool_width=pool_width, meter_to_pixel_scale=30, model=usv_30pix_poly)

        self.StartButton = QPushButton("Start")
        self.SuspendButton = QPushButton("Suspend")
        self.SuspendButton.setEnabled(False)
        self.Switch = QPushButton("Using EOT")

        self.InfoBox = QPlainTextEdit("Debug info:\n")
        self.InfoBox.setReadOnly(True)
        self.InfoBox.setMaximumHeight(200)

        self.Lu = QLabel("u: ")
        self.Lv = QLabel("v: ")
        self.Lr = QLabel("r: ")
        self.LEu = QLineEdit()
        self.LEv = QLineEdit()
        self.LEr = QLineEdit()
        self.LEu.setReadOnly(True)
        self.LEv.setReadOnly(True)
        self.LEr.setReadOnly(True)

        self.uvrLayout.addRow(self.Lu, self.LEu)
        self.uvrLayout.addRow(self.Lv, self.LEv)
        self.uvrLayout.addRow(self.Lr, self.LEr)

        self.left_EOT = EngineOrderTele(value_lim=100)
        self.right_EOT = EngineOrderTele(value_lim=100)
        self.left_EOT.setEnabled(False)
        self.right_EOT.setEnabled(False)

        self.canvasLayout.addWidget(self.canvas)
        self.canvasLayout.addLayout(self.infoLayout)

        self.infoLayout.addWidget(self.InfoBox)
        self.infoLayout.addLayout(self.uvrLayout)
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

        self.first_start = True
        self.simulating = False
        # msec
        self.simul_time_period = 10
        self.ui_refresh = 50
        self.control_refresh = 20

        self.simul_timer = QtCore.QTimer(self)
        self.ui_timer = QtCore.QTimer(self)
        self.control_timer = QtCore.QTimer(self)

        self.simul_timer.setInterval(self.simul_time_period)
        self.ui_timer.setInterval(self.ui_refresh)
        self.control_timer.setInterval(self.control_refresh)

        self.simul_timer.timeout.connect(self.simulator_step)
        self.ui_timer.timeout.connect(self.refresh_canvas)
        self.control_timer.timeout.connect(self.refresh_control)

    def suspend_clicked(self):
        self.simulating = False
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
        self.setStyleSheet(
            "QLineEdit {background-color: red;}"
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
            """

            #self.current_state = np.array([0, 0, psi, u, v, 0]).reshape([6, 1])
            self.current_state = np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
            self.simulator = FossenSimulator(
                self.hydro_params,
                time_step=self.simul_time_period/1000,
                init_state=self.current_state,
            )
            self.thruster = NaiveDoubleThruster(b=self.b)

            # monitor keyboard
            self.installEventFilter(self)

        self.simulating = True
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
        self.setStyleSheet(
            "QLineEdit {background-color: green;}"
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

    def update_uvr(self):
        u = self.current_state[3][0],
        v = self.current_state[4][0],
        r = self.current_state[5][0],
        self.LEu.setText(f"{u}")
        self.LEv.setText(f"{v}")
        self.LEr.setText(f"{r}")

    def simulator_step(self):
        left_force = self.l_thrust_input / 100 * 5.0
        right_force = self.r_thrust_input / 100 * 5.0
        tau = self.thruster.newton_to_tau(left_force, right_force)
        self.add_line_to_info(f"tau: {tau}")
        self.current_state = self.simulator.step(tau)

    def refresh_canvas(self):
        #self.heading = self.current_state[2][0] / np.pi * 180
        self.canvas.update_ship_state(
            self.current_state[0][0],
            self.current_state[1][0],
            self.current_state[2][0],
        )
        self.update_uvr()

    def clip_and_set_EOT(self):
        if self.l_thrust_input > 100:
            self.l_thrust_input = 100
        elif self.l_thrust_input < -100:
            self.l_thrust_input = -100
        if self.r_thrust_input > 100:
            self.r_thrust_input = 100
        elif self.r_thrust_input < -100:
            self.r_thrust_input = -100
        self.left_EOT.setValue(self.l_thrust_input)
        self.right_EOT.setValue(self.r_thrust_input)


    def eventFilter(self, obj, event):
        if event.type() == event.Type.KeyPress and not self.using_joystick and self.simulating:
            key = event.key()
            if key == QtCore.Qt.Key.Key_W:
                self.l_thrust_input += 5
                self.r_thrust_input += 5
                self.clip_and_set_EOT()
                return True  # 表示事件已处理
            elif key == QtCore.Qt.Key.Key_S:
                self.l_thrust_input -= 5
                self.r_thrust_input -= 5
                self.clip_and_set_EOT()
                return True  # 表示事件已处理
            elif key == QtCore.Qt.Key.Key_A:
                self.l_thrust_input -= 5
                self.r_thrust_input += 5
                self.clip_and_set_EOT()
                return True  # 表示事件已处理
            elif key == QtCore.Qt.Key.Key_D:
                self.l_thrust_input += 5
                self.r_thrust_input -= 5
                self.clip_and_set_EOT()
                return True  # 表示事件已处理
        return super().eventFilter(obj, event)

# data from <Research on Parameter Identification Method of Four-Thrusters AUSV Dynamics Model>
# 此参数会自激旋转,可能原因为非线性项缺失
b = 0.5
hydro_params = {
    "d11": 6.0,
    "d22": 7.1,
    "d33": 0.8,
    "m11": 13.0,
    "m22": 23.3,
    "m33": 1.3,
}

app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
app.setPalette(darkPalette)
mw = simulatorWindow(hydro_params=hydro_params, b=b)
mw.show()
app.exec()