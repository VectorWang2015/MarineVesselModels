from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QTimer

from USVWidgets.palette_dark import darkPalette
from USVWidgets.control import EngineOrderTele

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()

        layout = QVBoxLayout()
        self.eot = EngineOrderTele()
        self.eot.valueChanged.connect(self.eot_changed)
        self.label = QLabel("0")

        layout.addWidget(self.eot)
        layout.addWidget(self.label)
        self.central_widget.setLayout(layout)

        self.setCentralWidget(self.central_widget)

    def eot_changed(self, value):
        self.label.setText(f"{value}")


def main():
    app = QApplication([]) 
    app.setStyle("Fusion")
    app.setPalette(darkPalette)

    # 创建并显示主窗口
    main_window = MainWindow()
    main_window.show()

    app.exec()

if __name__ == "__main__":
    main()
