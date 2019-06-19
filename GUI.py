import sys
import Predict

from PyQt5.QtCore import QCoreApplication, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, QDesktopWidget,
    QMessageBox)


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def closeWindow(self):
        sys.exit(app.exec_())

    def initUI(self):
        recommendMoviesButton = QPushButton("推荐电影")
        recommendMoviesButton.clicked.connect(self.recommendMovies)

        exitSystemButton = QPushButton("退出系统")
        exitSystemButton.clicked.connect(self.exitSystem)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(recommendMoviesButton)
        hbox.addWidget(exitSystemButton)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')
        self.show()

    # 创建鼠标点击事件
    @pyqtSlot()
    def recommendMovies(self):
        print("➤开始执行推荐电影算法")
        Predict.runPredict(234, 1401, 20, 10, 20)

    @pyqtSlot()
    def exitSystem(self):
        sys.exit(app.exec_())

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            'Message',
            "Are you sure to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 控制窗口显示在屏幕中心的方法
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    # 每一pyqt5应用程序必须创建一个应用程序对象。sys.argv参数是一个列表，从命令行输入参数。
    app = QApplication(sys.argv)
    # QWidget部件是pyqt5所有用户界面对象的基类。他为QWidget提供默认构造函数。默认构造函数没有父类。
    w = Window()

    # resize()方法调整窗口的大小。
    w.resize(1000, 600)
    # 窗口居中
    w.center()
    # 窗口固定大小
    w.setFixedSize(w.width(), w.height())

    # 设置窗口的标题
    w.setWindowTitle('个性化电影推荐系统')
    # 展示窗口
    w.show()

    sys.exit(app.exec_())
