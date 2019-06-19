import sys
# import Predict

from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from PyQt5.QtCore import QCoreApplication, pyqtSlot, QRegExp
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, QDesktopWidget,
    QMessageBox, QLineEdit, QLabel, QGridLayout, QFrame, QTextEdit)


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

        # 用户ID输入框
        userLabel = QLabel(self)
        userLabel.setText("用户Id：")
        userLabel.move(20, 10)
        userLabel.setStyleSheet("color:red;")

        user_validato = QIntValidator(1, 6040, self)  # 实例化整型验证器，并设置范围为1-6040
        userLineEdit = QLineEdit(self)  # 整型文本框
        userLineEdit.setValidator(user_validato)  # 设置验证
        userLineEdit.setFixedSize(100, 30)
        userLineEdit.move(70, 10)

        # 用户信息显示框
        userInfLabel = QLabel(self)
        userInfLabel.setText("用户信息：")
        userInfLabel.move(200, 10)

        userInfEdit = QLineEdit(self)
        userInfEdit.setFixedSize(600, 30)
        userInfEdit.move(265, 10)
        userInfEdit.setDisabled(True)
        userInfEdit.setText("请先输入用户Id")

        # 电影ID输入框------------------------------------------------------------
        movieLabel = QLabel(self)
        movieLabel.setText("电影Id：")
        movieLabel.move(20, 50)
        movieLabel.setStyleSheet("color:red;")

        movie_validato = QIntValidator(1, 3952, self)  # 实例化整型验证器，并设置范围为1-3952
        movieLineEdit = QLineEdit(self)  # 整型文本框
        movieLineEdit.setValidator(movie_validato)  # 设置验证
        movieLineEdit.setFixedSize(100, 30)
        movieLineEdit.move(70, 50)

        # 电影信息显示框
        movieInfLabel = QLabel(self)
        movieInfLabel.setText("电影信息：")
        movieInfLabel.move(200, 50)

        movieInfEdit = QLineEdit(self)
        movieInfEdit.setFixedSize(600, 30)
        movieInfEdit.move(265, 50)
        movieInfEdit.setDisabled(True)
        movieInfEdit.setText("请先输入电影Id")

        # 预测评分显示框------------------------------------------------------------
        predictScoreLabel = QLabel(self)
        predictScoreLabel.setText("电影预测评分：")
        predictScoreLabel.move(20, 90)

        predictScoreEdit = QLineEdit(self)
        predictScoreEdit.setFixedSize(300, 30)
        predictScoreEdit.move(110, 90)
        predictScoreEdit.setDisabled(True)
        predictScoreEdit.setText("请先执行推荐算法")

        # ------------------------------------------------------------
        # 推荐同类型的电影
        resultLabel1 = QLabel(self)
        resultLabel1.setText("同类型的电影还有：")
        resultLabel1.move(20, 130)
        resultLabel1.setFixedSize(230, 30)

        resultEdit1 = QTextEdit(self)
        resultEdit1.setFixedSize(250, 380)
        resultEdit1.move(0, 170)
        resultEdit1.setDisabled(True)
        resultEdit1.setText("请先执行推荐算法")

        # 推荐用户可能喜欢的电影
        resultLabel2 = QLabel(self)
        resultLabel2.setText("您可能喜欢的电影有：")
        resultLabel2.move(270, 130)
        resultLabel2.setFixedSize(230, 30)

        resultEdit2 = QTextEdit(self)
        resultEdit2.setFixedSize(250, 380)
        resultEdit2.move(250, 170)
        resultEdit2.setDisabled(True)
        resultEdit2.setText("请先执行推荐算法")

        # 推荐喜欢该电影的人
        resultLabel3 = QLabel(self)
        resultLabel3.setText("喜欢该电影的人有：")
        resultLabel3.move(520, 130)
        resultLabel3.setFixedSize(230, 30)

        resultEdit3 = QTextEdit(self)
        resultEdit3.setFixedSize(250, 380)
        resultEdit3.move(500, 170)
        resultEdit3.setDisabled(True)
        resultEdit3.setText("请先执行推荐算法")

        # 推荐喜欢该电影的人还喜欢看的电影
        resultLabel4 = QLabel(self)
        resultLabel4.setText("喜欢该电影的人还喜欢看：")
        resultLabel4.move(770, 130)
        resultLabel4.setFixedSize(230, 30)

        resultEdit4 = QTextEdit(self)
        resultEdit4.setFixedSize(250, 380)
        resultEdit4.move(750, 170)
        resultEdit4.setDisabled(True)
        resultEdit4.setText("请先执行推荐算法")
        # ------------------------------------------------------------
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
        print()
        print("➤开始执行推荐电影算法")
        # Predict.runPredict(234, 1401, 20, 10, 20)

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
