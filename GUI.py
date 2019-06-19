import sys
import Predict
from Predict import movies_orig, movieid2idx, users_orig, userid2idx

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
        self.userLineEdit = QLineEdit(self)  # 整型文本框
        self.userLineEdit.setValidator(user_validato)  # 设置验证
        self.userLineEdit.setFixedSize(100, 30)
        self.userLineEdit.move(70, 10)

        # 用户信息显示框
        userInfLabel = QLabel(self)
        userInfLabel.setText("用户信息：")
        userInfLabel.move(200, 10)

        self.userInfEdit = QLineEdit(self)
        self.userInfEdit.setFixedSize(600, 30)
        self.userInfEdit.move(265, 10)
        self.userInfEdit.setDisabled(True)
        self.userInfEdit.setText("请先输入用户Id")

        # 电影ID输入框------------------------------------------------------------
        movieLabel = QLabel(self)
        movieLabel.setText("电影Id：")
        movieLabel.move(20, 50)
        movieLabel.setStyleSheet("color:red;")

        movie_validato = QIntValidator(1, 3952, self)  # 实例化整型验证器，并设置范围为1-3952
        self.movieLineEdit = QLineEdit(self)  # 整型文本框
        self.movieLineEdit.setValidator(movie_validato)  # 设置验证
        self.movieLineEdit.setFixedSize(100, 30)
        self.movieLineEdit.move(70, 50)

        # 电影信息显示框
        movieInfLabel = QLabel(self)
        movieInfLabel.setText("电影信息：")
        movieInfLabel.move(200, 50)

        self.movieInfEdit = QLineEdit(self)
        self.movieInfEdit.setFixedSize(600, 30)
        self.movieInfEdit.move(265, 50)
        self.movieInfEdit.setDisabled(True)
        self.movieInfEdit.setText("请先输入电影Id")

        # 预测评分显示框------------------------------------------------------------
        predictScoreLabel = QLabel(self)
        predictScoreLabel.setText("电影预测评分：")
        predictScoreLabel.move(20, 90)

        self.predictScoreEdit = QLineEdit(self)
        self.predictScoreEdit.setFixedSize(300, 30)
        self.predictScoreEdit.move(110, 90)
        self.predictScoreEdit.setDisabled(True)
        self.predictScoreEdit.setText("请先执行推荐算法")

        # ------------------------------------------------------------
        # 推荐同类型的电影
        resultLabel1 = QLabel(self)
        resultLabel1.setText("同类型的电影还有：")
        resultLabel1.move(20, 130)
        resultLabel1.setFixedSize(230, 30)

        self.resultEdit1 = QTextEdit(self)
        self.resultEdit1.setFixedSize(250, 400)
        self.resultEdit1.move(0, 170)
        self.resultEdit1.setDisabled(True)
        self.resultEdit1.setText("请先执行推荐算法")

        # 推荐用户可能喜欢的电影
        resultLabel2 = QLabel(self)
        resultLabel2.setText("您可能喜欢的电影有：")
        resultLabel2.move(270, 130)
        resultLabel2.setFixedSize(230, 30)

        self.resultEdit2 = QTextEdit(self)
        self.resultEdit2.setFixedSize(250, 400)
        self.resultEdit2.move(250, 170)
        self.resultEdit2.setDisabled(True)
        self.resultEdit2.setText("请先执行推荐算法")

        # 推荐喜欢该电影的人
        resultLabel3 = QLabel(self)
        resultLabel3.setText("喜欢该电影的人有：")
        resultLabel3.move(520, 130)
        resultLabel3.setFixedSize(230, 30)

        self.resultEdit3 = QTextEdit(self)
        self.resultEdit3.setFixedSize(250, 400)
        self.resultEdit3.move(500, 170)
        self.resultEdit3.setDisabled(True)
        self.resultEdit3.setText("请先执行推荐算法")

        # 推荐喜欢该电影的人还喜欢看的电影
        resultLabel4 = QLabel(self)
        resultLabel4.setText("喜欢该电影的人还喜欢看：")
        resultLabel4.move(770, 130)
        resultLabel4.setFixedSize(230, 30)

        self.resultEdit4 = QTextEdit(self)
        self.resultEdit4.setFixedSize(250, 400)
        self.resultEdit4.move(750, 170)
        self.resultEdit4.setDisabled(True)
        self.resultEdit4.setText("请先执行推荐算法")
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
        userIdInput = self.userLineEdit.text()
        movieIdInput = self.movieLineEdit.text()

        print(userIdInput, movieIdInput)

        if len(userIdInput) > 0 and len(movieIdInput) > 0:
            print()
            print("➤开始执行推荐电影算法")

            userIdInput = int(userIdInput)
            movieIdInput = int(movieIdInput)
            inferenceScore, stmResult, yfmResult, ofmResult, userList = Predict.runPredict(
                userIdInput, movieIdInput, 20, 10, 20)

            self.userInfEdit.setText(str(users_orig[userid2idx[userIdInput]]))
            self.movieInfEdit.setText(str(movies_orig[movieid2idx[movieIdInput]]))

            self.predictScoreEdit.setText(str(inferenceScore[0][0][0]))

            stmText = ""
            for val in (stmResult):
                stmText += str(val)
                stmText += str(movies_orig[val]) + "\n\n"
            self.resultEdit1.setText(stmText)

            yfmText = ""
            for val in (yfmResult):
                yfmText += str(val)
                yfmText += str(movies_orig[val]) + "\n\n"
            self.resultEdit2.setText(yfmText)

            likeMovieUsers = ""
            for val in (userList):
                likeMovieUsers += str(val) + "\n"
            self.resultEdit3.setText(likeMovieUsers)

            ofmText = ""
            for val in (ofmResult):
                ofmText += str(val)
                ofmText += str(movies_orig[val]) + "\n\n"
            self.resultEdit4.setText(ofmText)
        else:
            reply = QMessageBox.question(
                self,
                'Message',
                "输入不完整，请补全后再运行",
                QMessageBox.Yes,
                QMessageBox.Yes
            )

    @pyqtSlot()
    def exitSystem(self):
        sys.exit(app.exec_())

    # --------------------------------------------------------------------------
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
