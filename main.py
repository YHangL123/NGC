import os
import sys
from PyQt5.QtCore import Qt, QDate, QDateTime, QTime, QTimer, QSettings
from PyQt5.QtGui import QTextCursor
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QMessageBox, QInputDialog, QWidget
from UI.main_window import Ui_Main_window
from UI.ana_window import Ui_ana_window
from PyQt5.QtCore import QTextStream
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QMetaObject
from resources.LR.main_LR import main_lr, LR_pre
from resources.ANN.main_pso_ANN import main_pso_ann, ANN_pre
from resources.SVR.main_svr import train_model_svr,predict_RUL_model_svr,predict_model_svr
from resources.FNN.main_FNN import train_model_fnn,predict_model_fnn,predict_RUL_model_fnn
from resources.LSTM.main_lstm import  train_model_lstm,predict_model_lstm,predict_RUL_model_lstm
from resources.ARIMA.main_Parima import main_parima,ARIMA_pre
from resources.STFT.main_stft import train_model_stft,confusion_test_stft
from resources.CWT.main_cwt import train_model_cwt,confusion_test_cwt
import queue
import threading


class Main_window(QWidget, Ui_Main_window):
    def __init__(self, parent=None):
        super(Main_window, self).__init__(parent)
        self.setupUi(self)
        self.current_ana_page = None

    def ana_show(self):
        sender = self.sender()
        # 获得当前算法名称
        btn_name = sender.objectName()

        # 关闭旧的算法界面（如果存在）
        if self.current_ana_page:
            self.current_ana_page.close()

        # 传递名称
        self.ana_page = Ana_window(btn_name)
        self.ana_page.show()


# class ConsoleOutput:
#     def __init__(self, text_browser):
#         self.text_browser = text_browser
#
#     def write(self, text):
#         cursor = self.text_browser.textCursor()
#         cursor.movePosition(QTextCursor.End)
#         cursor.insertText(text)
#         self.text_browser.setTextCursor(cursor)
#         self.text_browser.ensureCursorVisible()

class Ana_window(QWidget, Ui_ana_window):
    def __init__(self, name, parent=None):
        super(Ana_window, self).__init__(parent)
        self.setupUi(self)
        self.name = name
        # 按钮-名称映射
        map_1 = {
            'A_1': '基于LR的重载齿轮箱故障预测技术',
            'A_2': '基于PSO-ANN的重载齿轮箱故障预测技术',
            'A_3': '基于AGT的重载齿轮箱故障预测阈值计算智能算法',
            'A_4': '基于RVM的故障预测阈值计算智能算法',
            'A_5': '基于FNN的重载齿轮箱寿命预估模型',
            'A_6': '基于SVR的重载齿轮箱寿命预估模型',
            'A_7': '基于LSTM的重载齿轮箱寿命预估模型',
            'A_8': '基于短时傅里叶变换和二维卷积神经网络的重载齿轮箱故障诊断模型',
            'A_9': '基于连续小波变换和二维卷积神经网络的重载齿轮箱故障诊断模型',
            'A_10': '基于ARIMA的重载齿轮箱寿命预估模型',
            'A_11': '基于机器学习分类器声学技术的重载齿轮箱故障诊断模型',
            'A_12': '基于深度卷积网络的智能化故障诊断模型诊断模型',
            'A_13': '基于深度信念网络和红外热成像的智能化故障诊断网络模型',
            'A_14': '基于PSO-ANN的重载齿轮箱故障预测技术',
            'A_15': '基于贝叶斯算法的重载齿轮箱寿命预估模型',
            'A_16': '基于集成学习算法的重载齿轮箱故障诊断模型'

        }
        # 通过映射关系拿到当前算法名称
        self.al_name = map_1[self.name]

        self.name_label.setText(self.al_name)
        # 创建并启动线程
        self.worker_thread_t = WorkerThread_T(self.name, self.textBrowser)
        self.worker_thread_t.output_ready.connect(self.on_output_ready_train)
        self.worker_thread_p = WorkerThread_P(self.name)
        self.worker_thread_p.output_ready.connect(self.on_output_ready_P)

    def closeEvent(self, event):
        self.worker_thread_p.quit()
        self.worker_thread_p.wait()
        self.worker_thread_t.quit()
        self.worker_thread_t.wait()

        event.accept()

    def select_data(self):
        # 打开文件夹
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选择文件夹", os.getcwd())
        self.file_path_1.clear()
        self.file_path_1.setText(folder_path)

    def select_model(self):
        # # 打开文件
        # fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", os.getcwd(),
        #                                                            "All Files(*);;")
        # 打开文件夹
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选择文件夹", os.getcwd())
        self.file_path_2.clear()
        self.file_path_2.setText(folder_path)

    def start_show(self):
        name_btn = self.name
        map_2 = {
            'A_1': 'LR',
            'A_2': 'ANN',
            'A_3': 'AGT',
            'A_4': 'RVM',
            'A_5': 'FNN',
            'A_6': 'SVR',
            'A_7': 'LSTM',
            'A_8': 'STFT',
            'A_9': 'CWT',
            'A_10': 'ARIMA',
            'A_11': '基于机器学习分类器声学技术的重载齿轮箱故障诊断模型',
            'A_12': '基于深度卷积网络的智能化故障诊断模型诊断模型',
            'A_13': '基于深度信念网络和红外热成像的智能化故障诊断网络模型',
            'A_14': '基于PSO-ANN的重载齿轮箱故障预测技术',
            'A_15': '基于贝叶斯算法的重载齿轮箱寿命预估模型',
            'A_16': '基于集成学习算法的重载齿轮箱故障诊断模型'
        }
        name = map_2[name_btn]

        with open(f'D:/Python projects/al_show/resources/{name}/logger/{name}log.txt') as file:
            for row in file:
                self.textBrowser_3.append(row)

        self.res_scene = QGraphicsScene()
        self.graphicsView.setScene(self.res_scene)
        path = f'D:/Python projects/al_show/resources/{name}/PIC/{name}.png'
        if path:
            pixmap = QPixmap(path)
            self.res_scene.addPixmap(pixmap)


    def start_train(self):
        self.textBrowser.clear()
        self.textBrowser.append('正在训练中，请稍后...\n')
        # 数据加载（相对路径）
        data_to_path = self.file_path_1.text()
        self.worker_thread_t.data_path = data_to_path

        self.worker_thread_t.start()

    def start_predict(self):
        self.textBrowser_2.clear()
        self.textBrowser_2.append('正在预测中，请稍后...\n')
        data_to_path = self.file_path_2.text()
        self.worker_thread_p.data_path = data_to_path

        self.worker_thread_p.start()

    # 处理输出信息的槽函数，更新 textBrowser
    def on_output_ready_train(self, output_str):
        self.textBrowser.append(output_str)

        # self.textBrowser.append(str(self.worker_thread_t.currentThreadId()))

    # 处理输出信息的槽函数，更新 textBrowser
    def on_output_ready_P(self, output_str):
        self.textBrowser_2.append(output_str)
        # self.textBrowser_2.append(str(self.worker_thread_p.currentThreadId()))


class WorkerThread_T(QThread):
    output_ready = pyqtSignal(str)  # 用于传递输出信息的信号

    def __init__(self, name, text_browser):
        super().__init__()
        self.data_path = None
        self.name = name
        self.textbrowser = text_browser

    def run(self):
        # 在这里执行耗时操作（例如调用main_lr函数）
        try:
            if self.name == 'A_1':
                sys.stdout = self  # 将标准输出流重定向，定向为子线程实例，然后传递到write方法中
                main_lr(self.data_path, ratio=0.128, test=0.3)

                sys.stdout = sys.__stdout__

            if self.name == 'A_2':
                sys.stdout = self
                main_pso_ann(self.data_path, ratio=0.128)
                sys.stdout = sys.__stdout__
            if self.name == 'A_5':
                sys.stdout = self
                train_model_fnn(model_save_path = 'D:/Python projects/al_show/resources/FNN/model/FNN_model.h5')
                sys.stdout = sys.__stdout__
            if self.name == 'A_6':
                sys.stdout = self
                train_model_svr(model_save_path = 'D:/Python projects/al_show/resources/SVR/model/SVR_model.h5')
                sys.stdout = sys.__stdout__
            if self.name == 'A_7':
                sys.stdout = self
                train_model_lstm(model_save_path = 'D:/Python projects/al_show/resources/LSTM/model/LSTM_model.h5')
                sys.stdout = sys.__stdout__
            if self.name == 'A_8':
                sys.stdout = self
                train_model_stft()
                sys.stdout = sys.__stdout__
            if self.name == 'A_9':
                sys.stdout = self
                train_model_cwt()
                sys.stdout = sys.__stdout__
            if self.name == 'A_10':
                sys.stdout = self
                main_parima(self.data_path)
                sys.stdout = sys.__stdout__

            self.textbrowser.append('训练完成!模型已保存！\n')
            self.textbrowser.append('******************************************************\n')

        except ValueError:
            self.textbrowser.append('文件选择错误！请选择正确的文件夹！')

    # 处理标准输出，发送输出信息信号
    def write(self, output):

        self.output_ready.emit(output)

    def flush(self):
        pass


class WorkerThread_P(QThread):
    output_ready = pyqtSignal(str)  # 用于传递输出信息的信号

    def __init__(self, name):
        super().__init__()
        self.data_path = None
        self.name = name

    def run(self):
        try:
            # 在这里执行耗时操作（例如调用main_lr函数）
            if self.name == 'A_1':
                sys.stdout = self  # 将标准输出流重定向，定向为子线程实例，然后传递到write方法中
                LR_pre(self.data_path)
                sys.stdout = sys.__stdout__
            if self.name == 'A_2':
                sys.stdout = self
                ANN_pre(self.data_path)
                sys.stdout = sys.__stdout__
            if self.name == 'A_5':
                sys.stdout = self
                predict_RUL_model_fnn(model_save_path='D:/Python projects/al_show/resources/FNN/model/FNN_model.h5')
                sys.stdout = sys.__stdout__
            if self.name == 'A_6':
                sys.stdout = self
                predict_RUL_model_svr(model_save_path = 'D:/Python projects/al_show/resources/SVR/model/SVR_model.h5')
                sys.stdout = sys.__stdout__
            if self.name == 'A_7':
                sys.stdout = self
                predict_RUL_model_fnn(model_save_path = 'D:/Python projects/al_show/resources/LSTM/model/LSTM_model.h5')
                sys.stdout = sys.__stdout__
            if self.name == 'A_8':
                sys.stdout = self
                confusion_test_stft()
                sys.stdout = sys.__stdout__
            if self.name == 'A_9':
                sys.stdout = self
                confusion_test_cwt()
                sys.stdout = sys.__stdout__
            if self.name == 'A_10':
                sys.stdout = self
                ARIMA_pre(path = r'D:/Python projects/al_show/resources/datasets/train/Bearing2_3',Total_fold_number=1955)
                sys.stdout = sys.__stdout__

            self.textbrowser.append('预测完成！\n')
            self.textbrowser.append('******************************************************\n')
        except ValueError:
            self.textbrowser.append('文件选择错误！请选择正确的文件夹！')

    # 处理标准输出，发送输出信息信号
    def write(self, output):

        self.output_ready.emit(output)

    def flush(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    myWin = Main_window()
    myWin.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
