import ml_models as md
from main_gui import QtCore, QtWidgets, Ui_MainWindow
from controller import ViewController
import controller
import sys

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
user_ML_model = md.machineLearningModel()
view_controller = ViewController(ui,user_ML_model)
MainWindow.show()
sys.exit(app.exec_())