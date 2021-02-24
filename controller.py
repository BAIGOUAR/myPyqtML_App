import os
import random
import sys
import threads
import seaborn as sns
from os.path import join, abspath
from main_gui import QtCore, QtWidgets
import personalised_widgets

# Set seaborn aesthetic parameters
sns.set()

from signals import*
from model_selection import*
from plot_visualization import*
from train import*
from load_dataset import*
from utils import*

#main UI class
class ViewController(signalConnections, modelSelection, matplotlibPlot, datasetLoader, datasetTrainer):

    def __init__(self, ui, ml_model):
        self.ui = ui
        self.ml_model = ml_model

        self.root_directory = get_project_root_directory()
        self.src_directory = get_project_root_directory()
        self.data_directory = self.root_directory+'data/'

        self.configure_gui()

    def configure_gui(self):

        ui = self.ui
        _translate = QtCore.QCoreApplication.translate

        # Seting up the thread pool for multi-threading
        ui.threadpool = QtCore.QThreadPool()
        # Populating the example_dataset_comboBox
        ui.example_dataset_comboBox.addItem('', '')
        list_of_datasets = os.listdir(transform_to_resource_path(self.data_directory))
        for dataset in list_of_datasets:
            if not dataset.startswith('.'):
                # Each Item receives the dataset name as text and the dataset path as data
                ui.example_dataset_comboBox.addItem(dataset.split('.')[0], self.data_directory + dataset)

        self.connect_signals()

        #Buttons
        ui.nn_classification_radioButton.click()
        ui.nn_regression_radioButton.click()
        ui.nn_regression_radioButton.click()

        ui.tabs_widget.setCurrentIndex(0)
        ui.pre_process_tabWidget.setCurrentIndex(0)
        ui.output_selection_stackedWidget.setCurrentIndex(0)

        # Disable these widgets while no dataset is loaded
        self.disable_widgets()

        #Creates the spinner for the model_train_widget in run-time
        ui.spinner_traning_results = personalised_widgets.QtWaitingSpinner(ui.model_train_widget)
        ui.spinner_traning_results.setSizePolicy(ui.model_train_widget.sizePolicy())


    def disable_widgets(self):
        ui = self.ui
        # Disable these widgets while no dataset is loaded
        widgets_to_disable = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton,
                              ui.remove_duplicates_pushButton, ui.remove_constant_variables_pushButton,
                              ui.numeric_scaling_pushButton, ui.remove_outliers_pushButton,
                              ui.addrule_filter_value_pushButton, ui.addrule_replace_value_pushButton,
                              ui.addrule_filter_value_pushButton,
                              ui.add_input_columns_pushButton, ui.add_output_columns_pushButton,
                              ui.train_model_pushButton, ui.remove_preprocessing_rule_pushButton,
                              ui.clear_preprocessing_rule_pushButton]

        for widget in widgets_to_disable:
            widget.setEnabled(False)

    def trigger_update_pre_process_thread(self):

        ui = self.ui
        ml_model = self.ml_model

        ui.pre_process_dataset_tableWidget.spinner.start()

        worker = threads.Pre_Process_Dataset_Thread(ui, ml_model)

        worker.signals.update_pre_process_tableWidget.connect(self.generate_qt_items_to_fill_tablewidget)
        worker.signals.display_message.connect(display_message)

        ui.threadpool.start(worker)

    def clear_listwidget(self, target_listwidget):
        ui = self.ui
        ml_model = self.ml_model
        is_regression = ui.regression_selection_radioButton.isChecked()

        if target_listwidget == ui.preprocess_sequence_listWidget:
            target_listwidget.clear()
            self.trigger_update_pre_process_thread()

        elif target_listwidget == ui.input_columns_listWidget:

            for _ in range(target_listwidget.count()):
                item = target_listwidget.takeItem(0)
                ui.available_columns_listWidget.addItem(item)

                # Adding the variables back to the clas_output_colum_comboBox
                if item.text() in ml_model.categorical_variables:
                    ui.clas_output_colum_comboBox.addItem(item.text())

            ui.train_model_pushButton.setDisabled(True)
            self.update_train_test_shape_label()

        elif target_listwidget == ui.output_columns_listWidget:

            if is_regression:
                ui.train_model_pushButton.setDisabled(True)

            for _ in range(target_listwidget.count()):
                item = target_listwidget.takeItem(0)
                ui.available_columns_listWidget.addItem(item)

    def update_train_model_button_status(self, is_regression):
        ui = self.ui
        if is_regression:
            if ui.output_columns_listWidget.count() > 0 and ui.input_columns_listWidget.count() > 0:
                ui.train_model_pushButton.setDisabled(False)
            else:
                ui.train_model_pushButton.setDisabled(True)
        else:
            if ui.input_columns_listWidget.count() > 0 and ui.clas_output_colum_comboBox.count() > 0:
                ui.train_model_pushButton.setDisabled(False)
            else:
                ui.train_model_pushButton.setDisabled(True)

    def create_listwidgetitem(self, text, data):
        string_to_add = text
        my_qlist_item = QtWidgets.QListWidgetItem()  # Create a QListWidgetItem
        my_qlist_item.setText(string_to_add)  # Add the text to be displayed in the listWidget
        my_qlist_item.setData(QtCore.Qt.UserRole, data)  # Set data to the item
        return my_qlist_item


    def check_neurons_number(self,widget):
        row = widget.currentRow()
        column = widget.currentColumn()

        widget.blockSignals(True)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        try:
            integer_value = int(widget.item(row,column).text())
            if integer_value <= 0:
                display_message(QtWidgets.QMessageBox.Information, 'Invalid Input',
                                'The number of neurons must be an integer greater than 0', 'Error')
                item.setText('1')
                widget.setItem(row, column,item)
        except:
            display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',
                            'The number of neurons must be an integer greater than 0', 'Error')
            item.setText('1')
            widget.setItem(row, column, item)
        widget.blockSignals(False)

    def update_table_widget(self, table_widget, function, data):
        ui = self.ui

        if function == 'update_header':
            table_widget.setHorizontalHeaderLabels(data['header_labels'])
            header = table_widget.horizontalHeader()
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        elif function == 'fill_table':
            i = data['i']
            j = data['j']
            qt_item = data['qt_item']
            table_widget.setItem(i, j, qt_item)

        elif function == 'stop_spinner':
            table_widget.spinner.stop()
            if table_widget.objectName() == 'dataset_tableWidget':
                ui.load_file_pushButton.setDisabled(False)
                ui.example_dataset_comboBox.setDisabled(False)
            elif table_widget.objectName() == 'pre_process_dataset_tableWidget':
                self.update_train_test_shape_label()

    def update_nn_layers_table(self, table, value):
        # This blockSignals(True) prevents check_neurons_number from running here
        table.blockSignals(True)
        if value > table.rowCount():
            while value > table.rowCount():
                table.insertRow(table.rowCount())
                item = QtWidgets.QTableWidgetItem(str(10))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(table.rowCount() - 1, 0, item)
                item = QtWidgets.QTableWidgetItem('Hidden Layer ' + str(table.rowCount()))
                table.setVerticalHeaderItem(table.rowCount() - 1, item)
        else:
            while value < table.rowCount():
                table.removeRow(table.rowCount() - 1)
        table.blockSignals(False)

    def update_svm_model_parameters(self, action, is_regression):

        ui = self.ui

        if is_regression:
            combobox = ui.reg_svm_kernel_comboBox
            spin_box = ui.reg_svm_kernel_degree_spinBox
            c_slider = ui.reg_svm_C_horizontalSlider
            c_label = ui.reg_svm_C_label
            epsilon_slider = ui.reg_svm_episilon_horizontalSlider
            epsilon_label = ui.reg_svm_episilon_labelu
            max_iter_slider = ui.reg_svm_maxiter_horizontalSlider
            max_iter_label = ui.reg_svm_maxiter_label
            check_box = ui.reg_svm_maxiter_nolimit_checkBox
        else:
            combobox = ui.clas_svm_kernel_comboBox
            spin_box = ui.clas_svm_kernel_degree_spinBox
            c_slider = ui.clas_svm_C_horizontalSlider
            c_label = ui.clas_svm_C_label
            max_iter_slider = ui.clas_svm_maxiter_horizontalSlider
            max_iter_label = ui.clas_svm_maxiter_label
            check_box = ui.clas_svm_maxiter_nolimit_checkBox

        if action == 'kernel_change':
            if combobox.currentText() == 'poly':
                spin_box.setEnabled(True)
            else:
                spin_box.setEnabled(False)
        elif action == 'regularisation_change':
            self.update_label_from_slider_change(c_slider.value(),c_label)
        elif action == 'epsilon_change':
            self.update_label_from_slider_change(epsilon_slider.value(),epsilon_label)
        elif action == 'max_iter_change':
            self.update_label_from_slider_change(max_iter_slider.value(),max_iter_label)
        elif action == 'no_limit_click':
            if check_box.isChecked():
                max_iter_slider.setEnabled(False)
            else:
                max_iter_slider.setEnabled(True)

    def update_input_output_columns(self, target_object):
        ui = self.ui
        ml_model = self.ml_model
        is_regression = ui.regression_selection_radioButton.isChecked()

        if target_object == 'clear_output_variables':
            if ui.tabs_widget.currentIndex() == 4 and not is_regression:
                ui.clear_output_columns_pushButton.click()
            return

        for selected_item in ui.available_columns_listWidget.selectedItems():
            item = ui.available_columns_listWidget.takeItem(ui.available_columns_listWidget.row(selected_item))
            is_variable_categorical = selected_item.text() in ml_model.categorical_variables
            is_output_variable = target_object.objectName() == 'output_columns_listWidget'
            if is_regression and is_variable_categorical and is_output_variable:
                ui.available_columns_listWidget.addItem(item)
                display_message(QtWidgets.QMessageBox.Information, 'Invalid Input',
                                'Categorical variables should not be used as regression output', 'Error')
            else:
                target_object.addItem(item)
                combobox = ui.clas_output_colum_comboBox
                items_in_combobox = [combobox.itemText(i) for i in range(combobox.count())]
                if selected_item.text() in items_in_combobox:
                    item_index = items_in_combobox.index(selected_item.text())
                    combobox.removeItem(item_index)

        if target_object.objectName() == 'input_columns_listWidget':
            self.update_train_test_shape_label()

        self.update_train_model_button_status(is_regression)

    def update_label_from_slider_change(self, slider_value, label_object):
        ui = self.ui
        ml_model = self.ml_model

        label_object.setText('{}'.format(slider_value))

        if label_object.objectName() == 'reg_nn_layers_label':
            self.update_nn_layers_table(ui.reg_nn_layers_tableWidget, slider_value)
        elif label_object.objectName() == 'clas_nn_layers_label':
            self.update_nn_layers_table(ui.clas_nn_layers_tableWidget, slider_value)
        elif label_object.objectName() == 'outliers_treshold_label':
            label_object.setText('{:.1f}'.format(slider_value / 10))
        elif label_object.objectName() == 'reg_nn_val_percent_label':
            label_object.setText('{}%'.format(slider_value))
        elif label_object.objectName() == 'reg_nn_alpha_label':
            label_object.setText('{}'.format(slider_value / 10000))
        elif label_object.objectName() == 'clas_nn_val_percent_label':
            label_object.setText('{}%'.format(slider_value))
        elif label_object.objectName() == 'clas_nn_alpha_label':
            label_object.setText('{}'.format(slider_value / 10000))
        elif label_object.objectName() == 'train_percentage_label':
            label_object.setText('{}%'.format(slider_value))
            ui.test_percentage_horizontalSlider.setValue(100 - slider_value)
            if ml_model.is_data_loaded:
                self.update_train_test_shape_label()
        elif label_object.objectName() == 'test_percentage_label':
            label_object.setText('{}%'.format(slider_value))
            ui.train_percentage_horizontalSlider.setValue(100 - slider_value)
            if ml_model.is_data_loaded:
                self.update_train_test_shape_label()
        elif label_object.objectName() == 'clas_svm_C_label':
            label_object.setText('{:.1f}'.format(slider_value / 10))

    def remove_item_from_listwidget(self, target_listwidget):
        ui = self.ui
        ml_model = self.ml_model

        for item in target_listwidget.selectedItems():
            taken_item = target_listwidget.takeItem(target_listwidget.row(item))

            if target_listwidget == ui.input_columns_listWidget or target_listwidget == ui.output_columns_listWidget:
                    ui.available_columns_listWidget.addItem(taken_item)

                    # Adding the variables back to the clas_output_colum_comboBox
                    if item.text() in ml_model.categorical_variables and target_listwidget == ui.input_columns_listWidget:
                        target_listwidget.addItem(taken_item.text())

        if target_listwidget == ui.input_columns_listWidget or target_listwidget == ui.output_columns_listWidget:
            self.update_train_model_button_status(ui.regression_selection_radioButton.isChecked())

        if target_listwidget == ui.preprocess_sequence_listWidget:
            self.trigger_update_pre_process_thread()