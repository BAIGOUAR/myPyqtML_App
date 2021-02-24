
from controller import*
from utils import*
class datasetLoader():
    # Loading the dataset
    def trigger_loading_dataset_thread(self, data_source):

        ui = self.ui
        ml_model = self.ml_model

        ui.dataset_tableWidget.spinner.start()
        ui.pre_process_dataset_tableWidget.spinner.start()

        if data_source.objectName() == 'load_file_pushButton':
            fileDlg = QtWidgets.QFileDialog()
            file_address = fileDlg.getOpenFileName()[0]

        elif data_source.objectName() == 'example_dataset_comboBox':
            selected_index = ui.example_dataset_comboBox.currentIndex()
            file_address = ui.example_dataset_comboBox.itemData(selected_index)

            # Delete empty entry in the comboBox - This just happens once
            if ui.example_dataset_comboBox.itemText(0) == '':
                ui.example_dataset_comboBox.blockSignals(True)
                ui.example_dataset_comboBox.removeItem(0)
                ui.example_dataset_comboBox.blockSignals(False)

        if file_address == '' or file_address == None:
            ui.dataset_tableWidget.spinner.stop()
            ui.pre_process_dataset_tableWidget.spinner.stop()
            return

        ui.load_file_pushButton.setDisabled(True)
        ui.example_dataset_comboBox.setDisabled(True)

        # Creating an object worker

        # Here we are laoding the dataset
        worker = threads.Load_Dataset_Thread(ui, ml_model, file_address)

        # Connecting the signals from the created worker to its functions
        worker.signals.stop_spinner.connect(self.update_table_widget)
        worker.signals.display_message.connect(display_message)
        worker.signals.update_train_test_shape_label.connect(self.update_train_test_shape_label)
        worker.signals.populate_tablewidget_with_dataframe.connect(
            self.generate_qt_items_to_fill_tablewidget)

        # Starts the thread
        ui.threadpool.start(worker)

    def generate_qt_items_to_fill_tablewidget(self, table_widget, filling_dataframe):

        table_widget.clear()

        # Fill dataset_tableWidget from the Dataset Load Tab with the head of the dataset
        number_of_rows_to_display = 50 #Todo: Give the user the option to choose this number
        table_widget.setRowCount(len(filling_dataframe.head(number_of_rows_to_display)))
        table_widget.setColumnCount(len(filling_dataframe.columns))

        # Adding the labels at the top of the Table
        data = {'header_labels': filling_dataframe.columns}
        # Updating the table_widget in the GUI Thread
        self.update_table_widget(table_widget, 'update_header', data)

        # Filling the Table with the dataset
        for i in range(table_widget.rowCount()):
            for j in range(table_widget.columnCount()):
                dataset_value = filling_dataframe.iloc[i, j]  # Get the value from the dataset
                dataset_value_converted = dataset_value if (type(dataset_value) is str) else '{:}'.format(dataset_value)
                qt_item = QtWidgets.QTableWidgetItem(dataset_value_converted)  # Creates an qt item
                qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
                # Updating the table_widget in the GUI Thread
                data = {'i': i, 'j': j, 'qt_item': qt_item}
                self.update_table_widget(table_widget, 'fill_table', data)
        # Stopping the loading spinner
        self.update_table_widget(table_widget, 'stop_spinner', data)