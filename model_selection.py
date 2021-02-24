from controller import*

class modelSelection():

    def model_selection_tab_events(self):
        ui = self.ui
        is_regression = ui.regression_selection_radioButton.isChecked()

        if is_regression:
            ui.regression_and_classification_stackedWidget.setCurrentIndex(0)  # Change to Regression Tab
            ui.train_metrics_stackedWidget.setCurrentIndex(0)  # Change to Regression Tab
            ui.output_selection_stackedWidget.setCurrentIndex(0)
            self.update_train_model_button_status(is_regression)

            if ui.nn_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(0)

            elif ui.svm_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(1)

            elif ui.randomforest_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(2)

            elif ui.gradientboosting_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(3)

        elif ui.classification_selection_radioButton.isChecked():
            ui.regression_and_classification_stackedWidget.setCurrentIndex(1)  # Change to Classification Tab
            ui.train_metrics_stackedWidget.setCurrentIndex(1)  # Change to Regression Tab
            ui.output_selection_stackedWidget.setCurrentIndex(1)
            self.update_train_model_button_status(is_regression)

            if ui.nn_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(0)

            elif ui.svm_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(1)

            elif ui.randomforest_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(2)

            elif ui.radioButton_naive_bayes.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(3)

            elif ui.knn_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(4)
