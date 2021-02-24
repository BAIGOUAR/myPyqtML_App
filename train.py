
from controller import*
import threads
from ml_models import machineLearningModel
from nn_scikit_model import get_nn_scikit_params
from svm_scikit_model import get_svm_scikit_model_params
from random_forest_scikit_model import*
from KNN_scikit_model import*
from NB_scikit_model import*


class datasetTrainer(machineLearningModel):

    def display_training_results(self, result, model_parameters):
        ui = self.ui
        ui.spinner_traning_results.stop()
        ui.train_model_pushButton.setDisabled(False)

        if model_parameters['is_regression']:
            ui.reg_mse_label.setText('{:.4f}'.format(result['mse']))
            ui.reg_rmse_label.setText('{:.4f}'.format(result['rmse']))
            ui.reg_r2_label.setText('{:.4f}'.format(result['r2_score']))
            ui.reg_mea_label.setText('{:.4f}'.format(result['mae']))
            self.trigger_plot_matplotlib_to_qt_widget_thread(target_widget=ui.model_train_widget,
                                                             content={'data': result['data_to_plot'],
                                                                      'output_variables': model_parameters[
                                                                          'output_variables'],
                                                                      'is_regression': model_parameters[
                                                                          'is_regression']})

        else:
            ui.clas_accuracy_label.setText('{:.4f}'.format(result['accuracy']))
            ui.clas_recall_label.setText('{:.4f}'.format(result['recall_score']))
            ui.clas_precision_label.setText('{:.4f}'.format(result['precision_score']))
            ui.clas_f1_score_label.setText('{:.4f}'.format(result['f1_score']))
            self.trigger_plot_matplotlib_to_qt_widget_thread(target_widget=ui.model_train_widget,
                                                             content={'data': result['data_to_plot'],
                                                                      'output_variables': model_parameters['output_variables'],
                                                                      'is_regression': model_parameters['is_regression']})

    def trigger_train_model_thread(self):
        ui = self.ui
        ml_model = self.ml_model

        train_percentage = (ui.train_percentage_horizontalSlider.value() / 100)
        test_percentage = (ui.test_percentage_horizontalSlider.value() / 100)
        shuffle_samples = ui.shuffle_samples_checkBox.isChecked()

        model_parameters = {'train_percentage': train_percentage,
                            'test_percentage': test_percentage,
                            'shuffle_samples': shuffle_samples
                            }

        is_regression = ui.regression_selection_radioButton.isChecked()

        input_variables = []
        for i in range(ui.input_columns_listWidget.count()):
            input_variables.append(ui.input_columns_listWidget.item(i).text())

        if is_regression:
            output_variables = []
            for i in range(ui.output_columns_listWidget.count()):
                output_variables.append(ui.output_columns_listWidget.item(i).text())
        else:
            output_variables = [ui.clas_output_colum_comboBox.currentText()]

        algorithm_index =  []
        algorithm = []
        if is_regression:
            algorithm_index = [ui.nn_regression_radioButton.isChecked(),
                               ui.svm_regression_radioButton.isChecked(),
                               ui.randomforest_regression_radioButton.isChecked(),
                               ui.gradientboosting_regression_radioButton.isChecked()].index(1)
            algorithm = ['nn', 'svm', 'random_forest', 'grad_boosting'][algorithm_index]
        else:
            algorithm_index = [ui.nn_classification_radioButton.isChecked(),
                               ui.svm_classification_radioButton.isChecked(),
                               ui.randomforest_classification_radioButton.isChecked(),
                               ui.radioButton_naive_bayes.isChecked(),
                               ui.knn_classification_radioButton.isChecked()].index(1)

            algorithm = ['nn', 'svm', 'random_forest', 'nb', 'knn', 'grad_boosting'][algorithm_index]


        if algorithm == 'nn':
            algorithm_parameters = get_nn_scikit_params(ui, is_regression)
        elif algorithm == 'svm':
            algorithm_parameters = get_svm_scikit_model_params(ui, is_regression)
        elif algorithm == 'random_forest':
            algorithm_parameters = get_rf_scikit_model_params(ui, is_regression)
        elif algorithm == 'knn':
            algorithm_parameters = get_knn_scikit_model_params(ui, is_regression)
        else:
            algorithm_parameters = get_nb_scikit_model_params(ui, is_regression)


        #elif algorithm == 'grad_boosting':
        #algorithm_parameters = {}

        model_parameters.update(
            {'is_regression': is_regression, 'algorithm': algorithm, 'input_variables': input_variables,
             'output_variables': output_variables})

        # Creating an object worker
        worker = threads.Train_Model_Thread(ml_model, model_parameters, algorithm_parameters, ui)

        # Connecting the signals from the created worker to its functions
        worker.signals.finished.connect(self.display_training_results)

        ui.train_model_pushButton.setDisabled(True)
        ui.spinner_traning_results.start()

        # Running the traning in a separate thread from the GUI
        ui.threadpool.start(worker)

    def update_train_test_shape_label(self):
        ui = self.ui
        ml_model = self.ml_model

        dataset_shape = ml_model.pre_processed_dataset.shape

        number_of_rows_train = round(dataset_shape[0] * ui.train_percentage_horizontalSlider.value() / 100)
        number_of_columns_train = ui.input_columns_listWidget.count()

        number_of_rows_test = round(dataset_shape[0] * ui.test_percentage_horizontalSlider.value() / 100)
        number_of_columns_test = ui.input_columns_listWidget.count()

        ui.train_dataset_shape_label.setText('{} x {}'.format(number_of_rows_train, number_of_columns_train))
        ui.test_dataset_shape_label.setText('{} x {}'.format(number_of_rows_test, number_of_columns_test))

