
from controller import*
from  preprocessing_utils import*

class signalConnections(PreprocessDataset):

    def connect_signals(self):
        ui = self.ui

        # Connecting load_file_pushButton - Dataset Load Tab
        ui.load_file_pushButton.clicked.connect(lambda: self.trigger_loading_dataset_thread(ui.load_file_pushButton))
        ui.example_dataset_comboBox.currentIndexChanged.connect(
            lambda: self.trigger_loading_dataset_thread(ui.example_dataset_comboBox))

        # Connecting columnSelection_comboBox - Visualise Tab
        ui.variable_to_plot_comboBox.currentIndexChanged.connect(lambda: self.update_visualisation_options())

        # Connecting radio_button_change - Visualise Tab
        ui.boxplot_radioButton.clicked.connect(lambda: self.update_visualisation_widgets())
        ui.plot_radioButton.clicked.connect(lambda: self.update_visualisation_widgets())
        ui.histogram_radioButton.clicked.connect(lambda: self.update_visualisation_widgets())

        ui.remove_duplicates_pushButton.clicked.connect(lambda: self.add_rm_duplicate_rows_rule())
        ui.remove_constant_variables_pushButton.clicked.connect(lambda: self.add_rm_constant_var_rule())
        ui.numeric_scaling_pushButton.clicked.connect(lambda: self.add_num_scaling_rule())
        ui.remove_outliers_pushButton.clicked.connect(lambda: self.add_rm_outliers_rule())
        ui.addrule_filter_value_pushButton.clicked.connect(lambda: self.generate_filtering_rule())
        ui.addrule_replace_value_pushButton.clicked.connect(lambda: self.generate_replacing_rule())

        neuros_table_regression = ui.reg_nn_layers_tableWidget
        neuros_table_regression.cellChanged.connect(lambda: self.check_neurons_number(neuros_table_regression))
        neuros_table_classification = ui.clas_nn_layers_tableWidget
        neuros_table_classification.cellChanged.connect(lambda: self.check_neurons_number(neuros_table_classification))

        ui.outliers_treshold_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.outliers_treshold_horizontalSlider.value(),
                                                         ui.outliers_treshold_label))

        ui.replace_columnSelection_comboBox.currentIndexChanged.connect(
            lambda: self.update_preprocess_replace_fields())  # Update the pre_process_replacing_stackedWidget according to the replace_columnSelection_comboBox
        ui.filter_columnSelection_comboBox.currentIndexChanged.connect(
            lambda: self.update_preprocess_filtering_fields())  # Update the pre_process_replacing_stackedWidget according to the replace_columnSelection_comboBox

        ui.add_input_columns_pushButton.clicked.connect(
            lambda: self.update_input_output_columns(ui.input_columns_listWidget))
        ui.add_output_columns_pushButton.clicked.connect(
            lambda: self.update_input_output_columns(ui.output_columns_listWidget))

        ui.remove_input_columns_pushButton.clicked.connect(
            lambda: self.remove_item_from_listwidget(ui.input_columns_listWidget))
        ui.remove_output_columns_pushButton.clicked.connect(
            lambda: self.remove_item_from_listwidget(ui.output_columns_listWidget))
        ui.remove_preprocessing_rule_pushButton.clicked.connect(
            lambda: self.remove_item_from_listwidget(ui.preprocess_sequence_listWidget))

        ui.clear_input_columns_pushButton.clicked.connect(lambda: self.clear_listwidget(ui.input_columns_listWidget))
        ui.clear_output_columns_pushButton.clicked.connect(lambda: self.clear_listwidget(ui.output_columns_listWidget))
        ui.clear_preprocessing_rule_pushButton.clicked.connect(
            lambda: self.clear_listwidget(ui.preprocess_sequence_listWidget))

        model_selection_radio_buttons = [ui.regression_selection_radioButton,
                                         ui.classification_selection_radioButton,
                                         ui.radioButton_naive_bayes,
                                         ui.knn_classification_radioButton,
                                         ui.nn_classification_radioButton,
                                         ui.randomforest_classification_radioButton,
                                         ui.svm_classification_radioButton,
                                         ui.svm_regression_radioButton,
                                         ui.randomforest_regression_radioButton,
                                         ui.nn_regression_radioButton,
                                         ui.gradientboosting_regression_radioButton]

        for model_option in model_selection_radio_buttons:
            model_option.clicked.connect(lambda: self.model_selection_tab_events())

        ui.reg_nn_layers_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_layers_horizontalSlider.value(),
                                                         ui.reg_nn_layers_label))
        ui.clas_nn_val_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_val_percentage_horizontalSlider.value(),
                                                         ui.reg_nn_val_percent_label))
        ui.reg_nn_max_iter_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_max_iter_horizontalSlider.value(),
                                                         ui.reg_nn_max_iter_label))
        ui.reg_nn_alpha_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_alpha_horizontalSlider.value(),
                                                         ui.reg_nn_alpha_label))
        ui.clas_nn_layers_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_layers_horizontalSlider.value(),
                                                         ui.clas_nn_layers_label))
        ui.reg_nn_val_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_val_percentage_horizontalSlider.value(),
                                                         ui.clas_nn_val_percent_label))
        ui.clas_nn_max_iter_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_max_iter_horizontalSlider.value(),
                                                         ui.clas_nn_max_iter_label))
        ui.clas_nn_alpha_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_alpha_horizontalSlider.value(),
                                                         ui.clas_nn_alpha_label))
        ui.train_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.train_percentage_horizontalSlider.value(),
                                                         ui.train_percentage_label))
        ui.test_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.test_percentage_horizontalSlider.value(),
                                                         ui.test_percentage_label))

        ui.clas_svm_kernel_comboBox.currentIndexChanged.connect(
            lambda: self.update_svm_model_parameters('kernel_change', False))
        ui.clas_svm_C_horizontalSlider.valueChanged.connect(
            lambda: self.update_svm_model_parameters('regularisation_change', False))
        ui.clas_svm_maxiter_nolimit_checkBox.clicked.connect(
            lambda: self.update_svm_model_parameters('no_limit_click', False))
        ui.clas_svm_maxiter_horizontalSlider.valueChanged.connect(
            lambda: self.update_svm_model_parameters('max_iter_change', False))
        ui.reg_svm_kernel_comboBox.currentIndexChanged.connect(
            lambda: self.update_svm_model_parameters('kernel_change', True))
        ui.reg_svm_C_horizontalSlider.valueChanged.connect(
            lambda: self.update_svm_model_parameters('regularisation_change', True))
        ui.reg_svm_episilon_horizontalSlider.valueChanged.connect(
            lambda: self.update_svm_model_parameters('epsilon_change', True))
        ui.reg_svm_maxiter_nolimit_checkBox.clicked.connect(
            lambda: self.update_svm_model_parameters('no_limit_click', True))
        ui.reg_svm_maxiter_horizontalSlider.valueChanged.connect(
            lambda: self.update_svm_model_parameters('max_iter_change', True))


        #random forest connections
        """
        #max_depth = ui.label_86.text()  #setText(_translate("MainWindow", "Max Depth"))
        min_samples_leaf = ui.label_87.text()  #setText(_translate("MainWindow", "min_samples_leaf"))
        criterion     = ui.label_88.value()  #setText(_translate("MainWindow", "Criterion"))
        nb_estimators = ui.label_89.text()  #setText(_translate("MainWindow", "Number of Estimators"))
        min_samples_split = ui.label_90.text()#.setText(_translate("MainWindow", "min_samples_split"))
        """

        ui.clas_rf_max_depth_label_86_slider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_rf_max_depth_label_86_slider.value(),
                                                         ui.label_9))

        ui.clas_rf_nb_estimators_label_89_slider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_rf_nb_estimators_label_89_slider.value(),
                                                         ui.clas_rf_nb_estimators_label_89))

        """
        ui.clas_rf_nb_estimators_label_89_slider.currentIndexChanged.connect.connect(
            lambda: self.update_label_from_slider_change(ui.clas_rf_nb_estimators_label_89_slider.value(),
                                                         ui.clas_rf_nb_estimators_label_89))
        """

        ###KNN
        ui.clas_knn_classification_n_neighbors.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_knn_classification_n_neighbors.value(),
                                                         ui.clas_knn_classification_episilon_label_6))

        ###naive Bayes


        ui.tabs_widget.currentChanged.connect(lambda: self.update_input_output_columns('clear_output_variables'))

        ui.train_model_pushButton.clicked.connect(lambda: self.trigger_train_model_thread())

