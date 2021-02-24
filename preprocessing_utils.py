from  ml_models import machineLearningModel

class PreprocessDataset(machineLearningModel):

    def remove_outliers(self, cut_off):
        # Remove Outliers by removing rows that are not within cut_off standard deviations from mean
        numeric_columns = self.pre_processed_dataset.select_dtypes(include=['float64', 'int']).columns.to_list()
        z_score = stats.zscore(self.pre_processed_dataset[numeric_columns])
        self.pre_processed_dataset = self.pre_processed_dataset[(np.abs(z_score) < cut_off).all(axis=1)]
        self.update_datasets_info()

    def scale_numeric_values(self):

        dataset = self.pre_processed_dataset
        self.input_scaler = MinMaxScaler(feature_range=(-1, 1))
        standardised_numeric_input = self.input_scaler.fit_transform(dataset[self.pre_processed_numeric_variables])
        # Updating the scaled values in the pre_processed_dataset
        dataset[self.pre_processed_numeric_variables] = standardised_numeric_input
        self.update_datasets_info()

    def remove_duplicate_rows(self):
        self.pre_processed_dataset.drop_duplicates(inplace=True)
        self.update_datasets_info()

    def remove_constant_variables(self):
        dataset = self.pre_processed_dataset
        self.pre_processed_dataset = dataset.loc[:, (dataset != dataset.iloc[0]).any()]
        self.update_datasets_info()

    def replace_values(self, target_variable, new_value, old_values):

        variable_data_type = self.pre_processed_column_types_pd_series[target_variable]

        if self.pre_processed_column_types_pd_series[target_variable].kind == 'f':
            value_to_replace = float(old_values)
            new_value = float(new_value)  # if '.' in new_value or 'e' in new_value.lower() else int(new_value)
        elif self.pre_processed_column_types_pd_series[target_variable].kind == 'i':
            value_to_replace = int(old_values)
            new_value = int(new_value)
        else:
            value_to_replace = old_values
        # Making sure the value to be replaced mataches with the dtype of the dataset
        value_to_replace = pd.Series(value_to_replace).astype(variable_data_type).values[0]
        # Converting to either float or int, depending if . or e is in the string
        self.pre_processed_dataset[target_variable].replace(to_replace=value_to_replace, value=new_value, inplace=True)

        self.update_datasets_info()

    def filter_out_values(self, filtering_variable, filtering_value, filtering_operator):

        column_of_filtering_variable = self.pre_processed_dataset[filtering_variable]
        dataset = self.pre_processed_dataset
        if self.pre_processed_column_types_pd_series[filtering_variable].kind == 'f':
            filtering_value = float(filtering_value)
        if self.pre_processed_column_types_pd_series[filtering_variable].kind == 'i':
            filtering_value = int(filtering_value)
        if filtering_operator == 'Equal to':
            self.pre_processed_dataset = dataset[~operator.eq(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Not equal to':
            self.pre_processed_dataset = dataset[~operator.ne(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Less than':
            self.pre_processed_dataset = dataset[~operator.lt(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Less than or equal to':
            self.pre_processed_dataset = dataset[~operator.le(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Greater than':
            self.pre_processed_dataset = dataset[~operator.gt(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Greater than or equal to':
            self.pre_processed_dataset = dataset[~operator.ge(column_of_filtering_variable, filtering_value)]

        self.update_datasets_info()

    def add_num_scaling_rule(self):
        ui = self.ui

        if self.ml_model.pre_processed_numeric_variables == []:
            display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',  # Display error message and return
                            'There are no numeric variables available', 'Error')
            # Todo Maybe disable the button instead of warning that is not possible
            return

        rule_text = 'Apply Numeric Scaling'
        rule_data = {'pre_processing_action': 'apply_num_scaling'}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_rm_outliers_rule(self):
        ui = self.ui

        cut_off = ui.outliers_treshold_horizontalSlider.value() / 10
        text = 'Remove Outliers, Cut-off = {}σ'.format(cut_off)
        data = {'pre_processing_action': 'rm_outliers', 'cut_off': cut_off}
        item_to_add = self.create_listwidgetitem(text, data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_numeric_filtering_rule(self):
        ui = self.ui
        filtering_value = ui.filtering_dataset_value_lineEdit.text()
        filtering_variable = ui.filter_columnSelection_comboBox.currentText()
        filtering_operator = ui.filter_operator_comboBox.currentText()
        if filtering_value != '':  # If input is not empty
            try:
                float(filtering_value)  # Check whether it is a valid numeric input
            except:
                display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',
                                'Type a valid numeric input for column {}'.format(filtering_operator), 'Error')
                return
            rule_text = 'Exclude values from {} {} {}'.format(filtering_variable, filtering_operator, filtering_value)
            rule_data = {'pre_processing_action': 'apply_filtering', 'variable': filtering_variable, 'is_numeric': True,
                         'filtering_operator': filtering_operator, 'filtering_value': filtering_value}
            item_to_add = self.create_listwidgetitem(rule_text, rule_data)
            self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    def add_categorical_filtering_rule(self):
        ui = self.ui
        filtering_value = ui.filtering_dataset_value_comboBox.currentText()
        filtering_variable = ui.filter_columnSelection_comboBox.currentText()
        filtering_operator = ui.filter_operator_comboBox.currentText()
        rule_text = 'Exclude values from {} {} {}'.format(filtering_variable, filtering_operator, filtering_value)
        rule_data = {'pre_processing_action': 'apply_filtering', 'variable': filtering_variable, 'is_numeric': False,
                     'filtering_operator': filtering_operator, 'filtering_value': filtering_value}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def generate_filtering_rule(self):
        ui = self.ui
        ml_model = self.ml_model

        current_variable_name = ui.filter_columnSelection_comboBox.currentText()
        variable_type = ml_model.column_types_pd_series[current_variable_name].kind
        is_numeric_variable = variable_type in 'iuf'  # iuf = i int (signed), u unsigned int, f float

        if is_numeric_variable:  # If numeric
            self.add_numeric_filtering_rule()

        else:  # If not numeric
            self.add_categorical_filtering_rule()

    def add_numeric_replacing_rule(self):
        ui = self.ui
        replacing_variable = ui.replace_columnSelection_comboBox.currentText()
        old_values = ui.replaced_value_lineEdit.text()
        new_values = ui.replacing_value_lineEdit.text()

        if old_values != '' and new_values != '':  # If inputs are not empty
            try:
                float(new_values), float(old_values)  # Check whether it is a valid input
            except:
                display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',  # Display error message and return
                                'Type a valid numeric input for column {}'.format(replacing_variable), 'Error')
                return
            rule_text = 'Replace {} in {} with {}'.format(old_values, replacing_variable, new_values)
            rule_data = {'pre_processing_action': 'replace_values', 'variable': replacing_variable, 'is_numeric': True,
                         'old_values': old_values, 'new_values': new_values}
            item_to_add = self.create_listwidgetitem(rule_text, rule_data)
            self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    def add_categorical_replacing_rule(self):
        ui = self.ui
        replacing_variable = ui.replace_columnSelection_comboBox.currentText()
        old_values = ui.replaced_value_comboBox.currentText()
        new_values = ui.replacing_value_lineEdit.text()

        if new_values != '':  # If inputs are not empty
            rule_text = 'Replace {} in {} with {}'.format(old_values, replacing_variable, new_values)
            rule_data = {'pre_processing_action': 'replace_values', 'variable': replacing_variable, 'is_numeric': False,
                         'old_values': old_values, 'new_values': new_values}
            item_to_add = self.create_listwidgetitem(rule_text, rule_data)
            self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    def generate_replacing_rule(self):
        ui = self.ui

        is_numeric_variable = ui.pre_process_replacing_stackedWidget.currentIndex() == 0

        if is_numeric_variable:  # If numeric
            self.add_numeric_replacing_rule()

        else:  # If not numeric
            self.add_categorical_replacing_rule()

    def add_rm_duplicate_rows_rule(self):
        ui = self.ui

        rule_text = 'Remove Duplicate Rows'
        rule_data = {'pre_processing_action': 'rm_duplicate_rows'}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_rm_constant_var_rule(self):
        ui = self.ui

        rule_text = 'Remove Constant Variables (Columns)'
        rule_data = {'pre_processing_action': 'rm_constant_var'}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_pre_processing_rule_to_listWidget(self, item, listWidget):

        listWidget.addItem(item)
        self.trigger_update_pre_process_thread()

    def update_preprocess_replace_fields(self):
        ui = self.ui
        ml_model = self.ml_model
        selected_value = ui.replace_columnSelection_comboBox.currentText()

        is_numeric_variable = ml_model.column_types_pd_series[
                                  selected_value].kind in 'iuf'  # iuf = i int (signed), u unsigned int, f float

        if is_numeric_variable:
            ui.pre_process_replacing_stackedWidget.setCurrentIndex(0)
        else:
            ui.pre_process_replacing_stackedWidget.setCurrentIndex(1)

            ui.replaced_value_comboBox.clear()
            unique_values = ml_model.dataset[selected_value].unique().tolist()

            # Filling the comboBoxes
            for each_value in unique_values:
                ui.replaced_value_comboBox.addItem(each_value)  # Fill comboBox

    def update_preprocess_filtering_fields(self):
        ui = self.ui
        ml_model = self.ml_model

        selected_value = ui.filter_columnSelection_comboBox.currentText()
        is_numeric_variable = ml_model.column_types_pd_series[
                                  selected_value].kind in 'iuf'  # iuf = i int (signed), u unsigned int, f float
        if is_numeric_variable:
            ui.pre_process_filtering_stackedWidget.setCurrentIndex(0)
            if ui.filter_operator_comboBox.count() == 2:  # 2 items mean only == and !=
                ui.filter_operator_comboBox.insertItem(2, 'Greater than or equal to')  # The index is always 2
                ui.filter_operator_comboBox.insertItem(2, 'Greater than')  # The list will keep shifting
                ui.filter_operator_comboBox.insertItem(2, 'Less than or equal to')
                ui.filter_operator_comboBox.insertItem(2, 'Less than')
        else:
            ui.pre_process_filtering_stackedWidget.setCurrentIndex(1)
            ui.filtering_dataset_value_comboBox.clear()
            unique_values = ml_model.dataset[selected_value].unique().tolist()
            if ui.filter_operator_comboBox.count() == 6:
                ui.filter_operator_comboBox.removeItem(2)  # Removing Less than
                ui.filter_operator_comboBox.removeItem(2)  # Removing Less than or equal to
                ui.filter_operator_comboBox.removeItem(2)  # Removing Greater than
                ui.filter_operator_comboBox.removeItem(2)  # Removing Greater than or equal to
            # Filling the comboBoxes
            for each_value in unique_values:
                ui.filtering_dataset_value_comboBox.addItem(each_value)  # Fill comboBox


