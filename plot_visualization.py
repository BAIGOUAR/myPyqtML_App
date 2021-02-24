
from controller import*

class matplotlibPlot():
    def update_visualisation_widgets(self):
        ui = self.ui
        ml_model = self.ml_model

        selected_column = ui.variable_to_plot_comboBox.currentText()  # Get the selected value in the comboBox

        ui.columnSummary_textBrowser.clear()
        description = ml_model.dataset[selected_column].describe()
        for i in range(len(description)):
            ui.columnSummary_textBrowser.append('{} = {}'.format(description.keys()[i].title(), description.values[i]))

        is_categorical = ml_model.column_types_pd_series[
                             selected_column].kind not in 'iuf'  # iuf = i int (signed), u unsigned int, f float

        self.trigger_plot_matplotlib_to_qt_widget_thread(target_widget=ui.dataVisualisePlot_widget,
                                                         content={'data': ml_model.dataset[selected_column],
                                                                  'is_categorical': is_categorical})

    def update_visualisation_options(self):
        ui = self.ui
        ml_model = self.ml_model

        selected_column = ui.variable_to_plot_comboBox.currentText()  # Get the selected value in the comboBox
        # Create a list of all radioButton objects
        radio_buttons_list = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton]
        # Check if the selected value in the variable_to_plot_comboBox is a numeric column in the dataset

        # iuf = i int (signed), u unsigned int, f float
        if ml_model.column_types_pd_series[selected_column].kind in 'iuf':
            radio_buttons_list[0].setEnabled(True)
            radio_buttons_list[1].setEnabled(True)
            radio_buttons_list[2].setEnabled(True)

            checked_objects = list(map(lambda x: x.isChecked(), radio_buttons_list))
            if not any(checked_objects):  # Checks whether any radio button is checked
                radio_buttons_list[0].setChecked(True)  # If not, checks the first one.

        else:  # If not numeric, disable all visualisation options
            radio_buttons_list[0].setEnabled(False)
            radio_buttons_list[1].setEnabled(False)
            radio_buttons_list[2].setEnabled(False)

        self.update_visualisation_widgets()


    #matplotlib thread
    def trigger_plot_matplotlib_to_qt_widget_thread(self, target_widget, content):

        # Creating an object worker
        worker = threads.Plotting_in_MplWidget_Thread(self.ui, target_widget, content)

        # Starts the thread
        self.ui.threadpool.start(worker)
