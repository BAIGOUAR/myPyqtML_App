

#importing models
from sklearn.neural_network import MLPRegressor, MLPClassifier #neural networs

def  get_nn_scikit_params(ui, is_regression):
        #regression
        if is_regression:
            n_of_hidden_layers = ui.reg_nn_layers_horizontalSlider.value()
            n_of_neurons_each_layer = []
            for i in range(n_of_hidden_layers):
                n_of_neurons_each_layer.append(int(ui.reg_nn_layers_tableWidget.item(i, 0).text()))
            activation_func = ui.reg_nn_actvfunc_comboBox.currentText()
            solver = ui.reg_nn_solver_comboBox.currentText()
            learning_rate = ui.reg_nn_learnrate_comboBox.currentText()
            max_iter = ui.reg_nn_max_iter_horizontalSlider.value()
            alpha = ui.reg_nn_alpha_horizontalSlider.value() / 10000
            validation_percentage = ui.reg_nn_val_percentage_horizontalSlider.value() / 100

            algorithm_parameters = {'n_of_hidden_layers': n_of_hidden_layers,
                                    'n_of_neurons_each_layer': n_of_neurons_each_layer,
                                    'activation_func': activation_func, 'solver': solver,
                                    'learning_rate': learning_rate,
                                    'max_iter': max_iter, 'alpha': alpha,
                                    'validation_percentage': validation_percentage}

        else: #        #classification
            n_of_hidden_layers = ui.clas_nn_layers_horizontalSlider.value()  # get number of layers
            n_of_neurons_each_layer = []
            for i in range(n_of_hidden_layers):
                n_of_neurons_each_layer.append(int(ui.clas_nn_layers_tableWidget.item(i, 0).text()))
            activation_func = ui.clas_nn_actvfunc_comboBox.currentText()
            solver = ui.clas_nn_solver_comboBox.currentText()
            learning_rate = ui.clas_nn_learnrate_comboBox.currentText()
            max_iter = ui.clas_nn_max_iter_horizontalSlider.value()
            alpha = ui.clas_nn_alpha_horizontalSlider.value() / 10000
            validation_percentage = ui.clas_nn_val_percentage_horizontalSlider.value() / 100

            algorithm_parameters = {'n_of_hidden_layers': n_of_hidden_layers,
                                    'n_of_neurons_each_layer': n_of_neurons_each_layer,
                                    'activation_func': activation_func, 'solver': solver,
                                    'learning_rate': learning_rate,
                                    'max_iter': max_iter, 'alpha': alpha,
                                    'validation_percentage': validation_percentage}


        return algorithm_parameters


def train_nn_model(algorithm_parameters, x_train, y_train):
    print("Training NN model ...")
    ml_model = MLPClassifier(hidden_layer_sizes=tuple(algorithm_parameters['n_of_neurons_each_layer']),
                  max_iter=algorithm_parameters['max_iter'],
                  solver=algorithm_parameters['solver'],
                  activation=algorithm_parameters['activation_func'],
                  alpha=algorithm_parameters['alpha'],
                  learning_rate=algorithm_parameters['learning_rate'],
                  validation_fraction=algorithm_parameters['validation_percentage'],
                    verbose=True)
    ml_model.fit(x_train, y_train)
    print("Training NN model finished!!")
    return ml_model
