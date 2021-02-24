

#importing models
from sklearn import svm                                        # SVM classifier
from sklearn.svm import SVR                                    #SVM regression

def get_svm_scikit_model_params(ui, is_regression):
    if is_regression:
        kernel = ui.reg_svm_kernel_comboBox.currentText()
        kernel_degree = ui.reg_svm_kernel_degree_spinBox.value()
        regularisation_parameter = float(ui.reg_svm_C_label.text())
        is_shrinking_enables = ui.reg_svm_shirinking_checkBox.isChecked()
        epsilon = float(ui.reg_svm_episilon_label.text())
        max_iter_no_limit_checked = ui.reg_svm_maxiter_nolimit_checkBox.isChecked()
        max_iter = int(ui.reg_svm_maxiter_label.text())
        algorithm_parameters = {'kernel': kernel,
                                'kernel_degree': kernel_degree,
                                'regularisation_parameter': regularisation_parameter,
                                'is_shrinking_enables': is_shrinking_enables,
                                'epsilon': epsilon,
                                'max_iter_no_limit_checked': max_iter_no_limit_checked,
                                'max_iter': max_iter}

    else:
        kernel = ui.clas_svm_kernel_comboBox.currentText()
        kernel_degree = ui.clas_svm_kernel_degree_spinBox.value()
        regularisation_parameter = float(ui.clas_svm_C_label.text())
        is_shrinking_enables = ui.clas_svm_shirinking_checkBox.isChecked()
        max_iter_no_limit_checked = ui.clas_svm_maxiter_nolimit_checkBox.isChecked()
        max_iter = int(ui.clas_svm_maxiter_label.text())
        algorithm_parameters = {'kernel': kernel,
                                'kernel_degree': kernel_degree,
                                'regularisation_parameter': regularisation_parameter,
                                'is_shrinking_enables': is_shrinking_enables,
                                'max_iter_no_limit_checked': max_iter_no_limit_checked,
                                'max_iter': max_iter}
    return algorithm_parameters

def train_svm_model(algorithm_parameters, x_train, y_train):

    print("Training SVM model ...")

    max_iter_no_limit_checked = algorithm_parameters['max_iter_no_limit_checked']
    if max_iter_no_limit_checked:
        svm_max_iter = -1
    else:
        svm_max_iter = algorithm_parameters['max_iter']

    ml_model = svm.SVC(kernel=algorithm_parameters['kernel'],
                       degree=algorithm_parameters['kernel_degree'],
                       C=algorithm_parameters['regularisation_parameter'],
                       shrinking=algorithm_parameters['is_shrinking_enables'],
                       max_iter=svm_max_iter,
                       verbose=True)
    ml_model.fit(x_train, y_train)
    print("Training SVM model finished!!")
    return  ml_model

