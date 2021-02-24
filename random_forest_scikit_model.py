
from sklearn.ensemble import RandomForestClassifier

def train_rf_model(algorithm_parameters, x_train, y_train):
    print("Training Random Fores model ...")
    ml_model = RandomForestClassifier(n_estimators=algorithm_parameters['n_estimators'],
                 max_depth=algorithm_parameters['max_depth'],
                 max_features=algorithm_parameters['max_features'],
                 verbose=True)
    ml_model.fit(x_train, y_train)
    print("Training Random Forest model finished!!")
    return ml_model

def get_rf_scikit_model_params(ui, is_regression):
    if is_regression:
        return []

    else:
        # min_samples_leaf = ui.label_87.text()  #setText(_translate("MainWindow", "min_samples_leaf"))
        # min_samples_split = ui.label_90.text()#.setText(_translate("MainWindow", "min_samples_split"))

        max_features =  str(ui.clas_rf_max_features_comboBox_label_85.currentText())       ##.setItemText(2, _translate("MainWindow", "log2"))
        max_depth = int(ui.rf_max_depth_label_86.text())  #setText(_translate("MainWindow", "Max Depth"))
        criterion     = ui.clas_rf_criterion_label_88.currentText()#setText(_translate("MainWindow", "Criterion"))
        n_estimators = int(ui.clas_rf_nb_estimators_label_89.text()) #setText(_translate("MainWindow", "Number of Estimators"))

        algorithm_parameters = {'max_depth': max_depth,
                                'criterion': criterion,
                                'n_estimators': n_estimators,
                                'min_samples_split': 2,
                                'min_samples_leaf': 2,
                                'max_features':max_features}
    return algorithm_parameters
