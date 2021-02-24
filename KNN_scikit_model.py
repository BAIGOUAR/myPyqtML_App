
from sklearn.neighbors import KNeighborsClassifier
def train_knn_model(algorithm_parameters, x_train, y_train):
    print("Training KNN model ...")
    ml_model = KNeighborsClassifier(
                n_neighbors= int(algorithm_parameters['n_neighbors']),
                 metric=algorithm_parameters['metric'],
                 algorithm=algorithm_parameters['algorithm'])
    ml_model.fit(x_train, y_train)
    print("Training KNN model finished!!")
    return ml_model

def get_knn_scikit_model_params(ui, is_regression):
    if is_regression:
        return []

    else:

        n_neighbors =  int(ui.clas_knn_classification_episilon_label_6.text())
        ##weights = int(ui.label_9.text())  #setText(_translate("MainWindow", "Max Depth"))
        distance_metric     = str(ui.clas_knn_classification_metric_comboBox_9.currentText())
        knn_algorithm_metric  = str(ui.clas_knn_classification_metric_comboBox.currentText())

        algorithm_parameters = {'n_neighbors': n_neighbors,
                                'metric': distance_metric,
                                'algorithm': knn_algorithm_metric}
    return algorithm_parameters
