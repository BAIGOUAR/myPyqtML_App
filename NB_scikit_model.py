
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

def train_nb_model(algorithm_parameters, x_train, y_train):
    print("Training {} Naive Bayes model ...".format(algorithm_parameters['nb_type']))

    ml_model  = 0
    if algorithm_parameters['nb_type'] == "Gaussian":
        ml_model = GaussianNB()
    elif  algorithm_parameters['nb_type'] == "Bernoulli":
        ml_model = BernoulliNB()
    else:
        if algorithm_parameters['nb_type'] == "Gaussian":
            ml_model = GaussianNB()

    ml_model.fit(x_train, y_train)
    print("Training {} Naive Bayes model finished!!".format(algorithm_parameters['nb_type']))
    return ml_model

def get_nb_scikit_model_params(ui, is_regression):
    if is_regression:
        return []

    else:

        nb_type     = str(ui.clas_naive_bayes_type_comboBox.currentText())
        algorithm_parameters = {'nb_type': nb_type}
        return algorithm_parameters
