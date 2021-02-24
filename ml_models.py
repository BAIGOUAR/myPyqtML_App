
from scipy import stats
import pandas as pd
import numpy as np
import operator
import os

#importing models
from sklearn import svm                                        # SVM classifier
from sklearn.svm import SVR                                    #SVM regression
from sklearn.neural_network import MLPRegressor, MLPClassifier #neural networs
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
#scaling
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score, f1_score, precision_score, accuracy_score, confusion_matrix

#plot /visualization
import matplotlib.pyplot as plt


from svm_scikit_model import train_svm_model
from nn_scikit_model import train_nn_model
from random_forest_scikit_model import train_rf_model
from KNN_scikit_model import train_knn_model
from NB_scikit_model import  train_nb_model


class machineLearningModel():

    def __init__(self):

        #rawd ata
        self.dataset = pd.DataFrame()           #dataFrame input
        self.column_types_pd_series = []
        self.categorical_variables = []
        self.integer_variables = []
        self.numeric_variables = []

        #preprocessed dataset
        self.pre_processed_dataset = pd.DataFrame()
        self.pre_processed_column_types_pd_series = []
        self.pre_processed_categorical_variables = []
        self.pre_processed_integer_variables = []
        self.pre_processed_numeric_variables = []

        #data loaded or no
        self.is_data_loaded = False

        self.input_scaler = []

    def read_dataset(self, address):

        filename, file_extension = os.path.splitext(address)

        try:
            # Todo: Check also for .data load_csfiles, etc
            if file_extension == '.csv':
                self.dataset = pd.read_csv(address)
            elif file_extension == '.xls' or file_extension == '.xlsx':
                self.dataset = pd.read_excel(address)
            else:
                return 'invalid_file_extension'

            self.is_data_loaded = True

            #drop NaN's values
            self.dataset.dropna(inplace=True)

            #copy data for processing to preserve original data
            self.pre_processed_dataset = self.dataset.copy()
            self.update_datasets_info()
            return 'sucess'
        except:
            return 'exception_in_the_file'


    def update_datasets_info(self):

        #Raw data
        dataset = self.dataset
        dataset.reset_index(inplace=True, drop=True)
        self.column_types_pd_series = dataset.dtypes
        self.categorical_variables = dataset.select_dtypes(include=['object']).columns.to_list()        # categorical = object inputs
        self.integer_variables = dataset.select_dtypes(include=['int64']).columns.to_list()             #integer
        self.numeric_variables = dataset.select_dtypes(include=['int64', 'float64']).columns.to_list()  #numerical

        #preprocessed --> updating index
        dataset = self.pre_processed_dataset          #copy of dataset
        dataset.reset_index(inplace=True, drop=True)  #reseting the index when aftaer droping na data
        self.pre_processed_column_types_pd_series = dataset.dtypes
        self.pre_processed_categorical_variables = dataset.select_dtypes(include=['object']).columns.to_list()
        self.pre_processed_integer_variables = dataset.select_dtypes(include=['int64']).columns.to_list()
        self.pre_processed_numeric_variables = dataset.select_dtypes(include=['int64', 'float64']).columns.to_list()

    def split_data_train_test(self, model_parameters):

        # Making a copy of the pre_processed_dataset using only input/output columns
        input_dataset = self.pre_processed_dataset[
            model_parameters['input_variables'] + model_parameters['output_variables']].copy()
        input_dataset.reset_index(inplace=True)
        # Selecting the categorical variables that are in the training set
        categorical_variables_in_training = list(set(self.pre_processed_categorical_variables) & set(
            model_parameters['input_variables']))

        le = LabelEncoder()

        self.categorical_encoders = {}

        X = self.pre_processed_dataset[model_parameters['input_variables']].iloc[:]

        x_train, x_test, y_train, y_test = train_test_split(X, \
            self.pre_processed_dataset[model_parameters['output_variables']],
            test_size=model_parameters['train_percentage'], random_state=0)

        data_indexes = np.array(input_dataset.index)
        if model_parameters['shuffle_samples']:
            np.random.shuffle(data_indexes)

        # Splitting the indexes of the Dtaframe into train_indexes and test_indexes
        train_indexes = data_indexes[0:round(len(data_indexes) *model_parameters['train_percentage'] )]
        test_indexes = data_indexes[round(len(data_indexes) * model_parameters['train_percentage']):]

        train_dataset = input_dataset.loc[train_indexes]
        test_dataset = input_dataset.loc[test_indexes]

        # if the target class is an integer which was scaled between 0 and 1
        if not model_parameters['is_regression'] and\
                self.pre_processed_column_types_pd_series[ model_parameters['output_variables'][0]].kind == 'i':  # Todo add condition to check if it was scaled as well
            original_target_categories = self.dataset[model_parameters['output_variables']].values
            y_train = original_target_categories[train_indexes]
            y_test = original_target_categories[test_indexes]

        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, categorical_variables_in_training

    def train(self, model_parameters, algorithm_parameters):

        split_dataset, categorical_variables_in_training = self.split_data_train_test(model_parameters)

        x_train = split_dataset['x_train']
        x_test = split_dataset['x_test']
        y_train = split_dataset['y_train']
        y_test = split_dataset['y_test']
        x_train.reset_index()
        x_test.reset_index()

        if model_parameters['is_regression']:

            if model_parameters['algorithm'] == 'nn':
                ml_model = MLPRegressor(hidden_layer_sizes=tuple(algorithm_parameters['n_of_neurons_each_layer']),
                                        max_iter=algorithm_parameters['max_iter'],
                                        solver=algorithm_parameters['solver'],
                                        activation=algorithm_parameters['activation_func'],
                                        alpha=algorithm_parameters['alpha'],
                                        learning_rate=algorithm_parameters['learning_rate'],
                                        validation_fraction=algorithm_parameters['validation_percentage'])
                ml_model.fit(x_train, y_train)
                y_pred = ml_model.predict(x_test)
            elif model_parameters['algorithm'] == 'svm':
                max_iter_no_limit_checked = algorithm_parameters['max_iter_no_limit_checked']
                if max_iter_no_limit_checked:
                    svm_max_iter = -1
                else:
                    svm_max_iter = algorithm_parameters['max_iter']
                ml_model = SVR(kernel=algorithm_parameters['kernel'],
                               degree=algorithm_parameters['kernel_degree'],
                               C=algorithm_parameters['regularisation_parameter'],
                               shrinking=algorithm_parameters['is_shrinking_enables'],
                               epsilon=algorithm_parameters['epsilon'],
                               max_iter=svm_max_iter)
                if len(y_train.shape) > 1:
                    ml_model = MultiOutputRegressor(ml_model)
                ml_model.fit(x_train, y_train)
                y_pred = ml_model.predict(x_test)
            elif model_parameters['algorithm'] == 'random_forest':
                algorithm_parameters = []
            elif model_parameters['algorithm'] == 'grad_boosting':
                algorithm_parameters = []

            r2_score_result = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            percentage_errors = self.mean_absolute_percentage_error(y_test, y_pred)
            if len(model_parameters['output_variables']) == 1:
                data_to_plot = percentage_errors
            else:
                data_to_plot = {'labels': model_parameters['output_variables'],
                                'values': percentage_errors.mean(axis=0)}

            training_output = {'r2_score': r2_score_result, 'mse': mse, 'mae': mae, 'rmse': rmse,
                               'data_to_plot': data_to_plot}
            return training_output

        else:

            self.output_class_label_encoder = LabelEncoder()

            # fit and transform train data
            y_train = self.output_class_label_encoder.fit_transform(y_train)
            y_test = self.output_class_label_encoder.fit_transform(y_test)

            for column in categorical_variables_in_training:
                x_train[column] = self.output_class_label_encoder.fit_transform(x_train[column].values)
                x_test[column]  = self.output_class_label_encoder.fit_transform(x_test[column].values)

            encoded_y_train = y_train.ravel()
            encoded_y_test  = y_test.ravel()

            if model_parameters['algorithm'] == 'nn':
                ml_model = train_nn_model(algorithm_parameters, x_train, encoded_y_train)
                encoded_y_pred = ml_model.predict(x_test)

            elif model_parameters['algorithm'] == 'svm':
                ml_model = train_svm_model(algorithm_parameters, x_train, encoded_y_train)
                encoded_y_pred = ml_model.predict(x_test)

            elif model_parameters['algorithm'] == 'random_forest':
                ml_model = train_rf_model(algorithm_parameters, x_train, encoded_y_train)
                encoded_y_pred = ml_model.predict(x_test)

            elif model_parameters['algorithm'] == 'knn':
                ml_model = train_knn_model(algorithm_parameters, x_train, encoded_y_train)
                encoded_y_pred = ml_model.predict(x_test)
            else:
                model_parameters['algorithm'] == 'nb'
                ml_model = train_nb_model(algorithm_parameters, x_train, encoded_y_train)
                encoded_y_pred = ml_model.predict(x_test)

            number_of_classes = len(np.unique(np.concatenate((y_train, y_test))))
            if number_of_classes > 2:
                average_value = 'macro'
            else:
                average_value = 'binary'

            recall = recall_score(encoded_y_test, encoded_y_pred, average=average_value)
            f1 = f1_score(encoded_y_test, encoded_y_pred, average=average_value)
            accuracy = accuracy_score(encoded_y_test, encoded_y_pred)
            precision = precision_score(encoded_y_test, encoded_y_pred, average=average_value)

            df_conf = pd.DataFrame(confusion_matrix(encoded_y_test, encoded_y_pred))
            df_conf.set_index(self.output_class_label_encoder.inverse_transform(df_conf.index), inplace=True)
            df_conf.columns = self.output_class_label_encoder.inverse_transform(df_conf.columns)

            training_output = {'recall_score': recall, 'f1_score': f1, 'precision_score': precision,
                               'accuracy': accuracy,
                               'data_to_plot': df_conf}
            return training_output
