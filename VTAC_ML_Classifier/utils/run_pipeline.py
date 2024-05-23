import os

import numpy as np
import pandas as pd
import yaml
import logging
import joblib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from imblearn.over_sampling import SMOTE
from yellowbrick import ClassBalance, ROCAUC
from yellowbrick.model_selection import FeatureImportances, ValidationCurve

from preparation import get_data, OneHotEncoderCustom, Cleaner
import matplotlib.pyplot as plt
from yellowbrick.features import ParallelCoordinates
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, PrecisionRecallCurve, ClassPredictionError
from yellowbrick.model_selection import RFECV

log = logging.getLogger(__name__)


def predict_from_best_pipeline(X, prob_flag=False, config_file=None):
    if config_file is None:
        config_file = '../config/config.yaml'

    vtac_ml_pipe = VTACMLPipe(config_path=config_file)

    vtac_ml_pipe.load_best_model(model_path='../output/best_model')
    y = vtac_ml_pipe.predict(X, prob=prob_flag)
    return y


class VTACMLPipe:
    '''
    Work in progress

    '''

    def __init__(self, config_path):
        '''
        initialize the pipeline with either a config path or a dataframe
        '''

        # initialize attributes
        self.config = None
        self.data = None
        self.X_columns = None
        self.y_columns = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.preprocessing = Pipeline(steps=[], verbose=True)
        self.full_pipeline = Pipeline(steps=[], verbose=True)
        self.models = {}
        self.best_model = None
        self.y_predict = None
        self.y_predict_prob = None

        # Load configs from config file
        self._load_config(config_path)

        # Defining Steps of the preprocessing pipeline
        cleaner = Cleaner(variables=self.X_columns)
        scaler = StandardScaler()
        normalizer = Normalizer()
        self.steps = [
            ('cleaner', cleaner),
            # ('ohe', ohe),
            ('scaler', scaler),
            ('normalizer', normalizer)
        ]
        self._create_pipe(self.steps)

        # still need to figure out the missing values, probably ohe then imputing but need to look at distributions
        # self.load_data(data=data)
        # creates prepross pipe

        # ohe = OneHotEncodercustom(variables=["EFLAG_B1"])

    # def split_data(self, data):
    #     data = data.dropna()
    #
    #     self.X_columns = self.config['Inputs']['columns']
    #     self.y_columns = self.config['Inputs']['target_column']
    #
    #     X = data[self.X_columns]
    #     y = data[self.y_columns]
    #     self._split_data(X, y, test_size=0.2)

    def get_data(self, path):
        data = pd.read_parquet(path=path, engine='fastparquet')
        return data

    def _load_data(self, data: pd.DataFrame, columns: list, target: str, test_size: float = 0.2):
        X = data[columns]
        y = data[target]
        self._split_data(X, y, test_size)

    def _create_pipe(self, steps):
        for step in steps:
            self.preprocessing.steps.append(step)

    def train(self, model_path=None, resample=False, scoring='f1'):
        '''

        Mostly done
        Creates full pipeline by appending model and searches param grid for best pipeline

        '''

        models = self.models

        if resample:
            self.X_train, self.y_train = self.resample(self.X_train, self.y_train)

        if self.preprocessing.steps is None:
            print("No preprocessing steps")

        for name, model in models.items():
            param_grid = self.config['Models'][name]['param_grid']

            print("Model: {}".format(name))

            self.full_pipeline = Pipeline(steps=self.preprocessing.steps.copy(), verbose=True)
            self.full_pipeline.steps.append((name, model))
            log.info(self.full_pipeline.steps)

            # model fitting
            grid_search = GridSearchCV(self.full_pipeline, param_grid, scoring=scoring, verbose=2)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_

            joblib.dump(grid_search, '/Users/jeremypalmerio/Repos/VTAC_ML/VTAC_ML_Classifier/output/grid_search.pkl')

        if model_path is not None:
            self.save_best_model(model_path)
        return self.best_model

    def resample(self, X, y):

        sm = SMOTE(sampling_strategy='minority', random_state=42)
        X_, y_ = sm.fit_resample(X, y)
        return X_, y_

    def save_best_model(self, model_name='best_model', model_path=None):
        if model_path is None:
            model_path = self.config['Outputs']['model_path']
            model_path = model_path + model_name
        joblib.dump(self.best_model, model_path)
        logging.info(f'Saved model to {model_path}')

    def load_best_model(self, model_path=None):
        if model_path is None:
            model_path = self.config['Outputs']['model_path'] + 'best_model'

        self.best_model = joblib.load(model_path)
        logging.info(f'Loaded model from {model_path}')

    def predict(self, X, prob=False):
        X = X[self.X_columns]
        self.y_predict = self.best_model.predict(X)
        if prob is True:
            self.y_predict_prob = self.best_model.predict_proba(X)
            return self.y_predict_prob
        else:
            return self.y_predict

    # predict with proba
    def evaluate(self, title, plot=False, score=f1_score):
        viz_path = self.config['Outputs']['viz_path']
        output_path = os.path.join(viz_path, title)
        print(self.best_model.steps)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Folder '{output_path}' created.")
        else:
            print(f"Folder '{output_path}' already exists.")
        # INCLUDE case in titles
        # model scoring
        train_pred = self.best_model.predict(self.X_train)
        test_pred = self.best_model.predict(self.X_test)

        train_conf_matrix = confusion_matrix(self.y_train, train_pred)
        test_conf_matrix = confusion_matrix(self.y_test, test_pred)

        # Evaluate model performance
        print('*' * 50)
        print('Training score:')
        print(
            f'MAE: {round(mean_absolute_error(self.y_train, train_pred), 4)} '
            f'| RMSE: {round(mean_squared_error(self.y_train, train_pred, squared=False), 4)} '
            f'| F1: {round(f1_score(self.y_train, train_pred), 4)}'
        )
        print('Confusion Matrix:')
        print(train_conf_matrix)
        print('-' * 20)
        print('Validation score:')
        print(
            f'MAE: {round(mean_absolute_error(self.y_test, test_pred), 4)} '
            f'| RMSE: {round(mean_squared_error(self.y_test, test_pred, squared=False), 4)} '
            f'| F1: {round(f1_score(self.y_test, test_pred), 4)}'
        )
        print('Confusion Matrix:')
        print(test_conf_matrix)

        if plot:

            _, ax_report = plt.subplots()

            report_viz = ClassificationReport(self.best_model, classes=["NOT_GRB", "IS_GRB"], support=True,
                                              ax=ax_report)
            report_viz.fit(self.X_train, self.y_train)  # Fit the visualizer and the model
            report_viz.score(self.X_test, self.y_test)  # Evaluate the model on the test data
            report_viz.show(outpath=output_path + '/classification_report.pdf')

            _, ax_cm_test = plt.subplots()

            cm_test_viz = ConfusionMatrix(self.best_model, classes=["NOT_GRB", "IS_GRB"], percent=True, axes=ax_cm_test)
            cm_test_viz.fit(self.X_train, self.y_train)
            cm_test_viz.score(self.X_test, self.y_test)
            cm_test_viz.show(outpath=output_path + '/confusion_matrix_test.pdf')

            _, ax_cm_train = plt.subplots()

            cm_train_viz = ConfusionMatrix(self.best_model, classes=["NOT_GRB", "IS_GRB"], percent=True, ax=ax_cm_train)
            cm_train_viz.fit(self.X_train, self.y_train)
            cm_train_viz.score(self.X_train, self.y_train)
            cm_train_viz.show(outpath=output_path + '/confusion_matrix_train.pdf')

            _, ax_roc = plt.subplots()

            roc_viz = ROCAUC(self.best_model, classes=["NOT_GRB", "IS_GRB"], ax=ax_roc)
            roc_viz.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
            roc_viz.score(self.X_test, self.y_test)  # Evaluate the model on the test data
            roc_viz.show(outpath=output_path + '/ROC_AUC.pdf')

            _, ax_pr_curve = plt.subplots()

            pr_curve_viz = PrecisionRecallCurve(self.best_model, classes=["NOT_GRB", "IS_GRB"], ax=ax_pr_curve)
            pr_curve_viz.fit(self.X_train, self.y_train)
            pr_curve_viz.score(self.X_test, self.y_test)
            pr_curve_viz.show(outpath=output_path + '/PR_curve.pdf')

            _, ax_class_pred = plt.subplots()
            ax_class_pred.semilogy()

            class_pred_viz = ClassPredictionError(self.best_model, classes=["NOT_GRB", "IS_GRB"], ax=ax_class_pred)
            class_pred_viz.fit(self.X_train, self.y_train)
            class_pred_viz.score(self.X_test, self.y_test)
            class_pred_viz.show(outpath=output_path + '/class_predictions.pdf')
            #
            if self.best_model.steps[-1][1] == RandomForestClassifier():
                _, ax_feature_imp = plt.subplots()

                feature_imp_viz = FeatureImportances(self.best_model.steps[-1][1], ax=ax_feature_imp)
                feature_imp_viz.fit(self.X, self.y)
                feature_imp_viz.show(outpath=output_path + '/feature_importances.pdf')



    def plot_knn_neighbors(self, knn, data, features, target):
        """
        Plot the KNN neighbors for a given dataset.

        Parameters:
        data (DataFrame): The dataset containing the features and target.
        features (list): List of feature names to plot.
        target (str): The target column name.
        n_neighbors (int): Number of neighbors to consider in the KNN model.
        """
        X = data[features]
        y = data[target]


        # Select a random point
        random_index = np.random.randint(0, len(X))
        random_point = X.iloc[random_index]

        # Find the neighbors of the random point
        neighbors = knn.kneighbors([random_point], return_distance=False)

        # Plot each pair of features
        num_features = len(features)
        for i in range(num_features):
            for j in range(i + 1, num_features):
                plt.figure(figsize=(8, 6))
                plt.scatter(X.iloc[:, i], X.iloc[:, j], c=y, cmap='viridis', marker='o', edgecolor='k', s=50)
                plt.scatter(random_point[i], random_point[j], c='red', marker='x', s=200, label='Random Point')
                plt.scatter(X.iloc[neighbors[0], i], X.iloc[neighbors[0], j], c='red', marker='o', edgecolor='k', s=100,
                            facecolors='none', label='Neighbors')
                plt.xlabel(features[i])
                plt.ylabel(features[j])
                plt.title(f'KNN Neighbors with {features[i]} vs {features[j]}')
                plt.legend()
                plt.savefig()

    # Example usage:
    # Assuming 'data' is a DataFrame containing your dataset
    # and 'target' is the name of the target column
    features = ['feature1', 'feature2', 'feature3']  # Replace with your actual feature names
    target = 'target'  # Replace with your actual target column name
    plot_knn_neighbors(data, features, target)

    def hyperparameter_valid_curve(self):
        best_model_name = self.best_model.steps[-1][0]
        param_grid = self.config['Models'][best_model_name]['param_grid']
        for param_name in param_grid:
            param_range = param_grid[param_name]

            _, ax_valid_curve = plt.subplots()

            valid_curve_viz = ValidationCurve(self.best_model,
                                              param_name=param_name,
                                              param_range=param_range,
                                              cv=5,
                                              scoring="f1",
                                              ax=ax_valid_curve
                                              )
            valid_curve_viz.fit(self.X, self.y)
            valid_curve_viz.show(outpath=f'../output/visualizations/{param_name}_valid_curve.pdf')

    def recursive_feature_elimination_plot(self):

        _, ax_feature_elimination = plt.subplots()
        visualizer = RFECV(self.best_model.steps[-1][1], cv=5, scoring='f1_weighted', ax=ax_feature_elimination)
        visualizer.fit(self.X, self.y)  # Fit the data to the visualizer
        visualizer.show(outpath='../output/visualizations/feature_elimination.pdf')

    def _split_data(self, X, y, test_size):
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = train_test_split(X, y,
                                         test_size=test_size,
                                         random_state=123)
        _, ax_class_balance = plt.subplots()
        ax_class_balance.semilogy()
        class_balance_visualizer = ClassBalance(labels=["NOT_GRB", "GRB"], ax=ax_class_balance,
                                                kwargs={'verbose': 2})
        class_balance_visualizer.fit(self.y_train, self.y_test)  # Fit the data to the visualizer
        class_balance_visualizer.show(outpath='../output/visualizations/class_balance.pdf')

    def _load_config(self, config_path):

        # loading config file and prepping data
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        data_path = self.config['Inputs']['path']
        data = get_data(path=data_path).dropna()
        self.X_columns = self.config['Inputs']['columns']
        self.X = data[self.X_columns]
        self.y_columns = self.config['Inputs']['target_column']
        self.y = data[self.y_columns]
        self._load_data(data, columns=self.X_columns, target=self.y_columns, test_size=0.2)
        for model in self.config['Models']:
            self.models[model] = eval(self.config['Models'][model]['class'])
