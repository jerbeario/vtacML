""" Pipeline for optimizing, training and using a classification pipeline to flag
observation as GRBs or not for the VTAC pipeline of the SVOM mission. """

import logging
import os
import pathlib

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from imblearn.over_sampling import SMOTE

from yellowbrick import ROCAUC
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import (
    ClassificationReport,
    ConfusionMatrix,
    PrecisionRecallCurve,
    ClassPredictionError,
)


from .preparation import Cleaner
from .utils import get_path

from .utils import ROOTDIR

# use ROOTDIR/

log = logging.getLogger(__name__)


def predict_from_best_pipeline(
    X: pd.DataFrame,
    prob_flag=False,
    model_name="0.974_rfc_best_model.pkl",
    config_path=None,
):
    """
    Predict using the best model pipeline.

    Parameters
    ----------
    X : array-like
        Features to predict.
    prob_flag : bool, optional
        Whether to return probabilities, by default False.
    model_name : str, optional
        Name of the model to use, by default '0.974_rfc_best_model.pkl'
    config_path : str, optional
        Path to the configuration file, by default '../config/config.yaml'.

    Returns
    -------
    ndarray
        Predicted values or probabilities.
    """

    vtac_ml_pipe = VTACMLPipe(config_file=config_path)
    print(model_name)
    vtac_ml_pipe.load_best_model(model_name=model_name)
    print(vtac_ml_pipe.best_model)
    y = vtac_ml_pipe.predict(X, prob=prob_flag)
    return y


class VTACMLPipe:
    """
    A machine learning pipeline for training and evaluating an optimal model for optical identification of GRBs for the
    SVOM mission.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. Default 'config/config.yaml'

    """

    def __init__(self, config_file="config/config.yaml"):
        """
        Initialize the VTACMLPipe.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        """

        # initialize attributes
        self.config = None
        self.X = None
        self.y = None
        self.X_columns = None
        self.y_columns = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.preprocessing = Pipeline(steps=[], verbose=True)
        self.full_pipeline = Pipeline(steps=[], verbose=True)
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.y_predict = None

        # Load configs from config file
        self.load_config(config_file)

        # Defining Steps of the preprocessing pipeline
        cleaner = Cleaner(variables=self.X_columns)
        scaler = StandardScaler()
        normalizer = Normalizer()
        self.steps = [
            ("cleaner", cleaner),
            # ('ohe', ohe),
            ("scaler", scaler),
            ("normalizer", normalizer),
        ]
        self._create_pipe(self.steps)

    def load_config(self, config_file):
        """
        Load the configuration file and prepare the data.

        Parameters
        ----------
        config_file : str
            The path to the configuration file.

        """
        config_path = get_path(config_file)

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # loading config file and prepping data
        data_file = self.config["Inputs"]["file"]
        data = self._get_data(data_file=data_file)

        self.X_columns = self.config["Inputs"]["columns"]
        self.X = data[self.X_columns]
        self.y_columns = self.config["Inputs"]["target_column"]
        self.y = data[self.y_columns]
        self._load_data(
            data, columns=self.X_columns, target=self.y_columns, test_size=0.2
        )

        # building models attribute
        for model in self.config["Models"]:
            if model == "rfc":
                self.models[model] = RandomForestClassifier()
            if model == "svc":
                self.models[model] = SVC()
            if model == "knn":
                self.models[model] = KNeighborsClassifier()
            if model == "lr":
                self.models[model] = LogisticRegression()
            if model == "dt":
                self.models[model] = DecisionTreeClassifier()
            if model == "ada":
                self.models[model] = AdaBoostClassifier()

    def train(
        self,
        save_all_model=False,
        save_path="outputs/models",
        resample_flag=False,
        scoring="f1",
        cv=5,
        verbose=1,
        n_jobs=None,
    ):
        """
        Train the pipeline with the given data.

        Parameters
        ----------
        save_all_model : bool, optional
            Whether to save best model of each model type to output directory. Default is False.
        save_path : str, optional
            Path to save models. Default is 'outputs/models'.
        resample_flag : bool, optional
            Whether to resample the data. Default is False
        scoring : str, optional
            The scoring function to use. Default is 'f1'.
        cv : int, optional
            The cross-validation split to use. Default is 5.
        n_jobs : int, optional
            The number of CPU cores to use. Default is None. See `sklearn.model_selection.GridSearchCV`.
        verbose : int, optional
            The verbosity of the training process. Default is 1.


        Returns
        -------
        Pipeline
            Trained machine learning pipeline.
        """

        models = self.models

        if resample_flag:
            self.X_train, self.y_train = self._resample(self.X_train, self.y_train)

        if self.preprocessing.steps is None:
            print("No preprocessing steps")
        # model_path = None
        best_score = 0
        for name, model in models.items():

            param_grid = self.config["Models"][name]["param_grid"]

            log.info("Model: {}".format(name))

            self.full_pipeline = Pipeline(
                steps=self.preprocessing.steps.copy(), verbose=False
            )
            self.full_pipeline.steps.append((name, model))
            log.info(self.full_pipeline.steps)

            # model fitting
            grid_search = GridSearchCV(
                self.full_pipeline,
                param_grid,
                scoring=scoring,
                verbose=verbose,
                cv=cv,
                n_jobs=n_jobs,
                pre_dispatch="2*n_jobs",
            )
            grid_search.fit(self.X_train, self.y_train)
            # grid_search.fit(self.X_train, self.y_train)
            self.cv_results[name] = grid_search.cv_results_
            print(self.cv_results[name])

            model_filename = (
                f"{round(grid_search.best_score_, 3)}_{name}_best_model.pkl"
            )
            model_path = get_path(save_path + model_filename)
            if save_all_model:
                joblib.dump(grid_search.best_estimator_, model_path)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_

            log.info("*" * 50)
            log.info(f"Best {name} Pipeline:")
            log.info(grid_search.best_estimator_)
            log.info(f"Best Score: {grid_search.best_score_}")
            log.info("*" * 50)
            log.info(f"Overall best model: {self.best_model}")

        # self.save_best_model(model_path=model_path)

    def save_best_model(self, model_name="best_model", model_path=None):
        """
        Saves best model from training to the specified path in the config file. Optionally change name and/or path
        of the model.

        Parameters
        -------
        model_name : str, optional
            Name of the model to be saved. Default='best_model'.
        model_path : str, optional
            Path to the model to be saved. Default='model_path' in config file


        """
        if model_path is None:
            model_path = get_path(
                f"{self.config['Outputs']['model_path']}/{model_name}"
            )
        else:
            print(model_path)
            model_path = get_path(model_path)
            print(model_path)

        joblib.dump(self.best_model, model_path)
        logging.info(f"Saved model to {model_path}")

    def load_best_model(self, model_name):
        """
        Loads 'model_name' into current pipeline.

        Parameters
        -------
        model_name : str
            The name of the model from the Outputs/models/ directory to be loaded.


        """
        model_path = get_path(f"{self.config['Outputs']['model_path']}/{model_name}")
        self.best_model = joblib.load(model_path)
        logging.info(f"Loaded {model_path}")

    @staticmethod
    def display_models():
        directory = ROOTDIR / 'output/models/'
        for path in directory.glob('*'):
            print(str(path).split('/')[-1])
            for file in path.glob('*.pkl'):
                print(file)

    def evaluate(self, name, plot=False):
        """
        Evaluate the best model with various metrics and visualization.

        Parameters
        ----------
        name : str
            The name for the evaluation output.
        plot : bool, optional
            If True, generates and saves evaluation plots, by default False.
        """
        viz_path = self.config["Outputs"]["viz_path"]
        output_path = get_path(f"{viz_path}/{name}")

        print(self.best_model.steps)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Folder '{output_path}' created.")
        else:
            print(f"Folder '{output_path}' already exists.")
        # INCLUDE case in titles
        # model scoring
        #
        # if self.best_model.steps[-1][0] == 'knn' and plot:
        #     print('plotting')
        #     self.preprocessing.fit(self.X)
        #     X = pd.DataFrame(self.preprocessing.transform(self.X))
        #     self.plot_knn_neighbors(knn=self.best_model.steps[-1][1], X=X, y=self.y, features=self.X_columns)
        #     print('done plotting')
        # else:
        print(self.best_model.steps[-1][1])

        # Evaluate model performance
        self._print_model_eval()

        if plot:

            _, ax_report = plt.subplots()

            report_viz = ClassificationReport(
                self.best_model,
                classes=["NOT_GRB", "IS_GRB"],
                support=True,
                ax=ax_report,
            )
            report_viz.fit(
                self.X_train, self.y_train
            )  # Fit the visualizer and the model
            report_viz.score(
                self.X_test, self.y_test
            )  # Evaluate the model on the test data
            report_viz.show(outpath=output_path + "/classification_report.pdf")

            _, ax_cm_test = plt.subplots()

            cm_test_viz = ConfusionMatrix(
                self.best_model,
                classes=["NOT_GRB", "IS_GRB"],
                percent=True,
                axes=ax_cm_test,
            )
            cm_test_viz.fit(self.X_train, self.y_train)
            cm_test_viz.score(self.X_test, self.y_test)
            cm_test_viz.show(outpath=output_path + "/confusion_matrix_test.pdf")

            _, ax_cm_train = plt.subplots()

            cm_train_viz = ConfusionMatrix(
                self.best_model,
                classes=["NOT_GRB", "IS_GRB"],
                percent=True,
                ax=ax_cm_train,
            )
            cm_train_viz.fit(self.X_train, self.y_train)
            cm_train_viz.score(self.X_train, self.y_train)
            cm_train_viz.show(outpath=output_path + "/confusion_matrix_train.pdf")

            _, ax_roc = plt.subplots()

            roc_viz = ROCAUC(self.best_model, classes=["NOT_GRB", "IS_GRB"], ax=ax_roc)
            roc_viz.fit(
                self.X_train, self.y_train
            )  # Fit the training data to the visualizer
            roc_viz.score(
                self.X_test, self.y_test
            )  # Evaluate the model on the test data
            roc_viz.show(outpath=output_path + "/ROC_AUC.pdf")

            _, ax_pr_curve = plt.subplots()

            pr_curve_viz = PrecisionRecallCurve(
                self.best_model, classes=["NOT_GRB", "IS_GRB"], ax=ax_pr_curve
            )
            pr_curve_viz.fit(self.X_train, self.y_train)
            pr_curve_viz.score(self.X_test, self.y_test)
            pr_curve_viz.show(outpath=output_path + "/PR_curve.pdf")

            _, ax_class_pred = plt.subplots()
            ax_class_pred.semilogy()

            class_pred_viz = ClassPredictionError(
                self.best_model, classes=["NOT_GRB", "IS_GRB"], ax=ax_class_pred
            )
            class_pred_viz.fit(self.X_train, self.y_train)
            class_pred_viz.score(self.X_test, self.y_test)
            class_pred_viz.show(outpath=output_path + "/class_predictions.pdf")
            #
            if self.best_model.steps[-1][1] == RandomForestClassifier():
                _, ax_feature_imp = plt.subplots()

                feature_imp_viz = FeatureImportances(
                    self.best_model.steps[-1][1], ax=ax_feature_imp
                )
                feature_imp_viz.fit(self.X, self.y)
                feature_imp_viz.show(outpath=output_path + "/feature_importances.pdf")
        # if plot_extra:
        #     self.hyperparameter_valid_curve(outpath=output_path)
        #     # self.recursive_feature_elimination_plot(outpath=output_path)
        #

    def predict(self, X, prob=False):
        """
        Predict using the best model.

        Parameters
        ----------
        X : DataFrame
            The input features for prediction.
        prob : bool, optional
            If True, returns the probability of the predictions, by default False.

        Returns
        -------
        ndarray
            The predicted values or probabilities.
        """
        X = X[self.X_columns]
        if prob is True:
            self.y_predict = self.best_model.predict_proba(X)
        else:
            self.y_predict = self.best_model.predict(X)
        return self.y_predict

    @staticmethod
    def _get_data(data_file: str):
        """
        Load data from a parquet file.

        Parameters
        ----------
        data_file : str
            The name of the data file to load.

        Returns
        -------
        DataFrame
            The loaded data.
        """
        data_path = get_path(f"/data/{data_file}")
        print(data_path)
        data = pd.read_parquet(data_path, engine="fastparquet")
        return data

    def _load_data(
        self, data: pd.DataFrame, columns: list, target: str, test_size: float = 0.2
    ):
        """
        Load the data from the source specified in the config.

        Parameters
        ----------
        data : pd.DataFrame
            The data to load.
        columns : list
            The columns to load.
        target: str
            The target column.
        test_size: float, optional
            The size of the test sample as a fraction of the total sample. Default is 0.2.

        Returns
        -------
        DataFrame
            Loaded data.
        """
        X = data[columns]
        y = data[target]
        self._split_data(X, y, test_size)

    def _split_data(self, X, y, test_size):
        """
        Split data into training and testing sets.

        Parameters
        ----------
        X : DataFrame
            The input features.
        y : array-like
            The target values.
        test_size : float
            The proportion of the dataset to include in the test split.

        Returns
        -------
        None
        """
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(
            X, y, test_size=test_size, random_state=123
        )

        # _, ax_class_balance = plt.subplots()
        # ax_class_balance.semilogy()
        # class_balance_visualizer = ClassBalance(labels=["NOT_GRB", "GRB"], ax=ax_class_balance,
        #                                         kwargs={'verbose': 2})
        # class_balance_visualizer.fit(self.y_train, self.y_test)  # Fit the data to the visualizer
        # class_balance_visualizer.show(outpath='/output/visualizations/class_balance.pdf')

    @staticmethod
    def _resample(X, y):
        """
        Resamples the input data

        Parameters
        -------
        X : pd.DataFrame
            input data
        y : pd.Series
            input label

        Returns
        -------
        X_ : pd.DataFrame
            resampled data
        y_ : pd.Series
            resampled label
        """
        sm = SMOTE(sampling_strategy="minority", random_state=42)
        X_, y_ = sm.fit_resample(X, y)
        return X_, y_

    def _create_pipe(self, steps):
        """
        Create the machine learning pipeline from the given steps.

        Parameters
        -------
        steps : list
            The steps to use for the machine learning preprocessing pipeline.

        Returns
        -------
        Pipeline
            The created machine learning pipeline.
        """
        for step in steps:
            self.preprocessing.steps.append(step)

    def _print_model_eval(self):
        """
        Prints the evaluation of the model, mean average error (MAE), root mean squared error (RMSE),
        f1 score and confusion matrices for training and testing datasets.
        """

        train_pred = self.best_model.predict(self.X_train)
        test_pred = self.best_model.predict(self.X_test)

        train_conf_matrix = confusion_matrix(self.y_train, train_pred)
        test_conf_matrix = confusion_matrix(self.y_test, test_pred)
        print("*" * 50)
        print("Training score:")
        print(
            f"MAE: {round(mean_absolute_error(self.y_train, train_pred), 4)} "
            f"| RMSE: {round(mean_squared_error(self.y_train, train_pred, squared=False), 4)} "
            f"| F1: {round(f1_score(self.y_train, train_pred), 4)}"
        )
        print("Confusion Matrix:")
        print(train_conf_matrix)
        print("-" * 20)
        print("Validation score:")
        print(
            f"MAE: {round(mean_absolute_error(self.y_test, test_pred), 4)} "
            f"| RMSE: {round(mean_squared_error(self.y_test, test_pred, squared=False), 4)} "
            f"| F1: {round(f1_score(self.y_test, test_pred), 4)}"
        )
        print("Confusion Matrix:")
        print(test_conf_matrix)


# def hyperparameter_valid_curve(self, outpath):
#     """
#         Validate hyperparameters and generate validation curves.
#
#         Parameters
#         ----------
#         outpath : str
#             The output path where the validation curve plots will be saved.
#
#         Returns
#         -------
#         None
#         """
#     best_model_name = self.best_model.steps[-1][0]
#     param_grid = self.config['Models'][best_model_name]['param_grid']
#     for param in param_grid:
#         # self.preprocessing.fit(self.X, self.y)
#         # processed_X = self.preprocessing.transform(self.X)
#         # processed_y = self.y
#         param_range = param_grid[param]
#         param_name = param.split('__')[1]
#         print(f'Validating {param_name} over range {param_range}')
#         _, ax_valid_curve = plt.subplots()
#
#         valid_curve_viz = ValidationCurve(self.best_model,
#                                           param_name=param,
#                                           param_range=param_range,
#                                           cv=5,
#                                           scoring="f1",
#                                           ax=ax_valid_curve
#                                           )
#         valid_curve_viz.fit(self.X, self.y)
#         valid_curve_viz.show(outpath=f'{outpath}/{param_name}_valid_curve.pdf')
#
# def recursive_feature_elimination_plot(self, outpath):
#     """
#        Generate a recursive feature elimination plot.
#
#        Parameters
#        ----------
#        outpath : str
#            The output path where the feature elimination plot will be saved.
#
#        Returns
#        -------
#        None
#        """
#     _, ax_feature_elimination = plt.subplots()
#     visualizer = RFECV(self.best_model.steps[-1][1], cv=5, scoring='f1_weighted', ax=ax_feature_elimination)
#     visualizer.fit(self.X, self.y)  # Fit the data to the visualizer
#     visualizer.show(outpath=outpath + 'feature_elimination.pdf')
#
# @staticmethod
# def plot_knn_neighbors(knn, X, y, features):
#     """
#         Plot the KNN neighbors for a given dataset.
#
#         Parameters
#         ----------
#         knn : KNeighborsClassifier
#             The KNN classifier.
#         X : DataFrame
#             The dataset containing the features.
#         y : array-like
#             The target values.
#         features : list
#             List of feature names to plot.
#
#         Returns
#         -------
#         None
#         """
#
#     # Select a random point
#     random_index = np.random.randint(0, len(X))
#     random_point = X.iloc[random_index]
#
#     # Find the neighbors of the random point
#     neighbors = knn.kneighbors([random_point], return_distance=False)
#
#     # Plot each pair of features
#     num_features = len(features)
#     for i in range(num_features):
#         for j in range(i + 1, num_features):
#             plt.figure(figsize=(8, 6))
#             plt.scatter(X.iloc[:, i], X.iloc[:, j], c=y, cmap='viridis', marker='o', edgecolor='k', s=50)
#             plt.scatter(random_point[i], random_point[j], c='red', marker='x', s=200, label='Random Point')
#             plt.scatter(X.iloc[neighbors[0], i], X.iloc[neighbors[0], j], c='red', marker='o', edgecolor='k', s=100,
#                         facecolors='none', label='Neighbors')
#             plt.xlabel(features[i])
#             plt.ylabel(features[j])
#             plt.title(f'KNN Neighbors with {features[i]} vs {features[j]}')
#             plt.legend()
#             plt.savefig(
#                 f'/output/visualizations/knn_plots/{features[i]}_vs_{features[j]}_knn_neighbors.pdf'
#             )
