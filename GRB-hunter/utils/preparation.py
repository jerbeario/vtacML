import os
import shutil
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class OneHotEncodercustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(
            handle_unknown='ignore')

    def fit(self, X, y=None):
        X_ = X.loc[:, self.variables]
        self.ohe.fit(X_)
        return self

    def transform(self, X):
        X_ = X.loc[:, self.variables]
        X_transformed = pd.DataFrame(self.ohe.transform(X_).toarray(),
                                     columns=self.ohe.get_feature_names_out())
        X.drop(self.variables, axis=1, inplace=True)
        X[self.ohe.get_feature_names_out()] = X_transformed[self.ohe.get_feature_names_out()].values
        return X


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, variables, strategy='drop'):
        self.variables = variables
        self.strategy = strategy
        self.fitted = False
        self.y = None

    def handle_nans(self, X, y=None):
        X_ = X[self.variables]
        if self.strategy == 'drop':
            #
            # data = data.dropna()
            # X_ = data[self.variables]
            # y_ = data.drop(columns=self.variables)
            pass  # Return the result of dropping missing values
        elif self.strategy == 'impute':

            # Impute missing values with mean
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(imputer.fit_transform(X_), columns=X_.columns, index=X_.index)

            # Implement imputation strategy if needed
            return X_imputed
        elif self.strategy == 'ignore':
            # Implement strategy to ignore missing values if needed
            pass
        else:
            raise ValueError("Invalid value for 'handle' argument.")

    def handle_outliers(self, X, y=None):

        return self

    def fit(self, X, y=None):
        self.y = y
        self.fitted = True
        return self  # Nothing to fit, just return self

    def transform(self, X, y=None):

        if not self.fitted:
            raise ValueError("The transformer has not been fitted yet. "
                             "Call 'fit' before 'transform'.")

        X_ = X.loc[:, self.variables]
        transformed_X = self.handle_nans(X_)  # Store the transformed data
        return transformed_X  # Return the transformed data


def get_data(path):
    data = pd.read_parquet(path=path, engine='fastparquet')
    return data


def get_class_weights(data, target_column):
    y = data[target_column]
    print(np.unique(y))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    weight_dict = dict(zip([0, 1], class_weights))
    return weight_dict


def plot_correlation(data, plot_path):
    if os.path.isdir(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path)

    plt.figure(figsize=(12, 12))
    corr_matrix = data.corr()
    sns.heatmap(abs(corr_matrix), cmap='Blues')
    plt.savefig(plot_path)
