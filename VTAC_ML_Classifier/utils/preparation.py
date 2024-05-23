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


class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
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
    def __init__(self, variables):
        self.variables = variables
        self.fitted = False
        self.y = None

    def handle_nans(self, X, y=None):
        X_ = X.copy()
        for column in self.variables:
            if X[column].isna().sum():
                # Create the isna column
                new_column_name = f'{column}_isnan'
                X_[new_column_name] = X_[column].isna().astype(int)
                # Impute missing values with the mean
                impute_value = X_[column].median()
                X_[column].fillna(impute_value, inplace=True)

        return X_

    def clean_binary(self, X, col, y=None):
        X[col] = X[col].str.decode("utf-8").apply(lambda x: 1 if x else 0)
        return X

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
        if 'NEW_SRC' in self.variables:
            transformed_X = self.clean_binary(transformed_X, 'NEW_SRC')
        if 'DMAG_CAT' in self.variables:
            transformed_X = self.clean_binary(transformed_X, 'DMAG_CAT')
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
