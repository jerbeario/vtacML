""" Modified estimator for cleaning data in the classification pipeline."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    """
    Custum one-hot encoding estimator.
    """

    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X, y=None):
        """
        Fit the OneHotEncoder.
        """
        X_ = X.loc[:, self.variables]
        self.ohe.fit(X_)
        return self

    def transform(self, X):
        """
        Transform the OneHotEncoder.
        """
        X_ = X.loc[:, self.variables]
        X_transformed = pd.DataFrame(
            self.ohe.transform(X_).toarray(), columns=self.ohe.get_feature_names_out()
        )
        X.drop(self.variables, axis=1, inplace=True)
        X[self.ohe.get_feature_names_out()] = X_transformed[
            self.ohe.get_feature_names_out()
        ].values
        return X


class Cleaner(BaseEstimator, TransformerMixin):
    """Custom cleaning estimator for cleaning data in the classification."""

    def __init__(self, variables):
        self.variables = variables
        self.fitted = False
        self.y = None

    def handle_nans(self, X, y=None):
        """
        Handle NaN values by adding an extra _isnan column for every feature and imputing the missing values
        with the column mean.

        Parameters
        ----------
        X : pd.DataFrame
            Observation data to fit or predict with nans.
        y : pd.Series, default=None
            Predicted feature.

        Returns
        -------
        X_ : pd.DataFrame
            Observation data without nans and added _isnan column for every feature.

        """
        X_ = X.copy()
        for column in self.variables:
            column_isnan = f"{column}_isnan"
            X_[column_isnan] = X_[column].isna().astype(int)
            if X[column].isna().sum():
                impute_value = X_[column].median()
                X_[column].fillna(impute_value, inplace=True)
        return X_

    @staticmethod
    def clean_binary(X, col, y=None):
        """Clean the binary data for prediction."""
        X[col] = X[col].str.decode("utf-8").apply(lambda x: 1 if x else 0)
        return X

    def handle_outliers(self, X, y=None):
        """Not implemented yet."""

        return self

    def fit(self, X, y=None):
        """Not relevant"""
        self.y = y
        self.fitted = True
        return self  # Nothing to fit, just return self

    def transform(self, X, y=None):
        """
        Transform method when custom cleaner is called during the pipeline training and fitting.

        Parameters
        --------------
        X : pd.DataFrame
            Observation data to fit or predict with.
        y : pd.Series, default=None
            Predicted feature.

        Returns
        ---------------
        X : pd.DataFrame
            Cleaned observation data.
        """
        e_flags = [
            "EFLAG_R0",
            "EFLAG_R1",
            "EFLAG_R2",
            "EFLAG_R3",
            "EFLAG_B0",
            "EFLAG_B1",
            "EFLAG_B2",
            "EFLAG_B3",
        ]

        e_flags_in_cols = [x for x in self.variables if x in e_flags]

        if not self.fitted:
            raise ValueError(
                "The transformer has not been fitted yet. "
                "Call 'fit' before 'transform'."
            )

        X_ = X.loc[:, self.variables]
        transformed_X = self.handle_nans(X_)  # Store the transformed data

        if len(e_flags_in_cols) > 0:
            for e_flag in e_flags_in_cols:
                transformed_X[e_flag] = transformed_X[e_flag].fillna(-1)
        if "NEW_SRC" in self.variables:
            transformed_X = self.clean_binary(transformed_X, "NEW_SRC")
        if "DMAG_CAT" in self.variables:
            transformed_X = self.clean_binary(transformed_X, "DMAG_CAT")

        return transformed_X.dropna()  # Return the transformed data
