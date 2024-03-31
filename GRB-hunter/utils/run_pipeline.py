import yaml
import logging
import joblib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from preparation import get_data, OneHotEncodercustom, Cleaner
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class VTACPipe:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path, 'r'))

        data_path = self.config['Inputs']['path']
        data = get_data(path=data_path)
        # still need to figure out the missing values, probably ohe then imputing but need to look at distributions
        data = data.dropna()

        X_columns = self.config['Inputs']['columns']
        y_columns = self.config['Inputs']['target_column']

        X = data[X_columns]
        y = data[y_columns]

        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = train_test_split(X, y,
                                         test_size=0.2,
                                         random_state=123)

        ohe = OneHotEncodercustom(variables=["EFLAG_B1"])
        cleaner = Cleaner(variables=X_columns,
                          strategy='impute'
                          )
        scaler = StandardScaler()
        normalizer = Normalizer()

        self.preprocessing = Pipeline(steps=[
            ('cleaner', cleaner),
            ('ohe', ohe),
            ('scaler', scaler),
            ('normalizer', normalizer)
        ],
            verbose=True)
        self.best_model = None
        self.y_predict = None

    def train(self, models, model_path=None):

        for name, model in models.items():
            param_grid = self.config[name]['param_grid']

            print("Model: {}".format(name))
            logging.info(self.preprocessing)

            full_pipeline = Pipeline(steps=self.preprocessing.steps.copy())
            full_pipeline.steps.append(
                (name, model)
            )
            logging.info(full_pipeline)
            # model fitting
            grid_search = GridSearchCV(full_pipeline, param_grid, verbose=2)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_

        if model_path is not None:
            joblib.dump(self.best_model, model_path)
            logging.info(f'Saved model to {model_path}')
        return self.best_model

    def predict(self, X):
        self.y_predict = self.best_model.predict(X)
        return self.y_predict

    def evaluate(self):
        pass


def run_pipeline():
    config = yaml.safe_load(open('/Users/jeremypalmerio/Repos/VTAC_ML/GRB-hunter/config/config.yaml', 'r'))

    data_path = config['Inputs']['path']
    data = get_data(path=data_path)
    # still need to figure out the missing values, probably ohe then imputing but need to look at distributions
    data = data.dropna()

    X_columns = config['Inputs']['columns']
    y_columns = config['Inputs']['target_column']

    X = data[X_columns]
    y = data[y_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123)
    knn = KNeighborsClassifier()
    rfc = RandomForestClassifier()
    ada = AdaBoostClassifier()

    models = {
        # "K-Nearest Neighbors": knn,
        "rfc": rfc,
        "ada": ada}

    ohe = OneHotEncodercustom(variables=["EFLAG_B1"])
    cleaner = Cleaner(variables=X_columns,
                      strategy='impute'
                      )
    scaler = StandardScaler()
    normalizer = Normalizer()
    preprocessing_pipe = Pipeline(steps=[
        ('cleaner', cleaner),
        ('ohe', ohe),
        ('scaler', scaler),
        ('normalizer', normalizer)
    ],
        verbose=True)

    for name, model in models.items():
        param_grid = config[name]['param_grid']

        print("Model: {}".format(name))
        complete_pipeline = Pipeline([
            ('cleaner', cleaner),
            ('ohe', ohe),
            ('scaler', scaler),
            ('normalizer', normalizer),
            (name, model)
        ])

        # model fitting
        grid_search = GridSearchCV(complete_pipeline, param_grid, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # model scoring
        train_pred = best_model.predict(X_train)
        valid_pred = best_model.predict(X_test)

        # Evaluate model performance
        print('*' * 50)
        print(f'{name} Training score:')
        print(
            f'MAE: {round(mean_absolute_error(y_train, train_pred), 4)} | RMSE: {round(mean_squared_error(y_train, train_pred, squared=False), 4)} | F1: {round(f1_score(y_train, train_pred), 4)}')
        print('-' * 20)
        print(f'{name} Validation score:')
        print(
            f'MAE: {round(mean_absolute_error(y_test, valid_pred), 4)} | RMSE: {round(mean_squared_error(y_test, valid_pred, squared=False), 4)} | F1: {round(f1_score(y_test, valid_pred), 4)}')




if __name__ == '__main__':
    # test1.py executed as script
    # do something
    # run_pipeline()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
