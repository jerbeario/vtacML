import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



def make_model(model_type, class_weights):
    model = None
    if model_type == 'RFClassifier':
        model = RandomForestClassifier(class_weight=class_weights)
    return model


def run_grid_search(X, y, model, param_grid, cv=5):
    grid_search = GridSearchCV(model,
                               param_grid=param_grid,
                               cv=cv,
                               verbose=2)
    grid_search.fit(X, y)
    return grid_search


def train_model(X_train, y_train, weights, model=None, config_path='./config/config.yaml'):
    config = yaml.safe_load(open(config_path, 'r'))
    model_type = config['Inputs']['model']
    if model is None:
        model = make_model(config['Inputs']['model'], class_weights=weights )
    print("SHAPE: ", X_train.shape, y_train.shape,)
    grid_search = run_grid_search(X_train, y_train, model, param_grid=config[model_type]['param_grid'])
    model.set_params(**grid_search.best_params_)
    model.fit(X_train, y_train)
    return model
