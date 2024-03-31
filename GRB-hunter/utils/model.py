from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def make_model(model):
    if model == 'RandomForestClassifier':
        return RandomForestClassifier()